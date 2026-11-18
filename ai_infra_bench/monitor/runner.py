"""Background async job runner and live telemetry poller for the AIB platform."""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from ai_infra_bench.monitor.telemetry import (
    parse_prometheus_text,
    post_grafana_annotation,
    probe_service_info,
    scrape_metrics_point,
    summarize_telemetry_points,
)
from ai_infra_bench.utils.req import api_url, sanitize_url

if TYPE_CHECKING:
    from ai_infra_bench.monitor.shell import TargetRegistry
    from ai_infra_bench.monitor.storage import PlatformStore

logger = logging.getLogger(__name__)


class EventBroker:
    """Thread-safe Server-Sent Events (SSE) fan-out broker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[dict[str, Any] | None]] = set()

    def subscribe(self) -> queue.Queue[dict[str, Any] | None]:
        q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=128)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any] | None]) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def publish(self, event_type: str, data: Any) -> None:
        payload = {"event": event_type, "data": data, "ts": round(time.time(), 3)}
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except queue.Empty:
                    pass

    def close_all(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for q in subscribers:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass


class _JobProgressBar:
    """Drop-in `tqdm` replacement that streams progress to SQLite + SSE."""

    def __init__(
        self,
        total: int,
        on_progress: Callable[[dict[str, Any]], None],
        stage_prefix: str = "",
    ) -> None:
        self.total = max(int(total), 1)
        self.completed = 0
        self.stage = stage_prefix
        self.live_tps: float | None = None
        self._on_progress = on_progress
        self._last_emit = 0.0

    def set_description(self, desc: str) -> None:
        self.stage = desc
        self._emit(force=True)

    def set_postfix(self, info: Mapping[str, Any]) -> None:
        avg_str = str(info.get("Avg TPS") or "")
        if avg_str:
            try:
                self.live_tps = round(float(avg_str.split()[0]), 2)
            except ValueError:
                pass

    def update(self, n: int = 1) -> None:
        self.completed += n
        self._emit(force=self.completed >= self.total)

    def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_emit) < 0.35:
            return
        self._last_emit = now
        pct = min(99, int(round(100.0 * self.completed / self.total)))
        data: dict[str, Any] = {
            "completed": self.completed,
            "total": self.total,
            "pct": pct,
            "stage": self.stage,
        }
        if self.live_tps is not None:
            data["live_tps"] = self.live_tps
        self._on_progress(data)

    def close(self) -> None:
        return None

    def __enter__(self) -> "_JobProgressBar":
        return self

    def __exit__(self, *args: Any) -> bool:
        self.close()
        return False


class PlatformRunner:
    """Async execution engine for benchmarks, correctness checks, and live polling."""

    def __init__(
        self,
        store: "PlatformStore",
        registry: "TargetRegistry | None" = None,
        *,
        prometheus_url: str | None = None,
        grafana_url: str | None = None,
        poll_interval_s: float = 5.0,
    ) -> None:
        self.store = store
        self.registry = registry
        self.prometheus_url = prometheus_url
        self.grafana_url = grafana_url
        self.poll_interval_s = max(poll_interval_s, 1.0)
        self.events = EventBroker()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_async: asyncio.Event | None = None
        self._target_locks: dict[str, asyncio.Lock] = {}
        self._job_tasks: dict[str, asyncio.Task[Any]] = {}
        self._started_lock = threading.Lock()

    def start(self) -> None:
        with self._started_lock:
            if self._thread is not None:
                return
            ready = threading.Event()

            def _run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._stop_async = asyncio.Event()
                poll_task = loop.create_task(self._poll_targets_loop())
                ready.set()
                try:
                    loop.run_until_complete(self._stop_async.wait())
                finally:
                    poll_task.cancel()
                    for task in list(self._job_tasks.values()):
                        task.cancel()
                    loop.run_until_complete(
                        asyncio.gather(
                            poll_task, *self._job_tasks.values(), return_exceptions=True
                        )
                    )
                    loop.close()

            self._thread = threading.Thread(
                target=_run_loop, name="aib-platform-runner", daemon=True
            )
            self._thread.start()
            ready.wait(timeout=5.0)

    def stop(self) -> None:
        with self._started_lock:
            if self._thread is None or self._loop is None:
                return
            if self._stop_async is not None:
                self._loop.call_soon_threadsafe(self._stop_async.set)
            self.events.close_all()
            self._thread.join(timeout=5.0)
            self._thread = None
            self._loop = None

    def submit_job(
        self, session_id: str, job_type: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.start()
        job = self.store.create_job(session_id, job_type, params)
        self.events.publish("job_updated", job)
        assert self._loop is not None

        def _schedule() -> None:
            assert self._loop is not None
            task = self._loop.create_task(self._execute_job(job["job_id"]))
            self._job_tasks[job["job_id"]] = task
            task.add_done_callback(lambda _t: self._job_tasks.pop(job["job_id"], None))

        self._loop.call_soon_threadsafe(_schedule)
        return job

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        if self._loop is not None and job_id in self._job_tasks:
            self._loop.call_soon_threadsafe(self._job_tasks[job_id].cancel)
        updated = self.store.update_job(
            job_id,
            status="cancelled",
            append_log="Job cancelled by user.",
        )
        if updated:
            self.events.publish("job_updated", updated)
        return updated

    # ------------------------------------------------------------------
    # Background live metrics polling for active Pull targets
    # ------------------------------------------------------------------
    async def _poll_targets_loop(self) -> None:
        while self._stop_async is not None and not self._stop_async.is_set():
            if self.registry is not None:
                targets = self.registry.snapshot()
                pull_targets = [t for t in targets if t.mode == "pull"]
                if pull_targets:
                    await asyncio.gather(
                        *(self._poll_one_target(t) for t in pull_targets),
                        return_exceptions=True,
                    )
            try:
                await asyncio.wait_for(
                    self._stop_async.wait(), timeout=self.poll_interval_s
                )
            except asyncio.TimeoutError:
                pass

    async def _poll_one_target(self, target: Any) -> None:
        point = await asyncio.to_thread(scrape_metrics_point, target.metrics_url, 1.8)
        if point is not None:
            enriched = self.store.record_live_point(target.label, point)
            self.store.touch_target(target.label)
            self.events.publish(
                "monitor_tick", {"target": target.label, "point": enriched}
            )

    def _sample_target_now(self, session: Mapping[str, Any]) -> dict[str, Any] | None:
        target_label = session.get("target_label")
        if self.registry is not None and target_label and session.get("mode") == "push":
            pushed = self.registry.get_pushed_metrics(str(target_label))
            if pushed:
                pt = parse_prometheus_text(pushed)
                pt["ts"] = round(time.time(), 3)
                return pt
        metrics_url = session.get("metrics_url") or api_url(
            session["base_url"], "/metrics"
        )
        if str(metrics_url).startswith("push://"):
            return None
        return scrape_metrics_point(str(metrics_url), timeout=1.5)

    async def _window_sampler(
        self,
        session: Mapping[str, Any],
        stop_evt: asyncio.Event,
        samples: list[dict[str, Any]],
    ) -> None:
        while not stop_evt.is_set():
            pt = await asyncio.to_thread(self._sample_target_now, session)
            if pt is not None:
                samples.append(pt)
                if lbl := session.get("target_label"):
                    self.store.record_live_point(str(lbl), pt)
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    # Job Dispatcher
    # ------------------------------------------------------------------
    async def _execute_job(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        session = self.store.get_session(job["session_id"])
        if session is None:
            self.store.update_job(
                job_id, status="failed", error_message="Target session not found"
            )
            return

        lock_key = sanitize_url(session["base_url"])
        lock = self._target_locks.setdefault(lock_key, asyncio.Lock())
        if lock.locked():
            queued_job = self.store.update_job(
                job_id,
                progress={"pct": 0, "stage": "Waiting for active run on same backend"},
                append_log=f"Waiting for another active job on {lock_key} to finish...",
            )
            if queued_job:
                self.events.publish("job_updated", queued_job)

        async with lock:
            # Refresh server fingerprint if not yet populated.
            if not (session.get("server_info") or {}).get("reachable"):
                probed = await asyncio.to_thread(
                    probe_service_info,
                    session["base_url"],
                    job["params"].get("api_key"),
                )
                if probed.get("reachable"):
                    session = (
                        self.store.update_session(
                            session["session_id"], server_info=probed
                        )
                        or session
                    )

            running_job = self.store.update_job(
                job_id,
                status="running",
                progress={"pct": 2, "stage": f"Starting {job['job_type']}"},
                append_log=f"Started {job['job_type']} against {session['base_url']}",
            )
            if running_job:
                self.events.publish("job_updated", running_job)

            try:
                jtype = job["job_type"]
                params = job["params"]
                if jtype == "bench":
                    await self._run_bench_job(job_id, session, params)
                elif jtype == "slo":
                    await self._run_slo_job(job_id, session, params)
                elif jtype == "eval_dataset":
                    await self._run_eval_dataset_job(job_id, session, params)
                elif jtype == "eval_logits":
                    await self._run_eval_logits_job(job_id, session, params)
                elif jtype == "eval_hidden_states":
                    await self._run_eval_hidden_states_job(job_id, session, params)
                elif jtype == "suite":
                    await self._run_suite_job(job_id, session, params)
                else:
                    raise ValueError(f"Unsupported job_type: {jtype!r}")

                done_job = self.store.update_job(
                    job_id,
                    status="succeeded",
                    progress={"pct": 100, "stage": "Completed"},
                    append_log="Job finished successfully.",
                )
                if done_job:
                    self.events.publish("job_updated", done_job)
                    self.events.publish(
                        "session_updated",
                        self.store.get_session_detail(session["session_id"]),
                    )
            except asyncio.CancelledError:
                cancelled = self.store.update_job(
                    job_id,
                    status="cancelled",
                    append_log="Execution cancelled.",
                )
                if cancelled:
                    self.events.publish("job_updated", cancelled)
                raise
            except Exception as exc:
                logger.exception("Platform job %s failed", job_id)
                failed = self.store.update_job(
                    job_id,
                    status="failed",
                    error_message=str(exc),
                    append_log=f"ERROR: {exc}",
                )
                if failed:
                    self.events.publish("job_updated", failed)

    def _emit_job_progress(
        self, job_id: str, progress: Mapping[str, Any], log_line: str | None = None
    ) -> None:
        updated = self.store.update_job(job_id, progress=progress, append_log=log_line)
        if updated:
            self.events.publish("job_updated", updated)

    # ------------------------------------------------------------------
    # Concrete Runners (bench, slo, eval_dataset, eval_logits, hidden_states, suite)
    # ------------------------------------------------------------------
    def _resolve_model(
        self, session: Mapping[str, Any], params: Mapping[str, Any]
    ) -> str | None:
        return (
            params.get("model")
            or (session.get("server_info") or {}).get("model")
            or (session.get("labels") or {}).get("model")
        )

    async def _run_bench_job(
        self, job_id: str, session: Mapping[str, Any], params: Mapping[str, Any]
    ) -> None:
        from ai_infra_bench.performance.bench import (
            _create_bench_client_session,
            flush_cache,
            load_requests,
            request_url_for_payloads,
            run_requests,
        )
        from ai_infra_bench.performance.bench_utils import (
            BENCH_AIOHTTP_TIMEOUT_SECONDS,
            compute_shared_prefix_len,
            parse_goodput,
            prepare_warmup_requests,
            set_seed,
            validate_args,
        )
        from ai_infra_bench.performance.struct import _OutputStats

        dataset = params.get("dataset") or (
            None if params.get("prompt") or params.get("payload_path") else "random"
        )
        raw_qps = params.get("qps")
        qps_list = (
            [float(x) for x in raw_qps]
            if isinstance(raw_qps, list) and raw_qps
            else ([float(raw_qps)] if isinstance(raw_qps, (int, float)) else None)
        )
        raw_mc = params.get("max_concurrency")
        if isinstance(raw_mc, list) and raw_mc:
            mc_list: list[int | None] = [
                int(x) if x is not None else None for x in raw_mc
            ]
        elif isinstance(raw_mc, int):
            mc_list = [raw_mc]
        else:
            mc_list = [None] if qps_list else [32]

        goodput_raw = params.get("goodput")
        if isinstance(goodput_raw, str):
            goodput_items = [tok for tok in goodput_raw.split() if tok]
        elif isinstance(goodput_raw, list):
            goodput_items = [str(tok) for tok in goodput_raw if tok]
        else:
            goodput_items = None

        model = self._resolve_model(session, params)
        input_len = int(params.get("input_len") or 1024)
        output_len = int(params.get("output_len") or 256)
        num_requests = int(params.get("num_requests") or 64)
        num_warmup = int(
            params.get("num_warmup_requests")
            if params.get("num_warmup_requests") is not None
            else 5
        )
        cache_ratio = float(params.get("cache_ratio") or 0.0)
        range_ratio = float(params.get("random_range_ratio") or 1.0)
        repeat = int(params.get("repeat") or 1)
        burstiness = float(
            params.get("burstiness") if params.get("burstiness") is not None else 1.0
        )
        disable_flush = bool(params.get("disable_flush_cache", False))
        label = str(params.get("label") or session["name"])

        ns = argparse.Namespace(
            base_url=sanitize_url(session["base_url"]),
            api_key=str(params.get("api_key") or "EMPTY"),
            model=model,
            seed=int(params.get("seed") or 42),
            override_payload=params.get("override_payload"),
            dump_path=None,
            dump_content="all",
            dump_finish_reason_length=False,
            metric_path=None,
            report_to=None,
            label=label,
            goodput=goodput_items,
            num_requests=num_requests,
            num_warmup_requests=num_warmup,
            filter_constrained_grammar_requests=False,
            max_concurrency=mc_list,
            qps=qps_list,
            request_rate=float(params.get("request_rate") or float("inf")),
            burstiness=burstiness,
            request_timeout=float(
                params.get("request_timeout") or BENCH_AIOHTTP_TIMEOUT_SECONDS
            ),
            prompt=params.get("prompt"),
            payload_path=params.get("payload_path"),
            payload_regex_path=params.get("payload_path"),
            dataset=dataset,
            input_len=input_len,
            output_len=output_len,
            random_range_ratio=range_ratio,
            cache_ratio=cache_ratio,
            tokenizer=params.get("tokenizer"),
            repeat=repeat,
            with_ts=False,
            debug=False,
            disable_flush_cache=disable_flush,
        )
        validate_args(ns)
        set_seed(ns.seed)

        requests = await asyncio.to_thread(load_requests, ns)
        if ns.num_requests:
            requests = requests[: ns.num_requests]
        request_url = request_url_for_payloads(ns.base_url, requests)
        flush_endpoint = api_url(ns.base_url, "/flush_cache")
        prefix_len = compute_shared_prefix_len(ns.input_len, cache_ratio)
        warmup_reqs = prepare_warmup_requests(requests, ns)
        goodput_thresholds = parse_goodput(ns.goodput)

        if warmup_reqs and prefix_len == 0 and not ns.disable_flush_cache:
            self._emit_job_progress(
                job_id,
                {"pct": 5, "stage": f"Warmup ({len(warmup_reqs)} reqs)"},
                f"Running warmup ({len(warmup_reqs)} requests)...",
            )
            async with _create_bench_client_session(
                ns.max_concurrency[0], ns.api_key, ns.request_timeout
            ) as client:
                await run_requests(
                    client,
                    request_url,
                    warmup_reqs,
                    ns.model,
                    ns.override_payload,
                    semaphore=None,
                )

        qps_values = ns.qps if ns.qps else [ns.request_rate]
        total_rounds = len(qps_values) * len(ns.max_concurrency) * ns.repeat
        round_idx = 0
        workload_desc = f"{dataset or 'custom'} {input_len}/{output_len}" + (
            f" cache={cache_ratio:g}" if cache_ratio else ""
        )

        for req_rate in qps_values:
            for max_conc in ns.max_concurrency:
                for rep in range(ns.repeat):
                    round_idx += 1
                    step_tag = (
                        f"qps={req_rate:g}"
                        if max_conc is None
                        else f"c={max_conc}"
                        + (f", qps={req_rate:g}" if req_rate != float("inf") else "")
                    )
                    stage_name = f"bench [{round_idx}/{total_rounds}] ({step_tag})"
                    base_pct = int(10 + 85 * ((round_idx - 1) / max(total_rounds, 1)))

                    def _on_prog(
                        p: dict[str, Any],
                        _b: int = base_pct,
                        _tot: int = total_rounds,
                        _st: str = stage_name,
                    ) -> None:
                        scaled = _b + int((85 / max(_tot, 1)) * (p["pct"] / 100.0))
                        self._emit_job_progress(
                            job_id,
                            {**p, "pct": min(98, scaled), "stage": _st},
                        )

                    pbar = _JobProgressBar(len(requests), _on_prog, stage_name)
                    semaphore = (
                        asyncio.Semaphore(max_conc) if max_conc is not None else None
                    )
                    run_stats: dict[str, Any] = {}
                    extra_kw: dict[str, Any] = {}
                    if burstiness != 1.0:
                        extra_kw["burstiness"] = burstiness
                    if max_conc is None:
                        extra_kw["run_stats"] = run_stats

                    samples: list[dict[str, Any]] = []
                    stop_sample = asyncio.Event()
                    sampler_task: asyncio.Task[None] | None = None

                    async with _create_bench_client_session(
                        max_conc, ns.api_key, ns.request_timeout
                    ) as client:
                        if not ns.disable_flush_cache:
                            await flush_cache(client, flush_endpoint)
                            if prefix_len > 0 and warmup_reqs:
                                await run_requests(
                                    client,
                                    request_url,
                                    warmup_reqs,
                                    ns.model,
                                    ns.override_payload,
                                    semaphore=None,
                                )

                        t_start_unix = time.time()
                        sampler_task = asyncio.create_task(
                            self._window_sampler(session, stop_sample, samples)
                        )
                        try:
                            perf_start = time.perf_counter()
                            outputs = await run_requests(
                                client,
                                request_url,
                                requests,
                                ns.model,
                                ns.override_payload,
                                semaphore,
                                pbar,  # type: ignore[arg-type]
                                req_rate,
                                perf_start,
                                max_concurrency=max_conc,
                                **extra_kw,
                            )
                            duration_s = max(time.perf_counter() - perf_start, 1e-9)
                        finally:
                            stop_sample.set()
                            if sampler_task is not None:
                                await asyncio.gather(
                                    sampler_task, return_exceptions=True
                                )
                        t_end_unix = time.time()

                    # Capture one final scrape immediately after completion if window was short.
                    final_pt = await asyncio.to_thread(self._sample_target_now, session)
                    if final_pt is not None:
                        samples.append(final_pt)

                    telemetry_slice = summarize_telemetry_points(samples)
                    ts_str = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(t_end_unix)
                    )
                    yaml_record = _OutputStats(
                        outputs,
                        duration_s,
                        max_conc,
                        req_rate,
                        label=label,
                        ts=ts_str,
                        goodput_thresholds=goodput_thresholds,
                        peak_concurrency=run_stats.get("peak_concurrency"),
                    ).build_yaml_record()

                    saved_run = self.store.add_bench_run(
                        session_id=session["session_id"],
                        job_id=job_id,
                        subcommand="bench",
                        workload=workload_desc,
                        yaml_record=yaml_record,
                        telemetry=telemetry_slice,
                        t_start=t_start_unix,
                        t_end=t_end_unix,
                    )
                    out_tps = (yaml_record.get("summary") or {}).get(
                        "output_throughput_tps", 0.0
                    )
                    tpot_mean = (
                        (yaml_record.get("metrics") or {})
                        .get("tpot_ms", {})
                        .get("mean")
                    )
                    kv_peak = (telemetry_slice.get("summary") or {}).get(
                        "kv_cache_peak_pct"
                    )
                    self._emit_job_progress(
                        job_id,
                        {
                            "pct": min(98, int(10 + 85 * (round_idx / total_rounds))),
                            "stage": stage_name,
                        },
                        f"Finished {step_tag} (rep {rep + 1}): "
                        f"output_tps={out_tps} tok/s, tpot_mean={tpot_mean} ms"
                        + (f", kv_peak={kv_peak}%" if kv_peak is not None else ""),
                    )
                    self.events.publish("bench_run_added", saved_run)

                    await asyncio.to_thread(
                        post_grafana_annotation,
                        self.grafana_url,
                        time_ms=int(t_start_unix * 1000),
                        time_end_ms=int(t_end_unix * 1000),
                        tags=[
                            "aib-bench",
                            str(session.get("target_label") or ""),
                            label,
                        ],
                        text=f"aib bench {step_tag} ({workload_desc}) -> {out_tps} tok/s",
                    )

    async def _run_slo_job(
        self, job_id: str, session: Mapping[str, Any], params: Mapping[str, Any]
    ) -> None:
        from ai_infra_bench.slo import parse_slo_config, run_slo

        model = self._resolve_model(session, params)
        slo_dict = {
            "endpoint": {
                "base_url": sanitize_url(session["base_url"]),
                "api_key": str(params.get("api_key") or "EMPTY"),
                "model": model,
            },
            "request": {
                "input_len": int(params.get("input_len") or 1024),
                "output_len": int(params.get("output_len") or 256),
            },
            "benchmark": {
                "num_requests": int(params.get("num_requests") or 50),
                "warmup_requests": int(params.get("warmup_requests") or 3),
                "seed": int(params.get("seed") or 42),
            },
            "search": {
                "parameter": str(params.get("parameter") or "qps"),
                "min": int(params.get("min") or 1),
                "max": int(params.get("max") or 32),
            },
            "conditions": params.get("conditions")
            or [
                {"metric": "success_rate", "operator": ">=", "value": 0.99},
                {"metric": "p99_ttft_ms", "operator": "<=", "value": 800.0},
            ],
        }
        config = parse_slo_config(slo_dict)
        t_start = time.time()
        self._emit_job_progress(
            job_id,
            {"pct": 25, "stage": f"SLO binary search ({config.search.parameter})"},
            f"Searching {config.search.parameter} in [{config.search.minimum}, {config.search.maximum}]...",
        )
        result = await run_slo(config)
        t_end = time.time()

        best = result.get("best_value")
        best_probe = next(
            (r for r in reversed(result.get("runs") or []) if r.get("passed")),
            (result.get("runs") or [{}])[-1],
        )
        probe_metrics = best_probe.get("metrics") or {}
        yaml_rec = {
            "label": str(params.get("label") or session["name"]),
            "mode": "slo",
            "max_concurrency": probe_metrics.get("max_concurrency"),
            "request_rate": probe_metrics.get("request_rate"),
            "summary": {
                "slo_status": result.get("status"),
                "slo_parameter": result.get("search_parameter"),
                "slo_best_value": best,
                "output_throughput_tps": round(
                    float(probe_metrics.get("output_throughput") or 0.0), 2
                ),
                "total_token_throughput_tps": round(
                    float(probe_metrics.get("total_token_throughput") or 0.0), 2
                ),
                "request_throughput_rps": round(
                    float(probe_metrics.get("request_throughput") or 0.0), 4
                ),
                "probes": result.get("runs"),
            },
            "metrics": {},
        }
        self.store.add_bench_run(
            session_id=session["session_id"],
            job_id=job_id,
            subcommand="slo",
            workload=f"slo:{config.search.parameter}[{config.search.minimum}..{config.search.maximum}]",
            yaml_record=yaml_rec,
            t_start=t_start,
            t_end=t_end,
        )
        self._emit_job_progress(
            job_id,
            {"pct": 95, "stage": "SLO search done"},
            f"SLO result: status={result.get('status')}, best {config.search.parameter}={best}",
        )

    async def _run_eval_dataset_job(
        self, job_id: str, session: Mapping[str, Any], params: Mapping[str, Any]
    ) -> None:
        from ai_infra_bench.correctness.eval_dataset import base as eval_base

        evals_raw = params.get("evals") or ["gsm8k"]
        evals = (
            [str(x) for x in evals_raw]
            if isinstance(evals_raw, list)
            else [str(evals_raw)]
        )
        model = self._resolve_model(session, params)
        ns = argparse.Namespace(
            base_url=sanitize_url(session["base_url"]),
            api_key=str(params.get("api_key") or "EMPTY"),
            model=model,
            override_payload=params.get("override_payload"),
            max_concurrency=int(params.get("max_concurrency") or 32),
            repeat=int(params.get("repeat") or 1),
            evals=evals,
            config=params.get("config"),
            dataset_path=params.get("dataset_path"),
            num_shots=int(
                params.get("num_shots") if params.get("num_shots") is not None else 5
            ),
            num_questions=(
                int(params["num_questions"])
                if params.get("num_questions") is not None
                else None
            ),
        )
        self._emit_job_progress(
            job_id,
            {"pct": 12, "stage": f"Loading datasets: {', '.join(evals)}"},
            f"Initializing eval-dataset for {evals}...",
        )
        runtime = await asyncio.to_thread(eval_base.EvalRuntime, ns)
        total_q = runtime.repeat * sum(e.get_length() for e in runtime.evals)

        orig_tqdm = eval_base.tqdm
        try:
            eval_base.tqdm = lambda total=total_q, **_kw: _JobProgressBar(  # type: ignore[assignment]
                total,
                lambda p: self._emit_job_progress(job_id, p),
                "eval-dataset",
            )
            await runtime.run()
        finally:
            eval_base.tqdm = orig_tqdm

        for rec in getattr(runtime, "recorded_runs", []):
            saved = self.store.add_correctness_run(
                session_id=session["session_id"],
                job_id=job_id,
                eval_type="dataset",
                suite_name=rec["suite_name"],
                passed=rec["passed"],
                correct_rate=rec["correct_rate"],
                wrong_rate=rec["wrong_rate"],
                failed_rate=rec["failed_rate"],
                truncated_rate=rec["truncated_rate"],
                summary=rec["summary"],
                cases=rec.get("cases"),
                t_start=rec.get("t_start"),
                t_end=rec.get("t_end"),
            )
            self._emit_job_progress(
                job_id,
                {"pct": 95, "stage": f"Saved {rec['suite_name']}"},
                f"Eval {rec['suite_name']}: correct={rec['correct_rate']:.2%}, "
                f"truncated={rec['truncated_rate']:.2%}",
            )
            self.events.publish("correctness_run_added", saved)

    async def _run_eval_logits_job(
        self, job_id: str, session: Mapping[str, Any], params: Mapping[str, Any]
    ) -> None:
        from ai_infra_bench.correctness import logits
        from ai_infra_bench.correctness.common import fetch_chat_completions

        ref_url = sanitize_url(str(params.get("ref_base_url") or session["base_url"]))
        tgt_url = sanitize_url(session["base_url"])
        rtol = float(params.get("rtol") if params.get("rtol") is not None else 0.0)
        atol = float(params.get("atol") if params.get("atol") is not None else 1e-2)
        max_tokens = int(params.get("max_tokens") or 8)
        top_k = int(params.get("top_logprobs") or 3)
        prompt = str(params.get("prompt") or logits.PROMPT)
        model = self._resolve_model(session, params)

        ns = argparse.Namespace(
            base_urls=[ref_url, tgt_url],
            api_key=str(params.get("api_key") or "EMPTY"),
            model=model,
            override_payload=params.get("override_payload"),
            prompt=prompt,
            max_tokens=max_tokens,
            top_logprobs=top_k,
            rtol=rtol,
            atol=atol,
        )
        payload = logits.build_payload(ns)
        t_start = time.time()
        self._emit_job_progress(
            job_id,
            {"pct": 30, "stage": "eval-logits"},
            f"Comparing logprobs: ref={ref_url} vs target={tgt_url} (atol={atol:g})",
        )
        responses = await fetch_chat_completions(
            [ref_url, tgt_url], payload, ns.api_key
        )
        ref_res = logits.extract_logprobs(ref_url, responses[0])
        tgt_res = logits.extract_logprobs(tgt_url, responses[1])
        cmp_res = logits.compare_results(ref_res, tgt_res, rtol, atol, True)
        serialized = logits.serialize_comparison(cmp_res, rtol=rtol, atol=atol)
        t_end = time.time()

        positions = serialized.pop("positions", [])
        saved = self.store.add_correctness_run(
            session_id=session["session_id"],
            job_id=job_id,
            eval_type="logits",
            suite_name="eval-logits",
            passed=serialized["passed"],
            correct_rate=serialized["token_match_rate"],
            ref_base_url=ref_url,
            summary={**serialized, "prompt": prompt},
            cases=positions,
            t_start=t_start,
            t_end=t_end,
        )
        self._emit_job_progress(
            job_id,
            {"pct": 95, "stage": "eval-logits done"},
            f"Logits check {'PASSED' if serialized['passed'] else 'FAILED'} "
            f"(max |Δ| = {serialized['max_abs_diff']})",
        )
        self.events.publish("correctness_run_added", saved)

    async def _run_eval_hidden_states_job(
        self, job_id: str, session: Mapping[str, Any], params: Mapping[str, Any]
    ) -> None:
        from ai_infra_bench.correctness import hidden_states
        from ai_infra_bench.correctness.common import fetch_chat_completions

        ref_url = sanitize_url(str(params.get("ref_base_url") or session["base_url"]))
        tgt_url = sanitize_url(session["base_url"])
        rtol = float(params.get("rtol") if params.get("rtol") is not None else 1e-2)
        atol = float(params.get("atol") if params.get("atol") is not None else 1e-2)
        max_tokens = int(params.get("max_tokens") or 1)
        prompt = str(params.get("prompt") or hidden_states.PROMPT)
        model = self._resolve_model(session, params)

        ns = argparse.Namespace(
            base_urls=[ref_url, tgt_url],
            api_key=str(params.get("api_key") or "EMPTY"),
            model=model,
            override_payload=params.get("override_payload"),
            prompt=prompt,
            max_tokens=max_tokens,
            rtol=rtol,
            atol=atol,
        )
        payload = hidden_states.build_payload(ns)
        t_start = time.time()
        self._emit_job_progress(
            job_id,
            {"pct": 30, "stage": "eval-hidden-states"},
            f"Comparing hidden states: ref={ref_url} vs target={tgt_url}",
        )
        responses = await fetch_chat_completions(
            [ref_url, tgt_url], payload, ns.api_key
        )
        ref_res = hidden_states.extract_hidden_states(ref_url, responses[0])
        tgt_res = hidden_states.extract_hidden_states(tgt_url, responses[1])
        cmp_res = hidden_states.compare_results(ref_res, tgt_res, rtol, atol)
        serialized = hidden_states.serialize_comparison(cmp_res, rtol=rtol, atol=atol)
        t_end = time.time()

        saved = self.store.add_correctness_run(
            session_id=session["session_id"],
            job_id=job_id,
            eval_type="hidden_states",
            suite_name="eval-hidden-states",
            passed=serialized["passed"],
            correct_rate=serialized["close_ratio"],
            ref_base_url=ref_url,
            summary={**serialized, "prompt": prompt},
            t_start=t_start,
            t_end=t_end,
        )
        self._emit_job_progress(
            job_id,
            {"pct": 95, "stage": "eval-hidden-states done"},
            f"Hidden states {'PASSED' if serialized['passed'] else 'FAILED'} "
            f"(cosine={serialized['cosine_similarity']:.6f}, max|Δ|={serialized['max_abs_diff']})",
        )
        self.events.publish("correctness_run_added", saved)

    async def _run_suite_job(
        self, job_id: str, session: Mapping[str, Any], params: Mapping[str, Any]
    ) -> None:
        steps = params.get("steps") or []
        if not isinstance(steps, list) or not steps:
            raise ValueError("suite job requires a non-empty 'steps' list")
        for idx, step in enumerate(steps, start=1):
            stype = step.get("type")
            sparams = step.get("params") or {}
            self.store.update_job(
                job_id,
                append_log=f"=== Suite Step [{idx}/{len(steps)}]: {stype} ===",
            )
            if stype == "eval_logits":
                await self._run_eval_logits_job(job_id, session, sparams)
            elif stype == "eval_hidden_states":
                await self._run_eval_hidden_states_job(job_id, session, sparams)
            elif stype == "eval_dataset":
                await self._run_eval_dataset_job(job_id, session, sparams)
            elif stype == "bench":
                await self._run_bench_job(job_id, session, sparams)
            elif stype == "slo":
                await self._run_slo_job(job_id, session, sparams)
            else:
                raise ValueError(f"Unknown suite step type: {stype!r}")
