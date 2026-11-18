"""SQLite (WAL) persistence and comparison store for the AIB platform."""

from __future__ import annotations

import collections
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Deque, Mapping, Sequence

from ai_infra_bench.utils.req import sanitize_url

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS service_sessions (
    session_id TEXT PRIMARY KEY,
    target_label TEXT,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    metrics_url TEXT,
    mode TEXT NOT NULL DEFAULT 'pull',
    status TEXT NOT NULL DEFAULT 'online',
    is_baseline INTEGER NOT NULL DEFAULT 0,
    labels_json TEXT NOT NULL DEFAULT '{}',
    server_info_json TEXT NOT NULL DEFAULT '{}',
    hardware_info_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT '',
    started_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    stopped_at REAL
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON service_sessions(status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_target ON service_sessions(target_label, status);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    params_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NOT NULL DEFAULT '{}',
    logs_text TEXT NOT NULL DEFAULT '',
    error_message TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    FOREIGN KEY(session_id) REFERENCES service_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS bench_runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT,
    session_id TEXT NOT NULL,
    subcommand TEXT NOT NULL DEFAULT 'bench',
    label TEXT NOT NULL DEFAULT '',
    workload TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT 'concurrency',
    max_concurrency INTEGER,
    peak_concurrency INTEGER,
    request_rate REAL,
    summary_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    spec_tokens_json TEXT NOT NULL DEFAULT '{}',
    finish_reasons_json TEXT NOT NULL DEFAULT '{}',
    telemetry_json TEXT NOT NULL DEFAULT '{}',
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES service_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_bench_session ON bench_runs(session_id, created_at ASC);

CREATE TABLE IF NOT EXISTS correctness_runs (
    eval_id TEXT PRIMARY KEY,
    job_id TEXT,
    session_id TEXT NOT NULL,
    eval_type TEXT NOT NULL,
    suite_name TEXT NOT NULL,
    passed INTEGER NOT NULL DEFAULT 0,
    correct_rate REAL,
    wrong_rate REAL,
    failed_rate REAL,
    truncated_rate REAL,
    ref_base_url TEXT,
    summary_json TEXT NOT NULL DEFAULT '{}',
    cases_file TEXT,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES service_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eval_session ON correctness_runs(session_id, created_at ASC);
"""


def _new_id(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def _json_loads(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except ValueError:
        return fallback


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _default_session_name(
    base_url: str,
    labels: Mapping[str, Any],
    server_info: Mapping[str, Any],
    target_label: str | None,
) -> str:
    model = (
        labels.get("model")
        or server_info.get("model")
        or (server_info.get("server_args") or {}).get("served_model_name")
        or (server_info.get("server_args") or {}).get("model_path")
    )
    if isinstance(model, str) and "/" in model:
        model = model.rstrip("/").rsplit("/", 1)[-1]
    cluster = labels.get("cluster") or labels.get("env")
    parts = [p for p in (model, cluster, target_label) if p]
    if parts:
        return " · ".join(str(p) for p in parts)
    return base_url


class PlatformStore:
    """Thread-safe SQLite store for sessions, jobs, benchmarks, and evaluations."""

    def __init__(self, db_path: Path, artifacts_dir: Path | None = None):
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = (
            artifacts_dir.expanduser().resolve()
            if artifacts_dir is not None
            else self.db_path.parent / "artifacts"
        )
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._live_buffers: dict[str, Deque[dict[str, Any]]] = {}
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(_SCHEMA_SQL)
            # Recover any orphaned 'queued'/'running' jobs from a prior crash.
            self._conn.execute(
                "UPDATE jobs SET status='failed', "
                "error_message='Interrupted by server restart', "
                "finished_at=? WHERE status IN ('queued', 'running')",
                (time.time(),),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Live in-memory rolling metrics buffer (fast UI & non-Prom fallback)
    # ------------------------------------------------------------------
    def record_live_point(
        self, target_label: str, point: Mapping[str, Any], maxlen: int = 90
    ) -> dict[str, Any]:
        with self._lock:
            buf = self._live_buffers.get(target_label)
            if buf is None:
                buf = collections.deque(maxlen=maxlen)
                self._live_buffers[target_label] = buf
            item = dict(point)
            item.setdefault("ts", round(time.time(), 3))

            # Derive instantaneous gen_throughput_tps from counter delta if needed.
            if item.get("gen_throughput_tps") is None and buf:
                prev = buf[-1]
                cur_tok = item.get("gen_tokens_total")
                prev_tok = prev.get("gen_tokens_total")
                dt = float(item["ts"]) - float(prev.get("ts") or 0.0)
                if (
                    isinstance(cur_tok, (int, float))
                    and isinstance(prev_tok, (int, float))
                    and dt > 0.1
                    and cur_tok >= prev_tok
                ):
                    item["gen_throughput_tps"] = round((cur_tok - prev_tok) / dt, 2)

            buf.append(item)
            return item

    def get_live_series(
        self, target_label: str, limit: int = 60
    ) -> list[dict[str, Any]]:
        with self._lock:
            buf = self._live_buffers.get(target_label)
            if not buf:
                return []
            items = list(buf)[-limit:]
            return items

    def get_latest_point(self, target_label: str) -> dict[str, Any] | None:
        with self._lock:
            buf = self._live_buffers.get(target_label)
            return dict(buf[-1]) if buf else None

    # ------------------------------------------------------------------
    # Service Sessions
    # ------------------------------------------------------------------
    def _deserialize_session(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "target_label": row["target_label"],
            "name": row["name"],
            "base_url": row["base_url"],
            "metrics_url": row["metrics_url"],
            "mode": row["mode"],
            "status": row["status"],
            "is_baseline": bool(row["is_baseline"]),
            "labels": _json_loads(row["labels_json"], {}),
            "server_info": _json_loads(row["server_info_json"], {}),
            "hardware_info": _json_loads(row["hardware_info_json"], {}),
            "notes": row["notes"] or "",
            "started_at": row["started_at"],
            "last_seen_at": row["last_seen_at"],
            "stopped_at": row["stopped_at"],
        }

    def upsert_online_session(
        self,
        *,
        base_url: str,
        target_label: str | None = None,
        metrics_url: str | None = None,
        mode: str = "pull",
        labels: Mapping[str, Any] | None = None,
        server_info: Mapping[str, Any] | None = None,
        hardware_info: Mapping[str, Any] | None = None,
        name: str | None = None,
        force_new: bool = False,
    ) -> dict[str, Any]:
        normalized_url = (
            base_url if base_url.startswith("push://") else sanitize_url(base_url)
        )
        now = time.time()
        lbl_dict = {str(k): str(v) for k, v in (labels or {}).items()}
        srv_dict = dict(server_info or {})
        hw_dict = dict(hardware_info or {})

        with self._lock:
            existing_row = None
            if not force_new:
                if target_label:
                    existing_row = self._conn.execute(
                        "SELECT * FROM service_sessions "
                        "WHERE target_label=? AND status='online' "
                        "ORDER BY started_at DESC LIMIT 1",
                        (target_label,),
                    ).fetchone()
                if existing_row is None:
                    existing_row = self._conn.execute(
                        "SELECT * FROM service_sessions "
                        "WHERE base_url=? AND status='online' "
                        "ORDER BY started_at DESC LIMIT 1",
                        (normalized_url,),
                    ).fetchone()

            if existing_row is not None and not force_new:
                cur = self._deserialize_session(existing_row)
                merged_labels = {**cur["labels"], **lbl_dict}
                merged_srv = {**cur["server_info"], **srv_dict}
                merged_hw = {**cur["hardware_info"], **hw_dict}
                updated_name = name or cur["name"]
                if (
                    not name
                    and cur["name"] in {cur["base_url"], cur["target_label"]}
                    and (merged_labels or merged_srv)
                ):
                    updated_name = _default_session_name(
                        normalized_url, merged_labels, merged_srv, target_label
                    )
                self._conn.execute(
                    "UPDATE service_sessions SET "
                    "target_label=COALESCE(?, target_label), "
                    "name=?, base_url=?, "
                    "metrics_url=COALESCE(?, metrics_url), "
                    "mode=?, labels_json=?, server_info_json=?, "
                    "hardware_info_json=?, last_seen_at=? "
                    "WHERE session_id=?",
                    (
                        target_label,
                        updated_name,
                        normalized_url,
                        metrics_url,
                        mode,
                        _json_dumps(merged_labels),
                        _json_dumps(merged_srv),
                        _json_dumps(merged_hw),
                        now,
                        cur["session_id"],
                    ),
                )
                self._conn.commit()
                return self.get_session(cur["session_id"])  # type: ignore[return-value]

            session_id = _new_id("sess")
            resolved_name = name or _default_session_name(
                normalized_url, lbl_dict, srv_dict, target_label
            )
            self._conn.execute(
                "INSERT INTO service_sessions ("
                "session_id, target_label, name, base_url, metrics_url, mode, "
                "status, is_baseline, labels_json, server_info_json, "
                "hardware_info_json, notes, started_at, last_seen_at, stopped_at"
                ") VALUES (?, ?, ?, ?, ?, ?, 'online', 0, ?, ?, ?, '', ?, ?, NULL)",
                (
                    session_id,
                    target_label,
                    resolved_name,
                    normalized_url,
                    metrics_url,
                    mode,
                    _json_dumps(lbl_dict),
                    _json_dumps(srv_dict),
                    _json_dumps(hw_dict),
                    now,
                    now,
                ),
            )
            self._conn.commit()
            return self.get_session(session_id)  # type: ignore[return-value]

    def mark_targets_offline(self, target_labels: Sequence[str]) -> int:
        if not target_labels:
            return 0
        now = time.time()
        placeholders = ",".join("?" for _ in target_labels)
        with self._lock:
            for lbl in target_labels:
                self._live_buffers.pop(lbl, None)
            cur = self._conn.execute(
                f"UPDATE service_sessions SET status='offline', stopped_at=?, last_seen_at=? "
                f"WHERE status='online' AND target_label IN ({placeholders})",
                (now, now, *target_labels),
            )
            self._conn.commit()
            return cur.rowcount

    def touch_target(self, target_label: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE service_sessions SET last_seen_at=? "
                "WHERE target_label=? AND status='online'",
                (time.time(), target_label),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM service_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            return self._deserialize_session(row) if row else None

    def get_online_session_for_target(self, target_label: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM service_sessions "
                "WHERE target_label=? AND status='online' "
                "ORDER BY started_at DESC LIMIT 1",
                (target_label,),
            ).fetchone()
            return self._deserialize_session(row) if row else None

    def list_sessions(
        self, *, status: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM service_sessions WHERE status=? "
                    "ORDER BY started_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM service_sessions ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            sessions = [self._deserialize_session(r) for r in rows]
            if not sessions:
                return []

            ids = [s["session_id"] for s in sessions]
            placeholders = ",".join("?" for _ in ids)
            bench_counts = dict(
                self._conn.execute(
                    f"SELECT session_id, COUNT(*) FROM bench_runs "
                    f"WHERE session_id IN ({placeholders}) GROUP BY session_id",
                    ids,
                ).fetchall()
            )
            eval_counts = dict(
                self._conn.execute(
                    f"SELECT session_id, COUNT(*) FROM correctness_runs "
                    f"WHERE session_id IN ({placeholders}) GROUP BY session_id",
                    ids,
                ).fetchall()
            )
            # Attach quick summary KPIs (peak output tps & latest eval scores)
            peak_rows = self._conn.execute(
                f"SELECT session_id, summary_json FROM bench_runs "
                f"WHERE session_id IN ({placeholders})",
                ids,
            ).fetchall()
            peaks: dict[str, float] = {}
            for r in peak_rows:
                summary = _json_loads(r["summary_json"], {})
                out_tps = summary.get("output_throughput_tps")
                if isinstance(out_tps, (int, float)):
                    sid = r["session_id"]
                    peaks[sid] = max(peaks.get(sid, 0.0), float(out_tps))

            eval_rows = self._conn.execute(
                f"SELECT session_id, eval_type, suite_name, passed, correct_rate "
                f"FROM correctness_runs WHERE session_id IN ({placeholders}) "
                f"ORDER BY created_at ASC",
                ids,
            ).fetchall()
            eval_briefs: dict[str, dict[str, Any]] = {}
            for r in eval_rows:
                sid = r["session_id"]
                b = eval_briefs.setdefault(sid, {})
                b[r["suite_name"]] = {
                    "eval_type": r["eval_type"],
                    "passed": bool(r["passed"]),
                    "correct_rate": r["correct_rate"],
                }

            for s in sessions:
                sid = s["session_id"]
                s["bench_count"] = bench_counts.get(sid, 0)
                s["eval_count"] = eval_counts.get(sid, 0)
                s["peak_output_tps"] = peaks.get(sid)
                s["eval_brief"] = eval_briefs.get(sid, {})
            return sessions

    def get_session_detail(self, session_id: str) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        session["bench_runs"] = self.list_bench_runs(session_id=session_id)
        session["correctness_runs"] = self.list_correctness_runs(session_id=session_id)
        return session

    def update_session(
        self,
        session_id: str,
        *,
        name: str | None = None,
        notes: str | None = None,
        labels: Mapping[str, Any] | None = None,
        server_info: Mapping[str, Any] | None = None,
        hardware_info: Mapping[str, Any] | None = None,
        is_baseline: bool | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            cur = self.get_session(session_id)
            if cur is None:
                return None
            if is_baseline is True:
                # Clear previous global baseline flag if setting a new primary baseline.
                self._conn.execute("UPDATE service_sessions SET is_baseline=0")
            new_name = name if name is not None else cur["name"]
            new_notes = notes if notes is not None else cur["notes"]
            new_labels = (
                {**cur["labels"], **{str(k): str(v) for k, v in labels.items()}}
                if labels is not None
                else cur["labels"]
            )
            new_srv = (
                {**cur["server_info"], **dict(server_info)}
                if server_info is not None
                else cur["server_info"]
            )
            new_hw = (
                {**cur["hardware_info"], **dict(hardware_info)}
                if hardware_info is not None
                else cur["hardware_info"]
            )
            new_base = (
                int(bool(is_baseline))
                if is_baseline is not None
                else int(cur["is_baseline"])
            )
            self._conn.execute(
                "UPDATE service_sessions SET name=?, notes=?, labels_json=?, "
                "server_info_json=?, hardware_info_json=?, is_baseline=? "
                "WHERE session_id=?",
                (
                    new_name,
                    new_notes,
                    _json_dumps(new_labels),
                    _json_dumps(new_srv),
                    _json_dumps(new_hw),
                    new_base,
                    session_id,
                ),
            )
            self._conn.commit()
            return self.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            eval_rows = self._conn.execute(
                "SELECT cases_file FROM correctness_runs WHERE session_id=?",
                (session_id,),
            ).fetchall()
            cur = self._conn.execute(
                "DELETE FROM service_sessions WHERE session_id=?", (session_id,)
            )
            self._conn.commit()
            for r in eval_rows:
                if r["cases_file"]:
                    (self.artifacts_dir / r["cases_file"]).unlink(missing_ok=True)
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    def create_job(
        self, session_id: str, job_type: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        job_id = _new_id("job")
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs ("
                "job_id, session_id, job_type, status, params_json, progress_json, "
                "logs_text, error_message, created_at, started_at, finished_at"
                ") VALUES (?, ?, ?, 'queued', ?, ?, '', NULL, ?, NULL, NULL)",
                (
                    job_id,
                    session_id,
                    job_type,
                    _json_dumps(dict(params)),
                    _json_dumps({"pct": 0, "stage": "queued"}),
                    now,
                ),
            )
            self._conn.commit()
            return self.get_job(job_id)  # type: ignore[return-value]

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: Mapping[str, Any] | None = None,
        append_log: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            new_status = status or row["status"]
            started_at = row["started_at"]
            finished_at = row["finished_at"]
            if new_status == "running" and started_at is None:
                started_at = now
            if (
                new_status in {"succeeded", "failed", "cancelled"}
                and finished_at is None
            ):
                finished_at = now
            cur_prog = _json_loads(row["progress_json"], {})
            if progress is not None:
                cur_prog.update(progress)
            logs = row["logs_text"] or ""
            if append_log:
                stamp = time.strftime("%H:%M:%S", time.localtime(now))
                line = f"[{stamp}] {append_log.rstrip()}\n"
                logs = (logs + line)[-64000:]
            err = error_message if error_message is not None else row["error_message"]
            self._conn.execute(
                "UPDATE jobs SET status=?, progress_json=?, logs_text=?, "
                "error_message=?, started_at=?, finished_at=? WHERE job_id=?",
                (
                    new_status,
                    _json_dumps(cur_prog),
                    logs,
                    err,
                    started_at,
                    finished_at,
                    job_id,
                ),
            )
            self._conn.commit()
            return self.get_job(job_id)

    def _deserialize_job(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "session_id": row["session_id"],
            "job_type": row["job_type"],
            "status": row["status"],
            "params": _json_loads(row["params_json"], {}),
            "progress": _json_loads(row["progress_json"], {}),
            "logs_text": row["logs_text"] or "",
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return self._deserialize_job(row) if row else None

    def list_jobs(
        self, *, session_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._lock:
            if session_id:
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [self._deserialize_job(r) for r in rows]

    # ------------------------------------------------------------------
    # Benchmark Runs
    # ------------------------------------------------------------------
    def add_bench_run(
        self,
        *,
        session_id: str,
        yaml_record: Mapping[str, Any],
        job_id: str | None = None,
        subcommand: str = "bench",
        workload: str = "",
        telemetry: Mapping[str, Any] | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> dict[str, Any]:
        run_id = _new_id("bench")
        now = time.time()
        raw_mc = yaml_record.get("max_concurrency")
        max_conc = int(raw_mc) if isinstance(raw_mc, int) else None
        raw_peak = yaml_record.get("peak_concurrency")
        peak_conc = int(raw_peak) if isinstance(raw_peak, int) else None
        raw_rr = yaml_record.get("request_rate")
        req_rate = (
            float(raw_rr)
            if isinstance(raw_rr, (int, float)) and raw_rr != float("inf")
            else None
        )

        with self._lock:
            self._conn.execute(
                "INSERT INTO bench_runs ("
                "run_id, job_id, session_id, subcommand, label, workload, mode, "
                "max_concurrency, peak_concurrency, request_rate, summary_json, "
                "metrics_json, spec_tokens_json, finish_reasons_json, telemetry_json, "
                "t_start, t_end, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    job_id,
                    session_id,
                    subcommand,
                    str(yaml_record.get("label") or ""),
                    workload,
                    str(yaml_record.get("mode") or "concurrency"),
                    max_conc,
                    peak_conc,
                    req_rate,
                    _json_dumps(yaml_record.get("summary") or {}),
                    _json_dumps(yaml_record.get("metrics") or {}),
                    _json_dumps(yaml_record.get("spec_tokens") or {}),
                    _json_dumps(yaml_record.get("finish_reasons") or {}),
                    _json_dumps(dict(telemetry or {})),
                    t_start if t_start is not None else now,
                    t_end if t_end is not None else now,
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM bench_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return self._deserialize_bench_run(row)

    @staticmethod
    def _deserialize_bench_run(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "job_id": row["job_id"],
            "session_id": row["session_id"],
            "subcommand": row["subcommand"],
            "label": row["label"],
            "workload": row["workload"],
            "mode": row["mode"],
            "max_concurrency": row["max_concurrency"],
            "peak_concurrency": row["peak_concurrency"],
            "request_rate": row["request_rate"],
            "summary": _json_loads(row["summary_json"], {}),
            "metrics": _json_loads(row["metrics_json"], {}),
            "spec_tokens": _json_loads(row["spec_tokens_json"], {}),
            "finish_reasons": _json_loads(row["finish_reasons_json"], {}),
            "telemetry": _json_loads(row["telemetry_json"], {}),
            "t_start": row["t_start"],
            "t_end": row["t_end"],
            "created_at": row["created_at"],
        }

    def list_bench_runs(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if session_id:
                rows = self._conn.execute(
                    "SELECT * FROM bench_runs WHERE session_id=? ORDER BY created_at ASC",
                    (session_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM bench_runs ORDER BY created_at DESC LIMIT 500"
                ).fetchall()
            return [self._deserialize_bench_run(r) for r in rows]

    # ------------------------------------------------------------------
    # Correctness Runs
    # ------------------------------------------------------------------
    def add_correctness_run(
        self,
        *,
        session_id: str,
        eval_type: str,
        suite_name: str,
        passed: bool,
        summary: Mapping[str, Any],
        job_id: str | None = None,
        correct_rate: float | None = None,
        wrong_rate: float | None = None,
        failed_rate: float | None = None,
        truncated_rate: float | None = None,
        ref_base_url: str | None = None,
        cases: Sequence[Mapping[str, Any]] | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> dict[str, Any]:
        eval_id = _new_id("eval")
        now = time.time()
        cases_file: str | None = None
        if cases:
            cases_file = f"{eval_id}.jsonl"
            out_path = self.artifacts_dir / cases_file
            with out_path.open("w", encoding="utf-8") as stream:
                for item in cases:
                    stream.write(_json_dumps(item) + "\n")

        with self._lock:
            self._conn.execute(
                "INSERT INTO correctness_runs ("
                "eval_id, job_id, session_id, eval_type, suite_name, passed, "
                "correct_rate, wrong_rate, failed_rate, truncated_rate, "
                "ref_base_url, summary_json, cases_file, t_start, t_end, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    eval_id,
                    job_id,
                    session_id,
                    eval_type,
                    suite_name,
                    int(bool(passed)),
                    correct_rate,
                    wrong_rate,
                    failed_rate,
                    truncated_rate,
                    ref_base_url,
                    _json_dumps(dict(summary)),
                    cases_file,
                    t_start if t_start is not None else now,
                    t_end if t_end is not None else now,
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM correctness_runs WHERE eval_id=?", (eval_id,)
            ).fetchone()
            return self._deserialize_correctness_run(row)

    @staticmethod
    def _deserialize_correctness_run(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "eval_id": row["eval_id"],
            "job_id": row["job_id"],
            "session_id": row["session_id"],
            "eval_type": row["eval_type"],
            "suite_name": row["suite_name"],
            "passed": bool(row["passed"]),
            "correct_rate": row["correct_rate"],
            "wrong_rate": row["wrong_rate"],
            "failed_rate": row["failed_rate"],
            "truncated_rate": row["truncated_rate"],
            "ref_base_url": row["ref_base_url"],
            "summary": _json_loads(row["summary_json"], {}),
            "has_cases": bool(row["cases_file"]),
            "t_start": row["t_start"],
            "t_end": row["t_end"],
            "created_at": row["created_at"],
        }

    def list_correctness_runs(
        self, *, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            if session_id:
                rows = self._conn.execute(
                    "SELECT * FROM correctness_runs WHERE session_id=? ORDER BY created_at ASC",
                    (session_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM correctness_runs ORDER BY created_at DESC LIMIT 500"
                ).fetchall()
            return [self._deserialize_correctness_run(r) for r in rows]

    def get_eval_cases(self, eval_id: str, limit: int = 500) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM correctness_runs WHERE eval_id=?", (eval_id,)
            ).fetchone()
            if row is None:
                return None
            record = self._deserialize_correctness_run(row)
            cases_file = row["cases_file"]

        cases: list[dict[str, Any]] = []
        if cases_file:
            path = self.artifacts_dir / cases_file
            if path.is_file():
                with path.open("r", encoding="utf-8") as stream:
                    for idx, line in enumerate(stream):
                        if idx >= limit:
                            break
                        line = line.strip()
                        if line:
                            cases.append(_json_loads(line, {}))
        record["cases"] = cases
        return record

    # ------------------------------------------------------------------
    # Multi-Session Comparison
    # ------------------------------------------------------------------
    def compare_sessions(self, session_ids: Sequence[str]) -> dict[str, Any]:
        details = []
        for sid in session_ids:
            detail = self.get_session_detail(sid)
            if detail is not None:
                details.append(detail)

        config_diffs: list[dict[str, Any]] = []
        if len(details) >= 2:
            base = details[0]
            cand = details[1]
            base_args = {
                **(base["server_info"].get("server_args") or {}),
                **base["labels"],
            }
            cand_args = {
                **(cand["server_info"].get("server_args") or {}),
                **cand["labels"],
            }
            for key in sorted(set(base_args) | set(cand_args)):
                bv = base_args.get(key)
                cv = cand_args.get(key)
                if bv != cv:
                    config_diffs.append({"key": key, "baseline": bv, "candidate": cv})

        return {
            "sessions": details,
            "config_diffs": config_diffs,
        }
