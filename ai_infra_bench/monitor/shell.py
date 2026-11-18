import cmd
import json
import re
import shlex
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ai_infra_bench.monitor.config import (
    DEFAULT_METRICS_PATH,
    ConfigurationError,
    ScrapeTarget,
    parse_labels,
    parse_targets,
    write_prometheus_config,
    write_sd_targets,
)

TARGET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


class ReloadError(RuntimeError):
    """Raised when Prometheus rejects or cannot receive a reload."""


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def reload_prometheus(url: str, timeout: float = 5) -> None:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/-/reload",
        data=b"",
        method="POST",
        headers={"User-Agent": "ai-infra-bench-monitoring"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise ReloadError(f"Prometheus reload returned HTTP {response.status}")
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise ReloadError(
            f"Prometheus reload returned HTTP {error.code}{suffix}"
        ) from error
    except (OSError, urllib.error.URLError) as error:
        raise ReloadError(f"could not reload Prometheus: {error}") from error


def fetch_target_status(url: str, timeout: float = 3) -> dict[str, dict]:
    endpoint = f"{url.rstrip('/')}/api/v1/targets?state=active"
    try:
        with _no_proxy_opener().open(endpoint, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise ReloadError(f"could not query Prometheus targets: {error}") from error
    if payload.get("status") != "success":
        raise ReloadError("Prometheus returned an unsuccessful target response")

    statuses = {}
    for target in payload.get("data", {}).get("activeTargets", []):
        label = target.get("labels", {}).get("aib_target")
        if label:
            statuses[label] = target
    return statuses


@dataclass(frozen=True)
class AddResult:
    added: tuple[ScrapeTarget, ...]
    existing: tuple[ScrapeTarget, ...]


@dataclass(frozen=True)
class DeleteResult:
    removed: tuple[ScrapeTarget, ...]
    missing: tuple[str, ...]


def validate_target_id(value: str) -> str:
    cleaned = value.strip()
    if not TARGET_ID_PATTERN.fullmatch(cleaned):
        raise ConfigurationError(
            f"invalid target identifier {value!r}; use letters, digits, '.', '_', or '-'"
        )
    return cleaned


class TargetRegistry:
    def __init__(
        self,
        targets: Sequence[ScrapeTarget],
        config_path: Path,
        scrape_interval: str,
        metrics_path: str,
        prometheus_url: str,
        reload_func: Callable[[str], None] = reload_prometheus,
        *,
        sd_file_path: Path | None = None,
        push_ttl_s: float = 60.0,
    ):
        self._lock = threading.RLock()
        self.targets = list(targets)
        self.config_path = config_path
        self.scrape_interval = scrape_interval
        self.metrics_path = metrics_path
        self.prometheus_url = prometheus_url
        self.reload_func = reload_func
        self.sd_file_path = sd_file_path
        self.push_ttl_s = max(push_ttl_s, 1.0)
        self._pushed_snapshots: dict[str, tuple[float, str]] = {}
        numeric_labels = [
            int(target.label.removeprefix("server-"))
            for target in self.targets
            if target.label.startswith("server-")
            and target.label.removeprefix("server-").isdigit()
        ]
        self.next_label = max(numeric_labels, default=-1) + 1

    @staticmethod
    def _key(target: ScrapeTarget) -> tuple[str, str, str]:
        return target.scheme, target.address, target.metrics_path

    def _allocate_label(self, existing_labels: set[str]) -> str:
        while f"server-{self.next_label}" in existing_labels:
            self.next_label += 1
        label = f"server-{self.next_label}"
        self.next_label += 1
        return label

    def _commit(self, targets: Sequence[ScrapeTarget]) -> None:
        if self.sd_file_path is not None:
            write_sd_targets(self.sd_file_path, targets)
            self.targets = list(targets)
            return

        write_prometheus_config(self.config_path, targets, self.scrape_interval)
        try:
            self.reload_func(self.prometheus_url)
        except Exception as reload_error:
            try:
                write_prometheus_config(
                    self.config_path, self.targets, self.scrape_interval
                )
                self.reload_func(self.prometheus_url)
            except Exception as rollback_error:
                raise ReloadError(
                    f"reload failed ({reload_error}); rollback also failed "
                    f"({rollback_error})"
                ) from rollback_error
            if isinstance(reload_error, ReloadError):
                raise reload_error
            raise ReloadError(
                f"Prometheus reload failed: {reload_error}"
            ) from reload_error
        self.targets = list(targets)

    def _prune_expired_locked(self) -> None:
        if not self._pushed_snapshots:
            return
        now = time.monotonic()
        expired = {
            label
            for label, (ts, _) in self._pushed_snapshots.items()
            if now - ts > self.push_ttl_s
        }
        if not expired:
            return
        for label in expired:
            self._pushed_snapshots.pop(label, None)
        remaining = [
            t for t in self.targets if not (t.mode == "push" and t.label in expired)
        ]
        if len(remaining) != len(self.targets):
            self._commit(remaining)

    def snapshot(self) -> list[ScrapeTarget]:
        with self._lock:
            self._prune_expired_locked()
            return list(self.targets)

    def add(
        self,
        base_urls: Sequence[str],
        *,
        labels: Sequence[str] | Mapping[str, str] | tuple[tuple[str, str], ...] = (),
        metrics_path: str | None = None,
        target_id: str | None = None,
    ) -> AddResult:
        normalized_labels = (
            labels
            if isinstance(labels, tuple)
            and all(isinstance(item, tuple) and len(item) == 2 for item in labels)
            else parse_labels(labels)  # type: ignore[arg-type]
        )
        if target_id is not None and len(base_urls) > 1:
            raise ConfigurationError(
                "target identifier can only be set for a single URL"
            )
        explicit_label = validate_target_id(target_id) if target_id else None
        parsed = parse_targets(
            base_urls,
            metrics_path or self.metrics_path,
            labels=normalized_labels,
        )

        with self._lock:
            self._prune_expired_locked()
            updated_targets = list(self.targets)
            index_by_key = {
                self._key(target): idx for idx, target in enumerate(updated_targets)
            }
            used_labels = {target.label for target in updated_targets}
            saved_next_label = self.next_label

            added: list[ScrapeTarget] = []
            existing: list[ScrapeTarget] = []
            changed = False

            try:
                for candidate in parsed:
                    key = self._key(candidate)
                    if key in index_by_key:
                        idx = index_by_key[key]
                        current = updated_targets[idx]
                        merged_labels = normalized_labels or current.labels
                        if merged_labels != current.labels:
                            current = replace(current, labels=merged_labels)
                            updated_targets[idx] = current
                            changed = True
                        existing.append(current)
                        continue

                    if explicit_label:
                        if explicit_label in used_labels:
                            raise ConfigurationError(
                                f"target label {explicit_label!r} is already in use"
                            )
                        assigned_label = explicit_label
                    else:
                        assigned_label = self._allocate_label(used_labels)
                    used_labels.add(assigned_label)
                    created = replace(
                        candidate,
                        label=assigned_label,
                        labels=normalized_labels,
                        mode="pull",
                    )
                    index_by_key[key] = len(updated_targets)
                    updated_targets.append(created)
                    added.append(created)
                    changed = True

                if changed:
                    self._commit(updated_targets)
            except Exception:
                self.next_label = saved_next_label
                raise

            return AddResult(tuple(added), tuple(existing))

    def upsert_pushed(
        self,
        target_id: str,
        relay_address: str,
        metrics_text: str,
        *,
        labels: Sequence[str] | Mapping[str, str] | tuple[tuple[str, str], ...] = (),
        base_url: str | None = None,
    ) -> ScrapeTarget:
        label = validate_target_id(target_id)
        normalized_labels = (
            labels
            if isinstance(labels, tuple)
            and all(isinstance(item, tuple) and len(item) == 2 for item in labels)
            else parse_labels(labels)  # type: ignore[arg-type]
        )
        pushed_target = ScrapeTarget(
            base_url=base_url or f"push://{label}",
            scheme="http",
            address=relay_address,
            metrics_path=f"/pushed/{label}/metrics",
            label=label,
            labels=normalized_labels,
            mode="push",
        )

        with self._lock:
            self._prune_expired_locked()
            existing_idx = next(
                (idx for idx, item in enumerate(self.targets) if item.label == label),
                None,
            )
            if existing_idx is not None and self.targets[existing_idx].mode != "push":
                raise ConfigurationError(
                    f"target label {label!r} conflicts with an active pull target"
                )

            self._pushed_snapshots[label] = (time.monotonic(), metrics_text)
            if existing_idx is None:
                self._commit([*self.targets, pushed_target])
            elif self.targets[existing_idx] != pushed_target:
                updated = list(self.targets)
                updated[existing_idx] = pushed_target
                self._commit(updated)
            return pushed_target

    def get_pushed_metrics(self, target_id: str) -> str | None:
        with self._lock:
            entry = self._pushed_snapshots.get(target_id)
            if entry is None:
                return None
            ts, metrics_text = entry
            if time.monotonic() - ts > self.push_ttl_s:
                self._prune_expired_locked()
                return None
            return metrics_text

    def _find_locked(self, identifier: str) -> ScrapeTarget | None:
        for target in self.targets:
            if identifier in {target.label, target.base_url, target.metrics_url}:
                return target
        try:
            parsed = parse_targets([identifier], self.metrics_path)[0]
        except ConfigurationError:
            return None
        key = self._key(parsed)
        return next(
            (target for target in self.targets if self._key(target) == key), None
        )

    def delete(self, identifiers: Sequence[str]) -> DeleteResult:
        with self._lock:
            self._prune_expired_locked()
            removed = []
            missing = []
            removed_labels = set()
            for identifier in identifiers:
                target = self._find_locked(identifier)
                if target is None:
                    missing.append(identifier)
                elif target.label not in removed_labels:
                    removed.append(target)
                    removed_labels.add(target.label)

            if removed:
                for label in removed_labels:
                    self._pushed_snapshots.pop(label, None)
                self._commit(
                    [
                        target
                        for target in self.targets
                        if target.label not in removed_labels
                    ]
                )
            return DeleteResult(tuple(removed), tuple(missing))


class MonitorShell(cmd.Cmd):
    prompt = "aib-monitor> "
    intro = (
        "Interactive monitor shell. Type 'help' for commands; "
        "'quit' stops Prometheus and Grafana."
    )

    def __init__(
        self,
        registry: TargetRegistry,
        process_check: Callable[[], None] | None = None,
    ):
        super().__init__()
        self.registry = registry
        self.process_check = process_check

    @staticmethod
    def _arguments(argument: str) -> list[str] | None:
        try:
            return shlex.split(argument)
        except ValueError as error:
            print(f"Invalid command: {error}")
            return None

    def precmd(self, line: str) -> str:
        if self.process_check is not None:
            self.process_check()
        return line

    def emptyline(self) -> None:
        return None

    def default(self, line: str) -> None:
        command = line.split(maxsplit=1)[0] if line.strip() else line
        print(f"Unknown command: {command}. Type 'help' for available commands.")

    def do_add(self, argument: str) -> None:
        """add URL [URL ...] [KEY=VALUE ...]

        Add one or more base URLs (with optional KEY=VALUE labels) to monitoring.
        """
        tokens = self._arguments(argument)
        if not tokens:
            print("Usage: add URL [URL ...] [KEY=VALUE ...]")
            return
        urls = [tok for tok in tokens if "://" in tok or "=" not in tok]
        label_tokens = [tok for tok in tokens if "://" not in tok and "=" in tok]
        if not urls:
            print("Usage: add URL [URL ...] [KEY=VALUE ...]")
            return
        try:
            result = self.registry.add(urls, labels=label_tokens)
        except (ConfigurationError, ReloadError) as error:
            print(f"Add failed: {error}")
            return
        for target in result.added:
            print(f"Added {target.label}: {target.metrics_url}")
        for target in result.existing:
            print(f"Already monitored {target.label}: {target.metrics_url}")

    def do_delete(self, argument: str) -> None:
        """delete LABEL_OR_URL [LABEL_OR_URL ...]

        Delete targets by server-N label or base URL.
        """
        identifiers = self._arguments(argument)
        if not identifiers:
            print("Usage: delete LABEL_OR_URL [LABEL_OR_URL ...]")
            return
        try:
            result = self.registry.delete(identifiers)
        except ReloadError as error:
            print(f"Delete failed: {error}")
            return
        for target in result.removed:
            print(f"Deleted {target.label}: {target.metrics_url}")
        for identifier in result.missing:
            print(f"Not monitored: {identifier}")

    do_del = do_delete
    do_remove = do_delete

    def complete_delete(
        self, text: str, line: str, begidx: int, endidx: int
    ) -> list[str]:
        candidates = [
            value
            for target in self.registry.snapshot()
            for value in (target.label, target.base_url)
        ]
        return [candidate for candidate in candidates if candidate.startswith(text)]

    complete_del = complete_delete
    complete_remove = complete_delete

    def do_list(self, argument: str) -> None:
        """list

        List currently configured targets.
        """
        if argument.strip():
            print("Usage: list")
            return
        targets = self.registry.snapshot()
        if not targets:
            print("No monitored targets.")
            return
        for target in targets:
            extras = []
            if target.mode == "push":
                extras.append("mode=push")
            extras.extend(f"{k}={v}" for k, v in target.labels)
            suffix = f" ({', '.join(extras)})" if extras else ""
            print(f"{target.label}: {target.metrics_url}{suffix}")

    def do_status(self, argument: str) -> None:
        """status

        Show Prometheus scrape health for all configured targets.
        """
        if argument.strip():
            print("Usage: status")
            return
        try:
            statuses = fetch_target_status(self.registry.prometheus_url)
        except ReloadError as error:
            print(f"Status failed: {error}")
            return
        targets = self.registry.snapshot()
        if not targets:
            print("No monitored targets.")
            return
        for target in targets:
            status = statuses.get(target.label)
            if status is None:
                print(f"{target.label}: pending ({target.metrics_url})")
                continue
            health = status.get("health", "unknown")
            error = status.get("lastError", "").strip()
            suffix = f" - {error}" if error else ""
            print(f"{target.label}: {health} ({target.metrics_url}){suffix}")

    def do_quit(self, argument: str) -> bool:
        """quit

        Stop the monitoring services and exit.
        """
        if argument.strip():
            print("Usage: quit")
            return False
        return True

    do_exit = do_quit

    def do_EOF(self, argument: str) -> bool:
        print()
        return True
