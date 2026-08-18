import cmd
import json
import shlex
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from ai_infra_bench.monitor.config import (
    ConfigurationError,
    ScrapeTarget,
    parse_targets,
    write_prometheus_config,
)


class ReloadError(RuntimeError):
    """Raised when Prometheus rejects or cannot receive a reload."""


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
        with urllib.request.urlopen(endpoint, timeout=timeout) as response:
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


class TargetRegistry:
    def __init__(
        self,
        targets: Sequence[ScrapeTarget],
        config_path: Path,
        scrape_interval: str,
        metrics_path: str,
        prometheus_url: str,
        reload_func: Callable[[str], None] = reload_prometheus,
    ):
        self.targets = list(targets)
        self.config_path = config_path
        self.scrape_interval = scrape_interval
        self.metrics_path = metrics_path
        self.prometheus_url = prometheus_url
        self.reload_func = reload_func
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

    def _commit(self, targets: Sequence[ScrapeTarget]) -> None:
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

    def add(self, base_urls: Sequence[str]) -> AddResult:
        parsed = parse_targets(base_urls, self.metrics_path)
        existing_by_key = {self._key(target): target for target in self.targets}
        added = []
        existing = []
        next_label = self.next_label
        for target in parsed:
            key = self._key(target)
            if key in existing_by_key:
                existing.append(existing_by_key[key])
                continue
            target = replace(target, label=f"server-{next_label}")
            next_label += 1
            added.append(target)
            existing_by_key[key] = target

        if added:
            self._commit([*self.targets, *added])
            self.next_label = next_label
        return AddResult(tuple(added), tuple(existing))

    def _find(self, identifier: str) -> ScrapeTarget | None:
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
        removed = []
        missing = []
        removed_labels = set()
        for identifier in identifiers:
            target = self._find(identifier)
            if target is None:
                missing.append(identifier)
            elif target.label not in removed_labels:
                removed.append(target)
                removed_labels.add(target.label)

        if removed:
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
        """add URL [URL ...]

        Add one or more SGLang base URLs and reload Prometheus.
        """
        urls = self._arguments(argument)
        if not urls:
            print("Usage: add URL [URL ...]")
            return
        try:
            result = self.registry.add(urls)
        except (ConfigurationError, ReloadError) as error:
            print(f"Add failed: {error}")
            return
        for target in result.added:
            print(f"Added {target.label}: {target.metrics_url}")
        for target in result.existing:
            print(f"Already monitored {target.label}: {target.metrics_url}")

    def do_delete(self, argument: str) -> None:
        """delete LABEL_OR_URL [LABEL_OR_URL ...]

        Delete targets by server-N label or base URL and reload Prometheus.
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
            for target in self.registry.targets
            for value in (target.label, target.base_url)
        ]
        return [candidate for candidate in candidates if candidate.startswith(text)]

    complete_del = complete_delete
    complete_remove = complete_delete

    def do_list(self, argument: str) -> None:
        """list

        List currently configured SGLang targets.
        """
        if argument.strip():
            print("Usage: list")
            return
        if not self.registry.targets:
            print("No monitored targets.")
            return
        for target in self.registry.targets:
            print(f"{target.label}: {target.metrics_url}")

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
        if not self.registry.targets:
            print("No monitored targets.")
            return
        for target in self.registry.targets:
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
