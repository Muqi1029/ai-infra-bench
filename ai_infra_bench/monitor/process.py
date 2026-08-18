import logging
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Sequence

logger = logging.getLogger(__name__)


class ProcessError(RuntimeError):
    """Raised when a monitoring service cannot start or exits unexpectedly."""


def _listen_address(host: str, port: int) -> str:
    host = host.strip("[]")
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def ensure_port_available(host: str, port: int, service: str) -> None:
    host = host.strip("[]")
    try:
        addresses = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
    except socket.gaierror as error:
        raise ProcessError(f"invalid --listen-address {host!r}: {error}") from error

    errors = []
    for family, socktype, protocol, _, address in addresses:
        with socket.socket(family, socktype, protocol) as sock:
            try:
                sock.bind(address)
                return
            except OSError as error:
                errors.append(error)
    detail = errors[-1] if errors else "address unavailable"
    raise ProcessError(f"{service} cannot bind {host}:{port}: {detail}")


def wait_for_http(
    url: str,
    process: subprocess.Popen,
    service: str,
    log_path: Path,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise ProcessError(
                f"{service} exited with code {return_code}; see {log_path}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.25)
    raise ProcessError(
        f"timed out waiting for {service} at {url}: {last_error}; see {log_path}"
    )


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    log_file: IO[bytes]
    log_path: Path

    def stop(self, timeout: float = 10) -> None:
        if self.process.poll() is not None:
            self.log_file.close()
            return
        logger.info("Stopping %s", self.name)
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            self.process.wait(timeout=timeout)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired) as error:
                    logger.warning("Could not fully stop %s: %s", self.name, error)
        finally:
            self.log_file.close()


class MonitoringStack:
    def __init__(self):
        self.processes: list[ManagedProcess] = []

    def start(
        self,
        name: str,
        command: Sequence[str],
        log_path: Path,
        ready_url: str,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> ManagedProcess:
        logger.info("Starting %s", name)
        log_file = log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                list(command),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        except OSError as error:
            log_file.close()
            raise ProcessError(f"failed to start {name}: {error}") from error
        managed = ManagedProcess(name, process, log_file, log_path)
        self.processes.append(managed)
        wait_for_http(ready_url, process, name, log_path, timeout)
        return managed

    def check(self) -> None:
        for managed in self.processes:
            return_code = managed.process.poll()
            if return_code is not None:
                raise ProcessError(
                    f"{managed.name} exited with code {return_code}; "
                    f"see {managed.log_path}"
                )

    def wait(self) -> None:
        while True:
            self.check()
            time.sleep(0.5)

    def stop(self) -> None:
        for managed in reversed(self.processes):
            try:
                managed.stop()
            except OSError as error:
                logger.warning("Could not stop %s: %s", managed.name, error)
        self.processes.clear()


def prometheus_command(
    executable: Path,
    config_path: Path,
    data_path: Path,
    listen_address: str,
    port: int,
    retention_time: str,
) -> list[str]:
    return [
        str(executable),
        f"--config.file={config_path}",
        f"--storage.tsdb.path={data_path}",
        f"--storage.tsdb.retention.time={retention_time}",
        f"--web.listen-address={_listen_address(listen_address, port)}",
        "--web.enable-lifecycle",
    ]


def grafana_command(executable: Path, home_path: Path, config_path: Path) -> list[str]:
    command = [str(executable)]
    if executable.name == "grafana":
        command.append("server")
    command.extend(["--homepath", str(home_path), "--config", str(config_path)])
    return command
