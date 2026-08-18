import hashlib
import logging
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class BinaryError(RuntimeError):
    """Raised when a monitoring service binary cannot be resolved."""


@dataclass(frozen=True)
class ArchiveSpec:
    product: str
    version: str
    url: str
    sha256: str


@dataclass(frozen=True)
class ResolvedBinary:
    product: str
    executable: Path
    home_path: Path | None = None


PROMETHEUS_VERSION = "3.13.2"
GRAFANA_VERSION = "13.1.3"


_PROMETHEUS_SHA256 = {
    (
        "darwin",
        "amd64",
    ): "e57095aed0b69e10edaee28b92718d4a65f46d466bf93aeda54075e901d15c2a",
    (
        "darwin",
        "arm64",
    ): "f68ca4f1dbedd6366bbfdd8ac5d2c0b7ba1f273474acc8d38eb33202fbeec7a4",
    (
        "linux",
        "amd64",
    ): "0e8c4d46101bd025ea8265e377d2caabc57f488fc1be1c367f37db69ea41be6f",
    (
        "linux",
        "arm64",
    ): "7cecb17a6f41d59814e1a0581a1f81f79051ad5973d1ecf39e23a9f747d6572a",
}


_GRAFANA_PACKAGES = {
    ("darwin", "amd64"): (
        "https://dl.grafana.com/grafana/release/13.1.3/"
        "grafana_13.1.3_31135815010_darwin_amd64.tar.gz",
        "abdb13038f8604c8fbb3d355e728ea217d4f67b41baf3e66369cff2f758179ca",
    ),
    ("darwin", "arm64"): (
        "https://dl.grafana.com/grafana/release/13.1.3/"
        "grafana_13.1.3_31135815010_darwin_arm64.tar.gz",
        "cbd4fc856fa5817a7fbc141d1e11cb1d79ca21cea15294cd32d9c82a666d382a",
    ),
    ("linux", "amd64"): (
        "https://dl.grafana.com/grafana/release/13.1.3/"
        "grafana_13.1.3_31135815010_linux_amd64.tar.gz",
        "e0fd22aa63901ebc961ee64195da60eef8624a831683ca10b26c7b068082e92b",
    ),
    ("linux", "arm64"): (
        "https://dl.grafana.com/grafana/release/13.1.3/"
        "grafana_13.1.3_31135815010_linux_arm64.tar.gz",
        "83eef49ccc6529da5ef3ffd2bc76dadfa66cca9a9684278bf858346cf2271b5d",
    ),
}


def _platform_key() -> tuple[str, str]:
    os_name = platform.system().lower()
    machine = platform.machine().lower()
    os_key = {"linux": "linux", "darwin": "darwin"}.get(os_name)
    arch_key = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine)
    if not os_key or not arch_key:
        raise BinaryError(
            f"automatic downloads do not support {os_name}/{machine}; "
            "install Prometheus and Grafana and use --runtime system"
        )
    return os_key, arch_key


def archive_spec(product: str) -> ArchiveSpec:
    key = _platform_key()
    if product == "prometheus":
        archive_name = f"prometheus-{PROMETHEUS_VERSION}.{key[0]}-{key[1]}.tar.gz"
        return ArchiveSpec(
            product=product,
            version=PROMETHEUS_VERSION,
            url=(
                "https://github.com/prometheus/prometheus/releases/download/"
                f"v{PROMETHEUS_VERSION}/{archive_name}"
            ),
            sha256=_PROMETHEUS_SHA256[key],
        )
    if product == "grafana":
        url, sha256 = _GRAFANA_PACKAGES[key]
        return ArchiveSpec(product, GRAFANA_VERSION, url, sha256)
    raise BinaryError(f"unknown monitoring product: {product}")


def _grafana_home(executable: Path) -> Path | None:
    candidates = []
    configured = os.environ.get("GF_PATHS_HOME")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            executable.parent.parent,
            Path("/usr/share/grafana"),
            Path("/usr/local/share/grafana"),
        ]
    )
    for candidate in candidates:
        if (candidate / "conf" / "defaults.ini").is_file():
            return candidate.resolve()
    return None


def _resolve_explicit(product: str, executable: Path, home_path: Path | None = None):
    executable = executable.expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise BinaryError(f"{product} executable is not runnable: {executable}")
    if product == "grafana":
        resolved_home = (
            home_path.expanduser().resolve() if home_path else _grafana_home(executable)
        )
        if (
            resolved_home is None
            or not (resolved_home / "conf" / "defaults.ini").is_file()
        ):
            raise BinaryError(
                "could not locate Grafana home (conf/defaults.ini); pass --grafana-home"
            )
        return ResolvedBinary(product, executable, resolved_home)
    return ResolvedBinary(product, executable)


def find_system_binary(
    product: str,
    explicit: Path | None = None,
    grafana_home: Path | None = None,
) -> ResolvedBinary | None:
    if explicit is not None:
        return _resolve_explicit(product, explicit, grafana_home)

    names = ["prometheus"] if product == "prometheus" else ["grafana", "grafana-server"]
    for name in names:
        found = shutil.which(name)
        if not found:
            continue
        executable = Path(found).resolve()
        if product == "grafana":
            home = (
                grafana_home.expanduser().resolve()
                if grafana_home
                else _grafana_home(executable)
            )
            if home is None:
                continue
            return ResolvedBinary(product, executable, home)
        return ResolvedBinary(product, executable)
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(spec: ArchiveSpec, destination: Path, attempts: int = 3) -> None:
    logger.info("Downloading %s %s from %s", spec.product, spec.version, spec.url)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        downloaded = destination.stat().st_size if destination.exists() else 0
        if downloaded and _file_sha256(destination) == spec.sha256:
            return
        headers = {"User-Agent": "ai-infra-bench-monitoring"}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
            logger.info(
                "Resuming %s download at %.0f MiB (attempt %d/%d)",
                spec.product,
                downloaded / 1024 / 1024,
                attempt,
                attempts,
            )
        request = urllib.request.Request(spec.url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                append = downloaded > 0 and status == 206
                if not append:
                    downloaded = 0
                next_report = ((downloaded // (25 * 1024 * 1024)) + 1) * (
                    25 * 1024 * 1024
                )
                mode = "ab" if append else "wb"
                with destination.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > 1024 * 1024 * 1024:
                            raise BinaryError(f"{spec.product} download exceeded 1 GiB")
                        output.write(chunk)
                        if downloaded >= next_report:
                            logger.info(
                                "Downloaded %.0f MiB of %s",
                                downloaded / 1024 / 1024,
                                spec.product,
                            )
                            next_report += 25 * 1024 * 1024

            actual = _file_sha256(destination)
            if actual == spec.sha256:
                return
            last_error = BinaryError(
                f"SHA256 mismatch for {spec.product}: expected {spec.sha256}, got {actual}"
            )
            destination.unlink(missing_ok=True)
        except BinaryError:
            raise
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            last_error = error

        if attempt < attempts:
            logger.warning(
                "%s download failed on attempt %d/%d: %s",
                spec.product,
                attempt,
                attempts,
                last_error,
            )
            time.sleep(attempt)

    raise BinaryError(
        f"failed to download {spec.product} after {attempts} attempts: {last_error}"
    ) from last_error


def _validate_archive(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise BinaryError(f"unsafe path in {archive.name}: {member.name}")
        if member.issym() or member.islnk():
            link_target = (target.parent / member.linkname).resolve()
            if root != link_target and root not in link_target.parents:
                raise BinaryError(f"unsafe link in {archive.name}: {member.name}")


def _find_executable(product: str, root: Path) -> Path:
    names = ["prometheus"] if product == "prometheus" else ["grafana", "grafana-server"]
    matches = [path for name in names for path in root.rglob(name) if path.is_file()]
    if not matches:
        raise BinaryError(f"{product} archive did not contain an executable")
    matches.sort(key=lambda path: ("/bin/" not in path.as_posix(), len(path.parts)))
    executable = matches[0]
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def download_binary(product: str, cache_dir: Path) -> ResolvedBinary:
    spec = archive_spec(product)
    os_key, arch_key = _platform_key()
    install_dir = cache_dir.expanduser().resolve() / (
        f"{product}-{spec.version}-{os_key}-{arch_key}"
    )
    if install_dir.is_dir():
        try:
            executable = _find_executable(product, install_dir)
            home = executable.parent.parent if product == "grafana" else None
            if product != "grafana" or (home / "conf" / "defaults.ini").is_file():
                return ResolvedBinary(product, executable, home)
        except BinaryError:
            shutil.rmtree(install_dir)

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{product}-", dir=install_dir.parent
    ) as temporary_dir:
        temporary = Path(temporary_dir)
        archive_path = temporary / "download.tar.gz"
        extracted = temporary / "extracted"
        extracted.mkdir()
        _download(spec, archive_path)
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                _validate_archive(archive, extracted)
                archive.extractall(extracted)
        except (OSError, tarfile.TarError) as error:
            raise BinaryError(f"failed to extract {product}: {error}") from error

        try:
            extracted.rename(install_dir)
        except FileExistsError:
            # Another concurrent invocation completed the same cache entry.
            pass

    executable = _find_executable(product, install_dir)
    home = executable.parent.parent if product == "grafana" else None
    if product == "grafana" and not (home / "conf" / "defaults.ini").is_file():
        raise BinaryError(f"downloaded Grafana home is invalid: {home}")
    return ResolvedBinary(product, executable, home)


def resolve_binary(
    product: str,
    runtime: str,
    cache_dir: Path,
    explicit: Path | None = None,
    grafana_home: Path | None = None,
) -> ResolvedBinary:
    if runtime != "download":
        found = find_system_binary(product, explicit, grafana_home)
        if found is not None:
            logger.info("Using system %s: %s", product, found.executable)
            return found
        if runtime == "system":
            raise BinaryError(
                f"{product} was not found; install it, pass --{product}-bin, "
                "or use --runtime auto"
            )
    if explicit is not None:
        return _resolve_explicit(product, explicit, grafana_home)
    return download_binary(product, cache_dir)
