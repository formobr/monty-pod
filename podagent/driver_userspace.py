"""The last Vulkan ladder rank: bring the graphics userspace of the HOST's exact driver version.

A container toolkit that grants only compute/utility/video injects no libEGL_nvidia and friends, so Vulkan
answers VK_ERROR_INCOMPATIBLE_DRIVER whatever the image carries (MISC-998). The userspace must match the
host kernel module to the patch level, so it is taken from the vendor's own .run of that very version,
checksum-verified, extracted without root, and reduced to an explicit allowlist.
Research receipt: docs/research/pod-driver-userspace-from-nvidia.md.

This module only provisions a directory; wiring it into the environment and re-probing is the ladder's job.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

DRIVER_VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")

# Named refusal reasons (the ladder prints them in the refusal trailer).
DRIVER_VERSION_UNAVAILABLE = "DRIVER_VERSION_UNAVAILABLE"
DRIVER_PACKAGE_UNAVAILABLE = "DRIVER_PACKAGE_UNAVAILABLE"
DOWNLOAD_TIMEOUT = "DOWNLOAD_TIMEOUT"
CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
INSUFFICIENT_DISK = "INSUFFICIENT_DISK"
EXTRACTION_FAILED = "EXTRACTION_FAILED"
DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
VULKAN_REPROBE_FAILED = "VULKAN_REPROBE_FAILED"

BASE_URL = "https://download.nvidia.com/XFree86/Linux-x86_64"
# Whole rank (download + extract + re-probe), well under the engine-declared 316 s boot.
RANK_WALL_S = 150.0
# Time the download must leave for extraction and the fresh re-probe.
AFTER_DOWNLOAD_RESERVE_S = 40.0
EXTRACT_TIMEOUT_S = 60.0
SHA_FETCH_TIMEOUT_S = 15.0
MIN_BYTES_PER_S = 1_000_000          # stall floor, measured per window
STALL_WINDOW_S = 10.0
CHUNK = 1 << 20
# Disk: archive + what --extract-only unpacks (1.2-1.5 GB measured) + the kept set (<= ~160 MB).
EXTRACT_BYTES = 1_600_000_000
KEPT_BYTES = 200_000_000
UNKNOWN_ARCHIVE_BYTES = 450_000_000

ALIAS = "libEGL_nvidia.so.0"
MANIFEST = "nvidia_icd.json"


def allowlist(version: str) -> tuple[str, ...]:
    return tuple(f"{stem}.so.{version}" for stem in (
        "libEGL_nvidia", "libnvidia-eglcore", "libnvidia-glsi", "libnvidia-glvkspirv", "libnvidia-gpucomp"))


def kept_names(version: str) -> set[str]:
    return {*allowlist(version), ALIAS, MANIFEST}


@dataclass
class FetchResult:
    ok: bool
    reason: str = ""
    detail: str = ""
    lib_dir: str = ""
    manifest: str = ""
    elapsed: float = 0.0
    cap: float = RANK_WALL_S


class Response(Protocol):
    status: int
    length: int | None

    def chunks(self) -> Iterable[bytes]: ...
    def close(self) -> None: ...


Transport = Callable[[str, float], Response]
Extractor = Callable[[str, str, float], "tuple[int, str]"]


class _Refusal(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class _HttpResponse:
    def __init__(self, r) -> None:
        self._r = r
        self.status = r.status_code
        try:
            self.length = int(r.headers["Content-Length"])
        except (KeyError, ValueError):
            self.length = None

    def chunks(self) -> Iterable[bytes]:
        import requests
        try:
            yield from self._r.iter_content(CHUNK)
        except requests.Timeout as e:
            raise TimeoutError(str(e)) from e
        except requests.RequestException as e:
            raise OSError(str(e)) from e

    def close(self) -> None:
        self._r.close()


def default_transport(url: str, timeout: float) -> Response:
    import requests
    try:
        return _HttpResponse(requests.get(url, stream=True, timeout=(min(10.0, timeout), timeout)))
    except requests.Timeout as e:
        raise TimeoutError(str(e)) from e
    except requests.RequestException as e:
        raise OSError(str(e)) from e


def default_extractor(run_file: str, target: str, timeout: float) -> tuple[int, str]:
    """(returncode, stdout+stderr tail). A missing `sh` raises FileNotFoundError; a timeout raises TimeoutError."""
    if shutil.which("sh") is None:
        raise FileNotFoundError("sh")
    try:
        r = subprocess.run(["sh", run_file, "--extract-only", "--target", target],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"extract exceeded {timeout:.0f}s") from e
    return r.returncode, (r.stdout or b"")[-300:].decode("utf-8", "replace")


def _verified(final: Path, version: str) -> bool:
    """The kept directory is complete: exactly the allowlist + alias + a manifest pointing at the alias."""
    try:
        if {p.name for p in final.iterdir()} != kept_names(version):
            return False
        for name in allowlist(version):
            if (final / name).is_symlink() or (final / name).stat().st_size == 0:
                return False
        if os.readlink(final / ALIAS) != f"libEGL_nvidia.so.{version}":
            return False
        icd = json.loads((final / MANIFEST).read_text())["ICD"]
        return icd["library_path"] == str(final / ALIAS) and isinstance(icd["api_version"], str)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _fetch_text(transport: Transport, url: str, timeout: float) -> str | None:
    """Small text file body, None on 404; any other failure is a refusal."""
    try:
        resp = transport(url, timeout)
    except TimeoutError as e:
        raise _Refusal(DOWNLOAD_TIMEOUT, f"{url.rsplit('/', 1)[-1]}: {e}") from e
    except OSError as e:
        raise _Refusal(DRIVER_PACKAGE_UNAVAILABLE, f"{url.rsplit('/', 1)[-1]}: {type(e).__name__}") from e
    try:
        if resp.status == 404:
            return None
        if resp.status != 200:
            raise _Refusal(DRIVER_PACKAGE_UNAVAILABLE, f"{url.rsplit('/', 1)[-1]}: HTTP {resp.status}")
        try:
            return b"".join(resp.chunks())[:4096].decode("utf-8", "replace")
        except TimeoutError as e:
            raise _Refusal(DOWNLOAD_TIMEOUT, f"{url.rsplit('/', 1)[-1]}: {e}") from e
        except OSError as e:
            raise _Refusal(DRIVER_PACKAGE_UNAVAILABLE, f"{url.rsplit('/', 1)[-1]}: {type(e).__name__}") from e
    finally:
        resp.close()


def _expected_sha(text: str) -> str:
    token = (text.split() or [""])[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise _Refusal(DRIVER_PACKAGE_UNAVAILABLE, "malformed .sha256sum")
    return token


def fetch_driver_userspace(
    version: str,
    base_dir: str | os.PathLike,
    *,
    transport: Transport | None = None,
    extractor: Extractor | None = None,
    clock: Callable[[], float] = time.monotonic,
    disk_usage: Callable[[str], object] = shutil.disk_usage,
    cap: float = RANK_WALL_S,
) -> FetchResult:
    """Provision <base_dir>/nvidia-<version>/ and return where. Never raises: every failure is a named
    FetchResult, and nothing but a verified directory is left behind."""
    start = clock()

    def done(**kw) -> FetchResult:
        return FetchResult(elapsed=clock() - start, cap=cap, **kw)

    if not isinstance(version, str) or not DRIVER_VERSION_RE.match(version):
        return done(ok=False, reason=DRIVER_VERSION_UNAVAILABLE, detail="not a driver version")
    base = Path(base_dir)
    final = base / f"nvidia-{version}"
    if _verified(final, version):
        return done(ok=True, lib_dir=str(final), manifest=str(final / MANIFEST), detail="reused")
    work: Path | None = None
    try:
        try:
            base.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=f".fetch-{version}-", dir=base))
        except OSError as e:
            return done(ok=False, reason=INSUFFICIENT_DISK,
                        detail=f"state dir {base} not writable ({type(e).__name__}: {e})")
        _provision(version, base, final, work, transport or default_transport,
                   extractor or default_extractor, clock, start, cap, disk_usage)
    except _Refusal as r:
        return done(ok=False, reason=r.reason, detail=r.detail)
    except OSError as e:
        return done(ok=False, reason=EXTRACTION_FAILED, detail=f"{type(e).__name__}: {e}")
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
    return done(ok=True, lib_dir=str(final), manifest=str(final / MANIFEST))


def _provision(version, base, final, work, transport, extractor, clock, start, cap, disk_usage) -> None:
    def remaining() -> float:
        return cap - (clock() - start)

    # 1. pick the variant: -no-compat32 first, the full .run when that one is not published (404)
    name = None
    for variant in (f"NVIDIA-Linux-x86_64-{version}-no-compat32.run", f"NVIDIA-Linux-x86_64-{version}.run"):
        left = remaining() - AFTER_DOWNLOAD_RESERVE_S
        if left <= 0:
            raise _Refusal(DOWNLOAD_TIMEOUT, "wall exhausted before the checksum fetch")
        body = _fetch_text(transport, f"{BASE_URL}/{version}/{variant}.sha256sum", min(SHA_FETCH_TIMEOUT_S, left))
        if body is not None:
            name, expected = variant, _expected_sha(body)
            break
    if name is None:
        raise _Refusal(DRIVER_PACKAGE_UNAVAILABLE, f"no .run or .sha256sum published for {version}")

    # 2. stream it, hashing as it arrives
    run_file = work / name
    left = remaining() - AFTER_DOWNLOAD_RESERVE_S
    if left <= 0:
        raise _Refusal(DOWNLOAD_TIMEOUT, "wall exhausted before the download")
    try:
        resp = transport(f"{BASE_URL}/{version}/{name}", min(SHA_FETCH_TIMEOUT_S, left))
    except TimeoutError as e:
        raise _Refusal(DOWNLOAD_TIMEOUT, str(e)) from e
    except OSError as e:
        raise _Refusal(DRIVER_PACKAGE_UNAVAILABLE, type(e).__name__) from e
    try:
        if resp.status != 200:
            raise _Refusal(DRIVER_PACKAGE_UNAVAILABLE, f"{name}: HTTP {resp.status}")
        need = (resp.length or UNKNOWN_ARCHIVE_BYTES) + EXTRACT_BYTES + KEPT_BYTES
        free = disk_usage(str(base)).free
        if free < need:
            raise _Refusal(INSUFFICIENT_DISK, f"need {need // 2**20} MiB, free {free // 2**20} MiB")
        digest = hashlib.sha256()
        total = win_bytes = 0
        win_start = clock()
        try:
            with open(run_file, "wb") as out:
                for chunk in resp.chunks():
                    out.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                    win_bytes += len(chunk)
                    now = clock()
                    if remaining() - AFTER_DOWNLOAD_RESERVE_S <= 0:
                        raise _Refusal(DOWNLOAD_TIMEOUT, f"wall hit after {total} bytes")
                    if now - win_start >= STALL_WINDOW_S:
                        if win_bytes < MIN_BYTES_PER_S * (now - win_start):
                            raise _Refusal(DOWNLOAD_TIMEOUT,
                                           f"stalled below {MIN_BYTES_PER_S} B/s after {total} bytes")
                        win_start, win_bytes = now, 0
        except TimeoutError as e:
            raise _Refusal(DOWNLOAD_TIMEOUT, f"read stalled after {total} bytes: {e}") from e
        except OSError as e:
            raise _Refusal(DOWNLOAD_TIMEOUT, f"transfer broke after {total} bytes: {type(e).__name__}") from e
    finally:
        resp.close()
    if digest.hexdigest() != expected:
        raise _Refusal(CHECKSUM_MISMATCH, f"{name}: got {digest.hexdigest()[:12]}, want {expected[:12]}")

    # 3. only now run the vendor script; it creates the target itself and refuses one that exists
    target = work / "extract"
    ext_timeout = min(EXTRACT_TIMEOUT_S, remaining() - AFTER_DOWNLOAD_RESERVE_S / 2)
    if ext_timeout <= 0:
        raise _Refusal(EXTRACTION_FAILED, "wall exhausted before extraction")
    try:
        rc, err = extractor(str(run_file), str(target), ext_timeout)
    except FileNotFoundError as e:
        raise _Refusal(DEPENDENCY_MISSING, f"{e.filename or e}") from e
    except TimeoutError as e:
        raise _Refusal(EXTRACTION_FAILED, str(e)) from e
    if rc != 0:
        reason = DEPENDENCY_MISSING if "not found" in err else EXTRACTION_FAILED
        raise _Refusal(reason, f"extractor rc={rc}: {err.strip()[-150:]}")

    # 4. keep the allowlist only: versioned names, the one alias, a manifest with the shipped api_version
    stage = work / "stage"
    stage.mkdir()
    for lib in allowlist(version):
        src = target / lib
        if not src.is_file() or src.is_symlink():
            raise _Refusal(EXTRACTION_FAILED, f"missing {lib} in the extracted tree")
        shutil.copyfile(src, stage / lib)
        (stage / lib).chmod(0o755)
    os.symlink(f"libEGL_nvidia.so.{version}", stage / ALIAS)
    try:
        api = json.loads((target / "nvidia_icd.json").read_text())["ICD"]["api_version"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise _Refusal(EXTRACTION_FAILED, f"no api_version in nvidia_icd.json ({type(e).__name__})") from e
    if not isinstance(api, str) or not DRIVER_VERSION_RE.match(api):
        raise _Refusal(EXTRACTION_FAILED, "malformed api_version in nvidia_icd.json")
    (stage / MANIFEST).write_text(json.dumps(
        {"file_format_version": "1.0.1", "ICD": {"library_path": str(final / ALIAS), "api_version": api}}) + "\n")
    if final.exists() or final.is_symlink():          # an unverified leftover of ours
        if final.is_dir() and not final.is_symlink():
            shutil.rmtree(final)
        else:
            final.unlink()
    os.rename(stage, final)
