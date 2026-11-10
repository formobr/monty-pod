"""Bring the graphics + video userspace of the HOST's exact driver version (Vulkan ladder's last rank, and the
NVENC probe when libnvidia-encode.so.1 cannot be loaded).

A container toolkit that grants only compute/utility injects no libEGL_nvidia and friends, so Vulkan
answers VK_ERROR_INCOMPATIBLE_DRIVER whatever the image carries (MISC-998); without the `video` set it also
injects no libnvidia-encode / libnvcuvid (MISC-1026). The userspace must match the
host kernel module to the patch level, so it is taken from the vendor's own .run of that very version,
checksum-verified, extracted without root, and reduced to an explicit allowlist.
Research receipts: docs/research/pod-driver-userspace-from-nvidia.md,
docs/research/pod-driver-userspace-video-libs.md.

This module only provisions a directory; wiring it into the environment and re-probing is the ladder's job.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from . import artifact

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
VIDEO_REPROBE_FAILED = "VIDEO_REPROBE_FAILED"

BASE_URL = "https://download.nvidia.com/XFree86/Linux-x86_64"
# Whole rank (download + extract + re-probe). 316 s is the fleet's median cold start, not a boot wall: the
# unclaimed pod is cancelled at 555 s, so the wall does not grow; the download gets faster instead (ranges).
RANK_WALL_S = 150.0
# Time the download must leave for extraction and the fresh re-probe.
AFTER_DOWNLOAD_RESERVE_S = 40.0
EXTRACT_TIMEOUT_S = 60.0
SHA_FETCH_TIMEOUT_S = 15.0
MIN_BYTES_PER_S = 1_000_000          # stall floor, measured per window
STALL_WINDOW_S = 10.0
CHUNK = 1 << 20
RANGE_STREAMS = 4
POLL_S = 0.5                         # supervisor wake-up (real seconds); every decision uses the injected clock
JOIN_DEADLINE_S = 5.0                # real seconds to wait for every worker after a stop
# Disk: archive + what --extract-only unpacks (1.2-1.5 GB measured) + the kept set (<= ~160 MB).
EXTRACT_BYTES = 1_600_000_000
KEPT_BYTES = 200_000_000
UNKNOWN_ARCHIVE_BYTES = 450_000_000

ALIAS = "libEGL_nvidia.so.0"
MANIFEST = "nvidia_icd.json"
# The video set a container runtime that grants no `video` capability never injects (receipt
# docs/research/pod-driver-userspace-video-libs.md). libnvidia-opticalflow stays OUT: nothing in the pod
# opens it, so keeping it would only widen what we copy for no consumer.
VIDEO_STEMS = ("libnvidia-encode", "libnvcuvid")
VIDEO_ALIAS_STEMS = VIDEO_STEMS


def allowlist(version: str) -> tuple[str, ...]:
    graphics = tuple(f"{stem}.so.{version}" for stem in (
        "libEGL_nvidia", "libnvidia-eglcore", "libnvidia-glsi", "libnvidia-glvkspirv", "libnvidia-gpucomp"))
    return graphics + tuple(f"{stem}.so.{version}" for stem in VIDEO_STEMS)


def video_aliases(version: str) -> dict[str, str]:
    """alias name -> the versioned file it must point at (what the loader asks for is the .so.1 name)."""
    return {f"{stem}.so.1": f"{stem}.so.{version}" for stem in VIDEO_ALIAS_STEMS}


def kept_names(version: str) -> set[str]:
    return {*allowlist(version), ALIAS, *video_aliases(version), MANIFEST}


@dataclass
class FetchResult:
    ok: bool
    reason: str = ""
    detail: str = ""
    lib_dir: str = ""
    manifest: str = ""
    elapsed: float = 0.0
    cap: float = RANK_WALL_S
    transfer: str = ""          # "bytes=<current-attempt completed>/<total> streams=<k>"; empty before any download
    source: str = ""            # "mirror" | "nvidia" once provisioning ran; empty for a reused directory
    mirror: str = ""            # "" = not asked or it worked; else "not_mirrored" | "error:<short>"


class Response(Protocol):
    status: int
    length: int | None
    # optional, read with getattr: content_range, content_encoding

    def chunks(self) -> Iterable[bytes]: ...
    def close(self) -> None: ...


Transport = Callable[..., Response]      # (url, timeout) or (url, timeout, headers=...)
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
        self.content_range = r.headers.get("Content-Range")
        self.content_encoding = r.headers.get("Content-Encoding")

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


def default_transport(url: str, timeout: float, headers: dict | None = None) -> Response:
    import requests
    try:
        return _HttpResponse(requests.get(url, stream=True, headers=headers,
                                          timeout=(min(10.0, timeout), timeout)))
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
        for alias, target in video_aliases(version).items():
            if os.readlink(final / alias) != target:
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


class _Progress:
    """What the current download attempt completed; the refusal trailer prints it."""

    def __init__(self) -> None:
        self.done = 0
        self.total: int | None = None
        self.streams = 0
        self._lock = threading.Lock()

    def reset(self, total: int | None, streams: int) -> None:
        with self._lock:
            self.done, self.total, self.streams = 0, total, streams

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n

    def text(self) -> str:
        if not self.streams:
            return ""
        return f"bytes={self.done}/{self.total if self.total is not None else '?'} streams={self.streams}"


def _with_transfer(detail: str, progress: _Progress) -> str:
    text = progress.text()
    return f"{detail} {text}" if text and detail else (text or detail)


def _check_disk(archive: int, base, disk_usage) -> None:
    need, free = archive + EXTRACT_BYTES + KEPT_BYTES, disk_usage(str(base)).free
    if free < need:
        raise _Refusal(INSUFFICIENT_DISK, f"need {need // 2**20} MiB, free {free // 2**20} MiB")


def _probe_ranges(transport: Transport, url: str, timeout: float) -> int | None:
    """Total length when the host answers a one-byte ranged GET with a proper 206, else None (single stream)."""
    try:
        resp = transport(url, timeout, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"})
    except (TimeoutError, OSError):
        return None
    try:
        if resp.status != 206 or getattr(resp, "content_encoding", None):
            return None
        m = re.fullmatch(r"bytes 0-0/(\d+)", getattr(resp, "content_range", None) or "")
        length = int(m.group(1)) if m else 0
        return length if length >= RANGE_STREAMS else None
    finally:
        resp.close()


def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    while view:
        n = os.pwrite(fd, view, offset)
        if n <= 0:
            raise OSError(errno.EIO, "pwrite wrote nothing")
        view, offset = view[n:], offset + n


class _RangeFailure(Exception):
    """One range went wrong for a reason that is not global: the single-stream fallback may still work."""


def _range_worker(transport, url, timeout, fd, length, start, end, progress, stop, responses, lock) -> None:
    resp = transport(url, timeout, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with lock:
        responses.append(resp)
        if stop.is_set():
            resp.close()
            return
    try:
        # all of this is checked before a single body byte is read or written
        if resp.status != 206:
            raise _RangeFailure(f"range {start}-{end}: HTTP {resp.status}")
        if getattr(resp, "content_encoding", None):
            raise _RangeFailure(f"range {start}-{end}: content-encoding")
        if getattr(resp, "content_range", None) != f"bytes {start}-{end}/{length}":
            raise _RangeFailure(f"range {start}-{end}: wrong Content-Range")
        if resp.length is not None and resp.length != end - start + 1:
            raise _RangeFailure(f"range {start}-{end}: wrong Content-Length")
        off = start
        try:
            for chunk in resp.chunks():
                if stop.is_set():
                    return
                if off + len(chunk) > end + 1:
                    raise _RangeFailure(f"range {start}-{end}: body longer than asked")
                _pwrite_all(fd, chunk, off)
                off += len(chunk)
                progress.add(len(chunk))
        except OSError as e:
            if e.errno == errno.ENOSPC:
                raise _Refusal(INSUFFICIENT_DISK, "disk full while writing") from e
            raise
        if off != end + 1 and not stop.is_set():
            raise _RangeFailure(f"range {start}-{end}: body short by {end + 1 - off} bytes")
    finally:
        resp.close()


def _join(threads) -> bool:
    """True when some worker is still alive after the bounded real-time deadline."""
    deadline = time.monotonic() + JOIN_DEADLINE_S
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    return any(t.is_alive() for t in threads)


def _stop_and_join(stop, responses, lock, threads) -> bool:
    """Stop every worker, close every response so a blocked read returns, then join."""
    stop.set()
    with lock:
        pending = list(responses)
    for resp in pending:
        try:
            resp.close()
        except Exception:  # noqa: BLE001 — closing must not mask the real verdict
            pass
    return _join(threads)


def _download_ranged(transport, url, run_file, length, base, disk_usage, remaining, clock, progress) -> bool:
    """True = the whole file is on disk, every byte from a verified range. False = fall back to one stream.
    Global failures (wall, stall, disk) raise and are never retried."""
    _check_disk(length, base, disk_usage)                          # on the probed length, before preallocating
    left = remaining() - AFTER_DOWNLOAD_RESERVE_S
    if left <= 0:
        raise _Refusal(DOWNLOAD_TIMEOUT, "wall exhausted before the download")
    timeout = min(SHA_FETCH_TIMEOUT_S, left)
    progress.reset(length, RANGE_STREAMS)
    fd = os.open(run_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stop, all_done, lock = threading.Event(), threading.Event(), threading.Lock()
    responses: list = []
    failures: list[BaseException] = []
    state = {"left": RANGE_STREAMS}
    threads: list[threading.Thread] = []
    verdict = ""
    stuck = False
    try:
        try:
            os.ftruncate(fd, length)
        except OSError as e:
            if e.errno == errno.ENOSPC:
                raise _Refusal(INSUFFICIENT_DISK, "disk full while preallocating") from e
            raise

        def run(start: int, end: int) -> None:
            try:
                _range_worker(transport, url, timeout, fd, length, start, end, progress, stop, responses, lock)
            except BaseException as e:  # noqa: BLE001 — recorded here, judged by the supervisor
                with lock:
                    if not stop.is_set():
                        failures.append(e)
                        stop.set()
            finally:
                with lock:
                    state["left"] -= 1
                    if state["left"] == 0:
                        all_done.set()

        win_start, win_done = clock(), progress.done               # the stall window opens before any worker runs
        for i in range(RANGE_STREAMS):
            t = threading.Thread(target=run, daemon=True, name=f"driver-range-{i}",
                                 args=(i * length // RANGE_STREAMS, (i + 1) * length // RANGE_STREAMS - 1))
            threads.append(t)
            t.start()

        # supervisor: wall and stall are judged here on the injected clock, summed over streams, not per chunk
        while True:
            all_done.wait(POLL_S)
            with lock:
                if failures:
                    break
                finished = state["left"] == 0
            now = clock()
            if remaining() - AFTER_DOWNLOAD_RESERVE_S <= 0:
                verdict = f"wall hit after {progress.done} bytes"
                break
            if now - win_start >= STALL_WINDOW_S:
                if progress.done - win_done < MIN_BYTES_PER_S * (now - win_start):
                    verdict = f"stalled below {MIN_BYTES_PER_S} B/s after {progress.done} bytes"
                    break
                win_start, win_done = now, progress.done
            if finished:
                break
        stuck = _stop_and_join(stop, responses, lock, threads) if (verdict or failures) else _join(threads)
    finally:
        if not any(t.is_alive() for t in threads):                 # never close an fd a live writer still uses
            os.close(fd)
    if stuck:
        raise _Refusal(DOWNLOAD_TIMEOUT, "download workers did not stop in time")
    if verdict:
        raise _Refusal(DOWNLOAD_TIMEOUT, verdict)
    for exc in failures:
        if isinstance(exc, _Refusal):
            raise exc
    return not failures


def _sha_file(path: Path, remaining) -> str:
    """SHA-256 of the finished file, with a deadline check per block (inside the post-download reserve)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if remaining() - AFTER_DOWNLOAD_RESERVE_S / 2 <= 0:
                raise _Refusal(DOWNLOAD_TIMEOUT, "wall exhausted while hashing")
            block = f.read(CHUNK)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _download_single(transport, url, run_file, base, disk_usage, remaining, clock, progress) -> str:
    """One stream, hashed as it arrives. Returns the hex digest."""
    name = url.rsplit("/", 1)[-1]
    left = remaining() - AFTER_DOWNLOAD_RESERVE_S
    if left <= 0:
        raise _Refusal(DOWNLOAD_TIMEOUT, "wall exhausted before the download")
    try:
        resp = transport(url, min(SHA_FETCH_TIMEOUT_S, left))
    except TimeoutError as e:
        raise _Refusal(DOWNLOAD_TIMEOUT, str(e)) from e
    except OSError as e:
        raise _Refusal(DRIVER_PACKAGE_UNAVAILABLE, type(e).__name__) from e
    try:
        if resp.status != 200:
            raise _Refusal(DRIVER_PACKAGE_UNAVAILABLE, f"{name}: HTTP {resp.status}")
        _check_disk(resp.length or UNKNOWN_ARCHIVE_BYTES, base, disk_usage)
        progress.reset(resp.length, 1)
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
                    progress.add(len(chunk))
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
    return digest.hexdigest()


def fetch_driver_userspace(
    version: str,
    base_dir: str | os.PathLike,
    *,
    transport: Transport | None = None,
    extractor: Extractor | None = None,
    clock: Callable[[], float] = time.monotonic,
    disk_usage: Callable[[str], object] = shutil.disk_usage,
    cap: float = RANK_WALL_S,
    mirror: Callable[[str], "artifact.TarRef | None"] | None = None,
) -> FetchResult:
    """Provision <base_dir>/nvidia-<version>/ and return where. Never raises: every failure is a named
    FetchResult, and nothing but a verified directory is left behind. `mirror` (our store's userspace tar)
    replaces only the download/sha/extract; any failure of it falls back to NVIDIA inside the same `cap`."""
    start = clock()
    progress = _Progress()
    state = {"source": "", "mirror": ""}

    def done(**kw) -> FetchResult:
        return FetchResult(elapsed=clock() - start, cap=cap, transfer=progress.text(),
                           source=state["source"], mirror=state["mirror"], **kw)

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
                   extractor or default_extractor, clock, start, cap, disk_usage, progress, mirror, state)
    except _Refusal as r:
        return done(ok=False, reason=r.reason, detail=_with_transfer(r.detail, progress))
    except OSError as e:
        return done(ok=False, reason=EXTRACTION_FAILED, detail=_with_transfer(f"{type(e).__name__}: {e}", progress))
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
    return done(ok=True, lib_dir=str(final), manifest=str(final / MANIFEST))


def _provision(version, base, final, work, transport, extractor, clock, start, cap, disk_usage, progress,
               mirror=None, state=None) -> None:
    """Records in `state` the source ("mirror"|"nvidia") and mirror note ("" unless the mirror was asked and
    failed: not_mirrored | error:<short>) — set before any NVIDIA refusal can raise, so the trailer has it."""
    state = state if state is not None else {}
    def remaining() -> float:
        return cap - (clock() - start)

    target = work / "extract"
    state["source"], state["mirror"] = "nvidia", ""
    if mirror is not None:
        state["mirror"] = _try_mirror(mirror, version, work, target, remaining)
        if not state["mirror"]:
            state["source"] = "mirror"
    if state["source"] == "nvidia":
        _nvidia_steps(version, base, work, target, transport, extractor, clock, remaining, disk_usage, progress)
    _finish(version, final, work, target)


def _fetch_within_wall(ref, tar_path, remaining) -> None:
    """artifact.fetch_verified in a daemon thread, supervised on the injected clock: the download is held to
    `remaining() - AFTER_DOWNLOAD_RESERVE_S`, so a slow or hung mirror costs the wall, never more."""
    box: list[BaseException] = []
    finished = threading.Event()

    def run() -> None:
        try:
            artifact.fetch_verified(ref, tar_path)
        except BaseException as e:  # noqa: BLE001 — handed to the supervisor
            box.append(e)
        finally:
            finished.set()

    threading.Thread(target=run, daemon=True, name="mirror-fetch").start()
    while not finished.wait(POLL_S):
        if remaining() - AFTER_DOWNLOAD_RESERVE_S <= 0:
            raise TimeoutError("wall hit during the mirror download")
    if box:
        raise box[0]


def _try_mirror(mirror, version, work, target, remaining) -> str:
    """Steps 1-3 from our store: "" on success, else the trailer token. The extract dir is left empty on any
    failure so the NVIDIA path starts clean; the wall is the one `remaining()` measures, so a fallback spends
    only what is left (no second wall)."""
    try:
        if remaining() - AFTER_DOWNLOAD_RESERVE_S <= 0:
            return "error:wall"
        ref = mirror(version)
        if ref is None:
            return "not_mirrored"
        tar_path = work / "userspace.tar.xz"
        _fetch_within_wall(ref, tar_path, remaining)
        if remaining() - AFTER_DOWNLOAD_RESERVE_S / 2 <= 0:
            raise TimeoutError("wall exhausted after the mirror fetch")
        artifact.safe_extract(tar_path, target)      # tar symlinks never trusted: a link member is refused below
        for lib in (*allowlist(version), "nvidia_icd.json"):
            p = target / lib
            if not p.is_file() or p.is_symlink():
                raise ValueError(f"missing {lib}")
        api = json.loads((target / "nvidia_icd.json").read_text())["ICD"]["api_version"]
        if not isinstance(api, str) or not DRIVER_VERSION_RE.match(api):
            raise ValueError("malformed api_version")
        return ""
    except Exception as e:  # noqa: BLE001 — any mirror failure means "use NVIDIA"
        shutil.rmtree(target, ignore_errors=True)
        (work / "userspace.tar.xz").unlink(missing_ok=True)
        return f"error:{type(e).__name__}"


def _nvidia_steps(version, base, work, target, transport, extractor, clock, remaining, disk_usage, progress) -> None:
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

    # 2. fetch it: parallel byte ranges when the host proves it serves them, else (or when a range fails for a
    #    non-global reason) one stream. Either way the whole file is hashed and compared exactly.
    url = f"{BASE_URL}/{version}/{name}"
    run_file = work / name
    left = remaining() - AFTER_DOWNLOAD_RESERVE_S
    if left <= 0:
        raise _Refusal(DOWNLOAD_TIMEOUT, "wall exhausted before the download")
    got = None
    length = _probe_ranges(transport, url, min(SHA_FETCH_TIMEOUT_S, left))
    if length is not None:
        if _download_ranged(transport, url, run_file, length, base, disk_usage, remaining, clock, progress):
            got = _sha_file(run_file, remaining)
        else:
            run_file.unlink(missing_ok=True)
            run_file = work / f"{name}.single"            # a FRESH name: nothing of the failed attempt is reused
    if got is None:
        got = _download_single(transport, url, run_file, base, disk_usage, remaining, clock, progress)
    if got != expected:
        raise _Refusal(CHECKSUM_MISMATCH, f"{name}: got {got[:12]}, want {expected[:12]}")

    # 3. only now run the vendor script; it creates the target itself and refuses one that exists
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


def _finish(version, final, work, target) -> None:
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
    for alias, tgt in video_aliases(version).items():
        os.symlink(tgt, stage / alias)
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
