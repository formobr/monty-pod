"""One content-addressed cache for every heavy tree the pod is handed.

Weights were the first tenant (`weights.py`); the Remotion bundle is the second (`bundle.py`). Both are
"a tar arrives by presigned URL, verify it, unpack it once, reuse it while the pod is warm", so the
fetch/verify/extract/publish core lives here and each tenant keeps only what is genuinely its own
(weights: locating config.json; bundle: building a per-job workspace).

CACHE KEY IS THE CONTENT HASH, never a name. The venv-tarball lane learned this the hard way: its first
version keyed by a fixed filename, so a changed dependency set silently served a stale env. Here a
different tar is a different sha256 is a different directory — a stale hit is not representable.

`.complete` is written only after a verified extract, so a fetch killed mid-write leaves a directory the
next run treats as ABSENT rather than as usable. It must also not WEDGE the slot: publish takes over a
sentinel-less directory sitting on the target instead of failing forever. Those are two separate
guarantees and they have two separate tests — a single "interrupted fetch" test passes vacuously while
the second one is broken.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

import requests

from .sanitize import safe_text

_CHUNK = 8 << 20
DONE = ".complete"
_PROGRESS_EVERY_S = 20.0

STALL_WHY = """
A TRANSFER THAT CRAWLS IS NOT A TRANSFER THAT IS WORKING.

Measured 2026-08-06 on a warm-up that never finished: the 3 GB weights pull ran at 101 MB/s for its first
minute and then decayed — 75, 34, 25, 20, and finally 17 MB/s AVERAGE, which is 9 MB moved in the last eight
minutes. Nothing here stopped it. The only thing that did was the caller's 600 s checkpoint wall, from
outside, which cannot say WHY it gave up and cannot try a different host.

So progress alone was never the health signal: a stream delivering 20 KB/s is still "making progress" and
will hold a rented box until someone else's deadline fires. The floor below turns that into a named, fast
failure the caller can act on — the provisioner already knows how to blacklist an offer and take another one,
but only if something tells it this one is bad.

THE FLOOR IS DELIBERATELY FAR BELOW ANY HEALTHY LINK. Observed good pulls sit at 50-100 MB/s; this refuses
only at three orders of magnitude below that, so a merely slow provider still completes and only a stalled
one dies. It is measured over a WINDOW, not from the start, because an average is dragged down by the very
stall it is supposed to detect — by the time a from-the-start average looks bad, the minutes are already gone.
"""

# Bytes per second, averaged over the window between two progress ticks, under which the transfer is declared
# stalled (STALL_WHY). Env-overridable because the right number is a property of the link, not of this code.
_MIN_BYTES_PER_S = float(os.environ.get("ARTIFACT_MIN_BYTES_PER_S", str(256 * 1024)))

# Where a long fetch reports to. Default is stderr only, which is invisible from the control plane — the
# 4.3 GB weights pull is the one place a pod can be alive and mute for ten minutes.
Progress = Callable[[str], None]


class TransferStalled(RuntimeError):
    """The source stopped delivering at a usable rate (STALL_WHY). A distinct type so a caller can tell
    "this HOST is bad, take another" from "this artifact is bad, do not retry it anywhere"."""


class TarRef(Protocol):
    """The shape every artifact reference shares: where to GET it, and what it must hash to."""

    url: str
    sha256: str
    size: int | None


def log(msg: str) -> None:
    print(f"[podagent] {safe_text(msg)}", file=sys.stderr, flush=True)


def cache_root(env_var: str, default: str) -> Path:
    return Path(os.environ.get(env_var, default))


def safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract, refusing any member that would escape `dest` (absolute path, `..`, or a link out).

    The tar is content-verified before we get here, so this is defence against a compromised ORIGIN, not
    against corruption — but an executor that unpacks whatever it is handed into an arbitrary path is a
    hole regardless of who signed the URL.
    """
    dest_resolved = dest.resolve()
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not (target == dest_resolved or dest_resolved in target.parents):
                raise ValueError(f"tar escapes its directory: {member.name!r}")
            if member.issym() or member.islnk():
                link = (target.parent / member.linkname).resolve()
                if not (link == dest_resolved or dest_resolved in link.parents):
                    raise ValueError(f"tar links outside its directory: {member.name!r}")
        tf.extractall(dest, filter="data")   # belt-and-braces over the explicit check above


def _chunks(url: str):
    """Byte stream for a presigned GET, or for a `file://` url.

    `file://` is how the LOCAL transport reaches this code — podagent.cp.download already makes the same
    degradation for spec inputs, and it matters here for the same reason: the laptop path then exercises
    the REAL fetch/verify/extract/publish sequence rather than a shortcut around it, so an artifact that
    would fail to verify or unpack on a pod fails locally first. `requests` has no file:// handler, so
    this split is not optional.
    """
    if url.startswith("file://"):
        with open(url_to_path(url), "rb") as fh:
            while chunk := fh.read(_CHUNK):
                yield chunk
        return
    with requests.get(url, stream=True, timeout=(30, 600)) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(_CHUNK):
            if chunk:
                yield chunk


def url_to_path(url: str) -> Path:
    from urllib.parse import unquote, urlparse
    return Path(unquote(urlparse(url).path))


# Not OPS_MAX_TRANSFERS (cp.py): that sizes the box-wide connection pool across many small concurrent ops
# steps (16-64); here N connections split ONE object, so it gets its own, smaller-default knob.
_FETCH_WORKERS_ENV = "ARTIFACT_FETCH_WORKERS"
_FETCH_WORKERS_DEFAULT = 6

# Below this, one more range costs more in extra TCP+TLS handshakes than the parallelism saves.
_MIN_PART_BYTES = 32 << 20

_RANGE_PROBE_TIMEOUT = (10, 20)


def range_fetch_width() -> int:
    """Parallel byte-range connections for one artifact (env ARTIFACT_FETCH_WORKERS, default 6) — not
    `fetch_width`: `infer_cliprank.fetch_width()` already owns that name for the unrelated tile pool."""
    raw = os.environ.get(_FETCH_WORKERS_ENV, "").strip()
    try:
        return max(1, int(raw)) if raw else _FETCH_WORKERS_DEFAULT
    except ValueError:
        return _FETCH_WORKERS_DEFAULT


def _range_probe(url: str) -> int | None:
    """Total size if `url` answers a byte-range GET with 206, else None — a HEAD alone cannot prove a
    presigned GET-only signature will also honour Range, so the probe IS the real request shape."""
    if url.startswith("file://"):
        return None
    try:
        resp = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=_RANGE_PROBE_TIMEOUT)
    except requests.RequestException:
        return None
    with resp:
        if resp.status_code != 206:
            return None
        try:
            return int(resp.headers.get("Content-Range", "").rsplit("/", 1)[-1])
        except ValueError:
            return None


def _content_range_start(headers: dict) -> int | None:
    try:
        return int(headers.get("Content-Range", "").split(" ", 1)[1].split("-", 1)[0])
    except (IndexError, ValueError):
        return None


def _fetch_range(url: str, dst: Path, start: int, end: int, counter: list[int], lock: threading.Lock,
                 abort: threading.Event) -> None:
    """One byte range straight into `dst` at its own offset — bounded memory (one `_CHUNK` per worker);
    `abort` is checked between chunks so a live sibling stops within about one chunk of a fatal sibling."""
    with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 600)) as resp:
        if resp.status_code != 206:
            raise TransferStalled(f"range {start}-{end} lost 206 mid-fetch (got {resp.status_code})")
        # a wrong offset is a BAD HOST, not corrupt bytes — catch it now, not as a sha256 mismatch 4.6 GB later
        if _content_range_start(resp.headers) != start:
            raise TransferStalled(f"range {start}-{end}: origin answered from a different offset")
        got = 0
        with open(dst, "r+b") as fh:
            fh.seek(start)
            for chunk in resp.iter_content(_CHUNK):
                if abort.is_set():
                    return
                if not chunk:
                    continue
                fh.write(chunk)
                got += len(chunk)
                with lock:
                    counter[0] += len(chunk)
    if abort.is_set():
        return
    expected = end - start + 1
    if got != expected:
        raise TransferStalled(f"range {start}-{end} delivered {got} of {expected} bytes")


def _download_ranged(ref: TarRef, dst: Path, total: int, progress: "Progress | None", label: str) -> int:
    """N ranged GETs run concurrently into `dst`. NEVER `with ex:` — `abort` (checked between chunks) stops
    a live sibling fast; `shutdown(wait=False)` below stays non-blocking regardless, for one that never
    gets another chunk to check it on."""
    width = min(range_fetch_width(), max(1, total // _MIN_PART_BYTES))
    part = -(-total // width)
    bounds = [(i * part, min((i + 1) * part, total) - 1) for i in range(width) if i * part < total]
    # safe to pre-size: the only caller stages into a dir ensure_tree always removes on any failure
    with dst.open("wb") as fh:
        fh.truncate(total)
    counter = [0]
    lock = threading.Lock()
    abort = threading.Event()
    t0 = last = time.monotonic()
    at_last = 0
    ex = cf.ThreadPoolExecutor(max_workers=len(bounds), thread_name_prefix="artifact-range")
    try:
        pending = {ex.submit(_fetch_range, ref.url, dst, start, end, counter, lock, abort)
                   for start, end in bounds}
        while pending:
            done, pending = cf.wait(pending, timeout=_PROGRESS_EVERY_S, return_when=cf.FIRST_EXCEPTION)
            for fut in done:
                exc = fut.exception(timeout=0)
                if exc is not None:
                    abort.set()
                    for other in pending:
                        other.cancel()
                    raise exc
            now = time.monotonic()
            total_now = counter[0]
            window_s = now - last
            window_rate = (total_now - at_last) / max(window_s, 1e-6)
            # a window with zero bytes so far is TTFB/connect, not a rate — download_verified structurally
            # cannot run its check before a chunk exists (:275); this mirrors that instead of firing on 0/0
            first_byte_tick = at_last == 0 and total_now > 0
            last, at_last = now, total_now
            if pending and total_now > 0 and not first_byte_tick and window_rate < _MIN_BYTES_PER_S:
                abort.set()
                raise TransferStalled(
                    f"{label or 'artifact'} moved {window_rate / 1e6:.2f} MB/s over the last "
                    f"{window_s:.0f}s ({total_now / 1e6:.0f} MB so far, {len(bounds)} parallel parts) — "
                    f"below the {_MIN_BYTES_PER_S / 1e6:.2f} MB/s floor; the source has stalled")
            if progress is not None:
                pct = f" ({100 * total_now / total:.0f}%)" if total else ""
                progress(f"{label or 'artifact'} fetch {total_now / 1e6:.0f} MB{pct} · "
                         f"{total_now / 1e6 / max(now - t0, 1e-6):.1f} MB/s avg · "
                         f"{window_rate / 1e6:.1f} MB/s now")
        return total
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _verify(dst: Path, ref: TarRef, total: int) -> int:
    """Hash the assembled file in one sequential pass — parallel parts land out of write order across
    threads, so the digest can only be taken over the finished file, never streamed part-by-part."""
    digest = hashlib.sha256()
    with dst.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    got = digest.hexdigest()
    if got != ref.sha256:
        raise ValueError(f"sha256 mismatch: expected {ref.sha256}, got {got} ({total} bytes)")
    if ref.size is not None and total != ref.size:
        raise ValueError(f"size mismatch: expected {ref.size} bytes, got {total}")
    return total


def fetch_verified(ref: TarRef, dst: Path, progress: "Progress | None" = None, label: str = "") -> int:
    """Ranged-parallel when the origin proves it (206 to a probe GET, size worth splitting), else exactly
    today's single-stream download_verified — a missing capability costs speed, never a failure."""
    total = _range_probe(ref.url)
    if total is None or total < 2 * _MIN_PART_BYTES:
        return download_verified(ref, dst, progress, label)
    got = _download_ranged(ref, dst, total, progress, label)
    return _verify(dst, ref, got)


def download_verified(ref: TarRef, dst: Path, progress: "Progress | None" = None,
                      label: str = "") -> int:
    """Stream the tar to `dst`, hashing as we go. A digest mismatch raises — we never extract unverified
    bytes, so a truncated or swapped object fails here instead of surfacing as mystery-bad output."""
    digest = hashlib.sha256()
    total = 0
    t0 = last = time.monotonic()
    at_last = 0                                   # bytes at the previous tick, for the WINDOW rate
    with dst.open("wb") as fh:
        for chunk in _chunks(ref.url):
            digest.update(chunk)
            fh.write(chunk)
            total += len(chunk)
            now = time.monotonic()
            if now - last >= _PROGRESS_EVERY_S:
                window_s = now - last
                window_rate = (total - at_last) / max(window_s, 1e-6)
                last, at_last = now, total
                if window_rate < _MIN_BYTES_PER_S:
                    # LOUD AND FAST, so the caller can drop this host and take another instead of holding a
                    # rented box until somebody else's deadline (STALL_WHY).
                    raise TransferStalled(
                        f"{label or 'artifact'} moved {window_rate / 1e6:.2f} MB/s over the last "
                        f"{window_s:.0f}s ({total / 1e6:.0f} MB so far) — below the "
                        f"{_MIN_BYTES_PER_S / 1e6:.2f} MB/s floor; the source has stalled")
                if progress is not None:
                    pct = f" ({100 * total / ref.size:.0f}%)" if ref.size else ""
                    # AVERAGE for the human, WINDOW for the decision — the average is what hid this stall for
                    # eight minutes, because it decays instead of dropping.
                    progress(f"{label or 'artifact'} fetch {total / 1e6:.0f} MB{pct} · "
                             f"{total / 1e6 / max(now - t0, 1e-6):.1f} MB/s avg · "
                             f"{window_rate / 1e6:.1f} MB/s now")
    got = digest.hexdigest()
    if got != ref.sha256:
        raise ValueError(f"sha256 mismatch: expected {ref.sha256}, got {got} ({total} bytes)")
    if ref.size is not None and total != ref.size:
        raise ValueError(f"size mismatch: expected {ref.size} bytes, got {total}")
    return total


def ensure_tree(ref: TarRef, root: Path, label: str = "", progress: "Progress | None" = None) -> Path:
    """Return the local directory holding this exact content, fetching only on a miss.

    Idempotent and safe to call per job: a warm pod pays the transfer exactly once per distinct artifact.
    The returned directory is treated as IMMUTABLE by every caller — a tenant that needs to write must
    copy out first, or one job silently poisons the cache for every later job (see bundle.workspace).
    """
    what = label or ref.sha256[:12]
    dest = root / ref.sha256
    if (dest / DONE).is_file():
        log(f"{what} — cache HIT {dest}")
        return dest

    root.mkdir(parents=True, exist_ok=True)
    log(f"{what} — cache MISS, fetching {ref.size or '?'} bytes")
    if progress is not None:
        progress(f"{what} — cache MISS, fetching {(ref.size or 0) / 1e6:.0f} MB")
    t0 = time.monotonic()
    # Stage into a sibling temp dir and rename: a concurrent or killed fetch can never publish a partial
    # tree under the content hash.
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{ref.sha256[:12]}-"))
    try:
        tar_path = staging / "artifact.tar"
        total = fetch_verified(ref, tar_path, progress, what)
        unpacked = staging / "d"
        unpacked.mkdir()
        if progress is not None:
            progress(f"{what} — {total / 1e6:.0f} MB verified, extracting")
        safe_extract(tar_path, unpacked)
        tar_path.unlink()
        (unpacked / DONE).write_text(ref.sha256)
        # An unfinished directory sitting on the target (killed fetch under an older layout, half-restored
        # backup) must not wedge the cache forever — it has no sentinel, so it is not an artifact, and we
        # take the slot over.
        if dest.exists() and not (dest / DONE).is_file():
            shutil.rmtree(dest, ignore_errors=True)
        try:
            unpacked.rename(dest)
        except OSError:
            # another job won the race; its copy is byte-identical by construction
            if not (dest / DONE).is_file():
                raise
        dt = time.monotonic() - t0
        log(f"{what} ready in {dt:.1f}s ({total / 1e6:.0f} MB, "
            f"{total / 1e6 / max(dt, 1e-6):.1f} MB/s) → {dest}")
        if progress is not None:
            progress(f"{what} ready in {dt:.0f}s ({total / 1e6:.0f} MB)")
        return dest
    finally:
        shutil.rmtree(staging, ignore_errors=True)
