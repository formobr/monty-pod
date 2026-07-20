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

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Protocol

import requests

_CHUNK = 8 << 20
DONE = ".complete"


class TarRef(Protocol):
    """The shape every artifact reference shares: where to GET it, and what it must hash to."""

    url: str
    sha256: str
    size: int | None


def log(msg: str) -> None:
    print(f"[podagent] {msg}", file=sys.stderr, flush=True)


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


def download_verified(ref: TarRef, dst: Path) -> int:
    """Stream the tar to `dst`, hashing as we go. A digest mismatch raises — we never extract unverified
    bytes, so a truncated or swapped object fails here instead of surfacing as mystery-bad output."""
    digest = hashlib.sha256()
    total = 0
    with dst.open("wb") as fh:
        for chunk in _chunks(ref.url):
            digest.update(chunk)
            fh.write(chunk)
            total += len(chunk)
    got = digest.hexdigest()
    if got != ref.sha256:
        raise ValueError(f"sha256 mismatch: expected {ref.sha256}, got {got} ({total} bytes)")
    if ref.size is not None and total != ref.size:
        raise ValueError(f"size mismatch: expected {ref.size} bytes, got {total}")
    return total


def ensure_tree(ref: TarRef, root: Path, label: str = "") -> Path:
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
    t0 = time.monotonic()
    # Stage into a sibling temp dir and rename: a concurrent or killed fetch can never publish a partial
    # tree under the content hash.
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{ref.sha256[:12]}-"))
    try:
        tar_path = staging / "artifact.tar"
        total = download_verified(ref, tar_path)
        unpacked = staging / "d"
        unpacked.mkdir()
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
        return dest
    finally:
        shutil.rmtree(staging, ignore_errors=True)
