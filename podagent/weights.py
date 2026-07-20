"""Model weights as a job INPUT, not image ballast.

The pod is a dumb executor that already receives every input as a presigned URL and holds no keys — a
checkpoint arrives the same way. `InferRequest.weights` carries a presigned GET for a tar of the model
directory plus that tar's sha256; the pod fetches it once, verifies it, extracts it, and hands the local
directory to `from_pretrained`.

CACHE KEY IS THE CONTENT HASH, never the model name. The venv-tarball lane learned this the hard way: its
first version keyed the cache by a fixed filename, so a changed dependency set silently served a stale
env. Here a different checkpoint is a different sha256 is a different directory — a stale hit is not
representable. `.complete` is written only after a verified extract, so a fetch killed mid-write leaves a
directory that the next run treats as absent rather than as a usable model.
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

import requests

from .models import WeightsRef

_CHUNK = 8 << 20
_DONE = ".complete"


def cache_root() -> Path:
    return Path(os.environ.get("WEIGHTS_CACHE", "/var/cache/monty/weights"))


def _log(msg: str) -> None:
    print(f"[podagent] {msg}", file=sys.stderr, flush=True)


def _safe_extract(tar_path: Path, dest: Path) -> None:
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
                raise ValueError(f"weights tar escapes its directory: {member.name!r}")
            if member.issym() or member.islnk():
                link = (target.parent / member.linkname).resolve()
                if not (link == dest_resolved or dest_resolved in link.parents):
                    raise ValueError(f"weights tar links outside its directory: {member.name!r}")
        tf.extractall(dest, filter="data")   # belt-and-braces over the explicit check above


def model_dir(root: Path) -> Path:
    """The directory inside an extracted tar that `from_pretrained` should be pointed at.

    Layout-agnostic on purpose: the seeded tars are HF-hub shaped (`<repo>/snapshots/<rev>/…`) but a flat
    tar of a model directory is just as valid. We locate the one directory holding a config.json rather
    than hard-coding either shape, so re-exporting the weights never silently breaks the pod.
    """
    if (root / "config.json").is_file():
        return root
    hits = sorted(p.parent for p in root.rglob("config.json"))
    # A hub tar carries refs/ + snapshots/<rev>/; deeper nesting means sub-configs, so prefer the shallowest.
    if not hits:
        raise ValueError(f"weights tar holds no config.json under {root}")
    shallowest = min(hits, key=lambda p: len(p.relative_to(root).parts))
    return shallowest


def _download_verified(ref: WeightsRef, dst: Path) -> int:
    """Stream the tar to `dst`, hashing as we go. A digest mismatch raises — we never extract unverified
    bytes, so a truncated or swapped object fails here instead of surfacing as mystery-bad inference."""
    digest = hashlib.sha256()
    total = 0
    with requests.get(ref.url, stream=True, timeout=(30, 600)) as resp:
        resp.raise_for_status()
        with dst.open("wb") as fh:
            for chunk in resp.iter_content(_CHUNK):
                if not chunk:
                    continue
                digest.update(chunk)
                fh.write(chunk)
                total += len(chunk)
    got = digest.hexdigest()
    if got != ref.sha256:
        raise ValueError(f"weights sha256 mismatch: expected {ref.sha256}, got {got} ({total} bytes)")
    if ref.size is not None and total != ref.size:
        raise ValueError(f"weights size mismatch: expected {ref.size} bytes, got {total}")
    return total


def ensure(ref: WeightsRef, model_id: str = "") -> Path:
    """Return a local directory holding the model, fetching it only if this exact content is not cached.

    Idempotent and safe to call per job: a warm pod pays the transfer exactly once per checkpoint.
    """
    root = cache_root()
    dest = root / ref.sha256
    if (dest / _DONE).is_file():
        _log(f"weights {model_id or ref.sha256[:12]} — cache HIT {dest}")
        return model_dir(dest)

    root.mkdir(parents=True, exist_ok=True)
    _log(f"weights {model_id or ref.sha256[:12]} — cache MISS, fetching {ref.size or '?'} bytes")
    t0 = time.monotonic()
    # Stage into a sibling temp dir and rename: a concurrent or killed fetch can never publish a partial
    # model under the content hash.
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{ref.sha256[:12]}-"))
    try:
        tar_path = staging / "weights.tar"
        total = _download_verified(ref, tar_path)
        unpacked = staging / "d"
        unpacked.mkdir()
        _safe_extract(tar_path, unpacked)
        tar_path.unlink()
        (unpacked / _DONE).write_text(ref.sha256)
        # An unfinished directory sitting on the target (older layout, half-restored backup) must not wedge
        # the cache forever — it has no sentinel, so it is not a model, and we take the slot over.
        if dest.exists() and not (dest / _DONE).is_file():
            shutil.rmtree(dest, ignore_errors=True)
        try:
            unpacked.rename(dest)
        except OSError:
            # another job won the race; its copy is byte-identical by construction
            if not (dest / _DONE).is_file():
                raise
        dt = time.monotonic() - t0
        _log(f"weights {model_id or ref.sha256[:12]} ready in {dt:.1f}s "
             f"({total / 1e6:.0f} MB, {total / 1e6 / max(dt, 1e-6):.1f} MB/s) → {dest}")
        return model_dir(dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
