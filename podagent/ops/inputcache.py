"""One fetch per OBJECT for the pod's whole life, not one per chain. A presigned URL carries a signature and
an expiry, so the same object minted twice is two different strings, and the per-chain memo keyed on that
string — `source_axis` binds the master in three chains, so one stage moved the recording three times."""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

CACHE_ENV = "OPS_INPUT_CACHE"
CACHE_DEFAULT = "/var/cache/monty/inputs"
DISABLE_ENV = "OPS_INPUT_CACHE_OFF"
MAX_GB_ENV = "OPS_INPUT_CACHE_MAX_GB"
MAX_GB_DEFAULT = 24.0
DONE = ".complete"

_lock = threading.Lock()
_locks: dict[str, threading.Lock] = {}


def enabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip().lower() not in ("1", "true", "yes")


def root() -> Path:
    return Path(os.environ.get(CACHE_ENV) or CACHE_DEFAULT)


def object_key(url: str) -> str | None:
    """The OBJECT a presigned URL names — host + path. The query is the CAPABILITY, not the identity: it
    changes on every mint, which is exactly what defeated the old cache."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc or not parts.path:
        return None
    return f"{parts.netloc}{parts.path}"


def _slot(key: str) -> Path:
    return root() / hashlib.sha256(key.encode()).hexdigest()[:32]


def _key_lock(key: str) -> threading.Lock:
    with _lock:
        return _locks.setdefault(key, threading.Lock())


def _entries() -> list[tuple[float, int, Path]]:
    """(mtime, bytes, slot) for every COMPLETE entry — a half-written one is nobody's to evict."""
    out: list[tuple[float, int, Path]] = []
    for slot in root().glob("*"):
        payload = slot / "payload"
        if not (slot / DONE).exists() or not payload.is_file():
            continue
        try:
            st = payload.stat()
        except OSError:
            continue
        out.append((st.st_mtime, st.st_size, slot))
    return out


def prune(keep_bytes: int | None = None, log=None) -> int:
    """Evict oldest-first until the cache fits. A pod's disk is smaller than the media it sees in a shift,
    so an unbounded cache trades one bug for a fuller one."""
    import shutil
    cap = keep_bytes if keep_bytes is not None else int(
        float(os.environ.get(MAX_GB_ENV) or MAX_GB_DEFAULT) * 1e9)
    entries = sorted(_entries())
    total = sum(size for _m, size, _s in entries)
    freed = 0
    for _mtime, size, slot in entries:
        if total <= cap:
            break
        shutil.rmtree(slot, ignore_errors=True)
        total -= size
        freed += size
        if log:
            log(f"[input-cache] evicted {slot.name} ({size / 1e6:.0f} MB)")
    return freed


def get(url: str, download, log=None) -> Path | None:
    """The object's local path, fetched at most once per pod. None = not cacheable, the caller transfers it
    itself. `download(url, dest)` stays the caller's, so this module owns the naming and nothing else."""
    if not enabled():
        return None
    key = object_key(url)
    if key is None:
        return None
    slot = _slot(key)
    payload, sentinel = slot / "payload", slot / DONE
    if sentinel.exists() and payload.exists():
        if log:
            log(f"[input-cache] hit {key} ({payload.stat().st_size / 1e6:.0f} MB) — no transfer")
        return payload
    # per OBJECT, never globally: two chains pulling DIFFERENT inputs must still overlap
    with _key_lock(key):
        if sentinel.exists() and payload.exists():
            return payload
        slot.mkdir(parents=True, exist_ok=True)
        tmp = slot / f"payload.{os.getpid()}.part"
        download(url, tmp)
        # sentinel AFTER the rename: a fetch killed mid-write must read as ABSENT, never as usable
        tmp.replace(payload)
        sentinel.write_text("", encoding="utf-8")
        if log:
            log(f"[input-cache] stored {key} ({payload.stat().st_size / 1e6:.0f} MB)")
    prune(log=log)      # outside the per-object lock: eviction must not hold up another object's transfer
    return payload
