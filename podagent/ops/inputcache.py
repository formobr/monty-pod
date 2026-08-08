"""One fetch per OBJECT for the pod's whole life, not one per chain. A presigned URL carries a signature and
an expiry, so the same object minted twice is two different strings, and the per-chain memo keyed on that
string — `source_axis` binds the master in three chains, so one stage moved the recording three times."""
from __future__ import annotations

import hashlib
import os
import shutil
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
_slot_locks: dict[str, threading.Lock] = {}


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
    slot_name = _slot(key).name
    with _lock:
        found = _locks.get(key)
        if found is None:
            found = _slot_locks.setdefault(slot_name, threading.Lock())
            _locks[key] = found
        return found


def _slot_lock(slot: Path) -> threading.Lock:
    """The same lock even for an entry inherited from a previous agent process."""
    with _lock:
        return _slot_locks.setdefault(slot.name, threading.Lock())


def _copy_or_reflink(src: Path, dst: Path) -> None:
    """Create an independent copy. FICLONE is copy-on-write, never a mutable hardlink."""
    try:
        import fcntl
        with src.open("rb") as source, dst.open("xb") as target:
            fcntl.ioctl(target.fileno(), 0x40049409, source.fileno())  # Linux FICLONE
            target.flush()
            os.fsync(target.fileno())
    except (ImportError, OSError):
        dst.unlink(missing_ok=True)
        with src.open("rb") as source, dst.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())


def _invalidate_locked(slot: Path, log=None) -> None:
    """Remove any COMPLETE value after a newer PUT landed but could not be adopted.

    Caller holds the object's lock, so no reader can slip between the durable write and this invalidation.
    Sentinel goes first: even if payload cleanup fails, `get` must download the remote truth.
    """
    sentinel, payload = slot / DONE, slot / "payload"
    try:
        sentinel.unlink(missing_ok=True)
        payload.unlink(missing_ok=True)
    except OSError as exc:
        if log:
            log(f"[input-cache] stale slot invalidation failed ({type(exc).__name__})")


def upload_and_adopt(url: str, src: Path, upload, log=None) -> None:
    """PUT an immutable snapshot, then make that *same snapshot* the local object-cache entry.

    Snapshot → PUT → adoption is serialised per object. Therefore workspace mutation after snapshot cannot
    change the bytes on the wire, and two overlapping writers cannot leave the cache describing an older
    PUT than the object store. A failed PUT never reaches adoption. If snapshot/cache maintenance itself is
    unavailable, transport falls back to its previous direct PUT and remains correct, merely uncached.
    """
    key = object_key(url)
    if key is None or not src.is_file():
        upload(src, url)
        return
    slot = _slot(key)
    payload, sentinel = slot / "payload", slot / DONE
    with _key_lock(key):
        if not enabled():
            upload(src, url)  # exception preserves the old COMPLETE value: remote did not commit B
            _invalidate_locked(slot, log=log)
            return
        snapshot = slot / f"payload.{os.getpid()}.{threading.get_ident()}.put-snapshot"
        try:
            try:
                slot.mkdir(parents=True, exist_ok=True)
                _copy_or_reflink(src, snapshot)
                snapshot.chmod(0o444)
                before = _sha256(snapshot)
            except Exception as exc:  # cache/snapshot availability is never transport availability
                if log:
                    log(f"[input-cache] snapshot unavailable ({type(exc).__name__}); PUT not cached")
                upload(src, url)  # exception preserves old cache; success makes it stale
                _invalidate_locked(slot, log=log)
                return
            upload(snapshot, url)
            try:
                after = _sha256(snapshot)
                if before != after:
                    # The uploader only has read access in production. If that invariant ever changes,
                    # refuse to claim an exact binding; the durable PUT itself has already succeeded.
                    _invalidate_locked(slot, log=log)
                    if log:
                        log(f"[input-cache] PUT snapshot changed while uploading {key}; not adopted")
                    return
                # No reader/pruner can observe the payload without the checksum that names its PUT body:
                # all three use this same object lock, and sentinel absence means cache miss.
                sentinel.unlink(missing_ok=True)
                snapshot.replace(payload)
                sentinel.write_text(before, encoding="ascii")
            except Exception as exc:  # cache maintenance cannot reverse a successful durable PUT
                _invalidate_locked(slot, log=log)
                if log:
                    log(f"[input-cache] could not adopt {key} ({type(exc).__name__}); PUT remains successful")
                return
            if log:
                try:
                    mb = payload.stat().st_size / 1e6
                except OSError:
                    mb = 0.0
                log(f"[input-cache] adopted {key} ({mb:.0f} MB) after PUT")
        finally:
            try:
                snapshot.unlink(missing_ok=True)
            except OSError:
                pass  # cleanup cannot mask the PUT's success or original transport exception
    try:
        prune(log=log)
    except (OSError, ValueError) as exc:
        if log:
            log(f"[input-cache] prune failed after PUT ({type(exc).__name__}); object remains durable")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


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
        slot_lock = _slot_lock(slot)
        if not slot_lock.acquire(blocking=False):
            continue
        try:
            # Re-check under the same per-object lock used by get/upload_and_adopt: an entry may have been
            # replaced since `_entries`, and an incomplete transfer is never ours to evict.
            payload = slot / "payload"
            if not (slot / DONE).exists() or not payload.is_file():
                continue
            actual = payload.stat().st_size
            shutil.rmtree(slot, ignore_errors=True)
            total -= actual
            freed += actual
        except OSError:
            continue
        finally:
            slot_lock.release()
        if log:
            log(f"[input-cache] evicted {slot.name} ({size / 1e6:.0f} MB)")
    return freed


def get(url: str, download, *, lease: Path, log=None) -> Path | None:
    """Copy/reflink an object into a caller-owned lease, fetching it at most once per pod.

    The persistent payload is never returned: once this function returns, pruning may delete its slot while
    ffmpeg safely reads the independent workspace lease. None means the URL is not cacheable.
    """
    if not enabled():
        return None
    key = object_key(url)
    if key is None:
        return None
    slot = _slot(key)
    payload, sentinel = slot / "payload", slot / DONE
    # per OBJECT, never globally: two chains pulling DIFFERENT inputs must still overlap
    with _key_lock(key):
        hit = sentinel.exists() and payload.exists()
        if not hit:
            slot.mkdir(parents=True, exist_ok=True)
            tmp = slot / f"payload.{os.getpid()}.{threading.get_ident()}.part"
            download(url, tmp)
            # sentinel AFTER the rename: a fetch killed mid-write must read as ABSENT, never as usable
            tmp.replace(payload)
            sentinel.write_text("", encoding="utf-8")
            if log:
                log(f"[input-cache] stored {key} ({payload.stat().st_size / 1e6:.0f} MB)")
        elif log:
            log(f"[input-cache] hit {key} ({payload.stat().st_size / 1e6:.0f} MB) — no transfer")
        lease.parent.mkdir(parents=True, exist_ok=True)
        tmp_lease = lease.with_name(f".{lease.name}.{os.getpid()}.{threading.get_ident()}.lease-part")
        try:
            _copy_or_reflink(payload, tmp_lease)
            tmp_lease.replace(lease)
        finally:
            tmp_lease.unlink(missing_ok=True)
    prune(log=log)      # outside the per-object lock: eviction must not hold up another object's transfer
    return lease
