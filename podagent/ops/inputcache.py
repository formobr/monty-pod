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
RETAINED = ".retained"

RETENTION_WHY = """
A PUT EXISTS SO A LATER READER CAN FIND THE BYTES. WHEN EVERY LATER READER IS THIS SAME BOX, IT BUYS NOTHING.

`upload_and_adopt` already proves the shape: after a durable PUT the same immutable snapshot becomes the local
object-cache entry, so the next chain that binds that address pays no transfer. Retention keeps the second
half and drops the first — the produced file is adopted under its canonical object key WITHOUT crossing the
wire. That is only legal when the object's whole audience is chains that will land on THIS worker, which is a
claim the caller makes per output and the runner refuses to guess.

THE MARKER IS THE DIFFERENCE BETWEEN A CACHE AND A STORE. An ordinary entry may be evicted at any moment: the
remote copy is the truth and a miss costs one download. A retained entry IS the truth — evicting it would
delete the only copy of a master, so a retained slot is never pruned and the cap is measured over the rest.
The marker also names the worker that holds it, which is what lets a refusal say WHERE the bytes are instead
of only that they are missing.
"""

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
        # The marker outliving its payload would make an empty slot permanently unevictable.
        (slot / RETAINED).unlink(missing_ok=True)
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
                # A durable PUT gives this object a remote truth, so any earlier retention claim on the
                # same key is over and the entry becomes evictable again.
                (slot / RETAINED).unlink(missing_ok=True)
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


class RetentionUnavailable(RuntimeError):
    """A retained output could not be made readable locally. LOUD by construction: nothing uploaded it, so a
    swallowed failure here is an object that exists nowhere and a later chain that 404s far from the cause."""


def adopt_local(url: str, src: Path, *, holder: str, log=None) -> Path:
    """Make `src` the local object-cache entry for `url`'s key WITHOUT uploading it (RETENTION_WHY).

    Same snapshot→adopt order as the PUT path, minus the PUT: the payload is an immutable copy, so later
    workspace mutation cannot change what a reader gets, and the sentinel lands only over a complete file.
    Returns the payload path. Raises RetentionUnavailable — a retained object has no remote fallback.
    """
    key = object_key(url)
    if key is None:
        raise RetentionUnavailable(
            f"retained output binds {url[:60]!r}, which names no cacheable object — retention keys on the "
            f"object a presigned URL addresses (host + path), so a non-http binding cannot be retained")
    if not src.is_file():
        raise RetentionUnavailable(f"retained output for {key} is not a file at {src}")
    if not enabled():
        raise RetentionUnavailable(
            f"retained output for {key} cannot be adopted: the input cache is OFF ({DISABLE_ENV}). Nothing "
            f"uploaded this object, so there is no remote copy to fall back to — re-run without retain")
    slot = _slot(key)
    payload, sentinel, marker = slot / "payload", slot / DONE, slot / RETAINED
    with _key_lock(key):
        snapshot = slot / f"payload.{os.getpid()}.{threading.get_ident()}.retain-snapshot"
        try:
            slot.mkdir(parents=True, exist_ok=True)
            _copy_or_reflink(src, snapshot)
            snapshot.chmod(0o444)
            digest = _sha256(snapshot)
            sentinel.unlink(missing_ok=True)
            snapshot.replace(payload)
            # Marker BEFORE the sentinel: a reader that sees a usable entry must already see that it is the
            # only copy, or the pruner could evict what nothing can re-download.
            marker.write_text(holder, encoding="utf-8")
            sentinel.write_text(digest, encoding="ascii")
        except Exception as exc:
            _invalidate_locked(slot, log=log)
            raise RetentionUnavailable(
                f"retained output for {key} could not be adopted ({type(exc).__name__}); nothing uploaded "
                f"it, so this object now exists nowhere") from exc
        finally:
            try:
                snapshot.unlink(missing_ok=True)
            except OSError:
                pass
    if log:
        try:
            mb = payload.stat().st_size / 1e6
        except OSError:
            mb = 0.0
        log(f"[input-cache] RETAINED {key} ({mb:.0f} MB) on {holder} — not uploaded")
    try:
        prune(log=log)
    except (OSError, ValueError) as exc:
        if log:
            log(f"[input-cache] prune failed after retain ({type(exc).__name__})")
    return payload


def retained_holder(url: str) -> str | None:
    """The worker named on this object's retention marker, or None if this box holds no retained copy."""
    key = object_key(url)
    if key is None or not enabled():
        return None
    slot = _slot(key)
    if not ((slot / DONE).exists() and (slot / "payload").is_file()):
        return None
    try:
        return (slot / RETAINED).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _entries() -> list[tuple[float, int, Path]]:
    """(mtime, bytes, slot) for every EVICTABLE entry — a half-written one is nobody's to evict, and neither
    is a RETAINED one: nothing uploaded it, so eviction is deletion, not a re-download (RETENTION_WHY)."""
    out: list[tuple[float, int, Path]] = []
    for slot in root().glob("*"):
        payload = slot / "payload"
        if not (slot / DONE).exists() or not payload.is_file():
            continue
        if (slot / RETAINED).exists():
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
            if (slot / RETAINED).exists():
                continue      # adopted between _entries and here: its only copy is not ours to delete
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
