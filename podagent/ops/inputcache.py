"""One fetch per OBJECT for the pod's whole life, not one per chain. A presigned URL carries a signature and
an expiry, so the same object minted twice is two different strings, and the per-chain memo keyed on that
string — `source_axis` binds the master in three chains, so one stage moved the recording three times."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlsplit

CACHE_ENV = "OPS_INPUT_CACHE"
CACHE_DEFAULT = "/var/cache/monty/inputs"
DISABLE_ENV = "OPS_INPUT_CACHE_OFF"
MAX_GB_ENV = "OPS_INPUT_CACHE_MAX_GB"
MAX_GB_DEFAULT = 24.0
DONE = ".complete"
RETAINED = ".retained"
# Share of the ONE cap (MAX_GB_ENV) retention may hold. A module constant, not an env read: the box writes
# the pod's environment whole, so a second budget nobody assigns could only ever be its default — the exact
# shape the engine's own knob-registry test exists to refuse. Half leaves the ordinary cache half its working set.
RETAINED_SHARE = 0.5

RETENTION_WHY = """
A PUT EXISTS SO A LATER READER CAN FIND THE BYTES. WHEN EVERY LATER READER IS THIS SAME BOX, IT BUYS NOTHING.

`upload_and_adopt` already proves the shape: after a durable PUT the same immutable snapshot becomes the local
object-cache entry, so the next chain that binds that address pays no transfer. Retention keeps the second
half and drops the first — the produced file is adopted under its canonical object key WITHOUT crossing the
wire. That is only legal when the object's whole audience is chains that will land on THIS worker, which is a
claim the caller makes per output and the runner refuses to guess.

THE MARKER IS THE DIFFERENCE BETWEEN A CACHE AND A STORE. An ordinary entry may be evicted at any moment: the
remote copy is the truth and a miss costs one download. A retained entry IS the truth — evicting it would
delete the only copy of a master, so a retained slot is never a pruning CANDIDATE. The marker also names the
worker that holds it and the object key it was adopted under, which is what lets a refusal say WHERE the bytes
are, and what lets a newer run recognise the generation it supersedes.

"NEVER EVICTED" IS NOT "NOT COUNTED", AND THE FIRST DRAFT CONFLATED THEM. Retained slots were skipped by
`_entries`, so `prune` summed a total of ZERO over an empty candidate list and freed nothing: 25 runs of a
1 GB master under 25 distinct work prefixes left 25 GB on a box whose cap says 24, and the failure was ENOSPC
rather than an eviction. Three things close it, and none of them is a new knob (the box writes the pod's
environment whole, so a second budget nobody assigns would only ever be its default):

  · retained bytes are IN the one cap that already exists. `prune` counts them into the total and evicts
    ordinary entries against them, so retention squeezes the cache instead of the filesystem.
  · retention may claim at most `RETAINED_SHARE` of that cap, and an adoption that would cross the line is
    REFUSED, loudly, naming the tally and the ceiling. A step that fails is recoverable; a full disk takes
    the whole box down with every chain on it.
  · a retained object is scoped to the RUN that wrote it. The object key's directory is the work prefix and
    a re-run's prefix is new, so an adoption of the same basename under a different prefix RELEASES the older
    generation — nobody can read it (its readers name their own run's address) and nothing will re-upload it.
    A marker whose holder is not this process's identity is released for the same reason: the lease that
    could have claimed those bytes is gone.

THE POD STILL LEARNS NOTHING. "Same artifact, newer run" is read off the key's own shape — basename under a
different directory — never off a slug, a job id or anything else that would make this box know what it holds.
"""

ACCOUNT_WHY = """
THE PRUNER READS A LEDGER, NOT THE DISK. `prune` runs after every fetch and every PUT, and it used to walk the
whole cache twice to decide whether anything was over the cap. With 23,914 entries / 10.6 GB that walk measured
1.75 s, and one smoke envelope did 187 cache operations — 327 s of pure scanning (v0.20.153: fetch_broll
46 s -> 381 s while the bytes moved FELL). A long-lived pod's cache only grows, so the cost only grows.

So the account is kept in memory: ONE scan seeds it (lazily, on first use), every insert/evict on this process
re-reads that ONE slot from disk and updates it, and `prune` compares the running total to the cap — O(1) when
under it, oldest-first off an insertion-ordered index when over it. The index is the eviction order; the disk
is still the truth for the slot being evicted, re-checked under its object lock as before.

What the account cannot see is a change nobody on this process made (an operator `rm`, a disk swap). A
RECONCILE re-walks the disk every `RECONCILE_OPS` operations or `RECONCILE_S` seconds, whichever comes first,
on a background thread so no fetch waits on it; changes made while it walks win over what it saw.
"""
# 1.75 s per walk at 24k entries: once per 10 min or per 2000 ops is < 0.3 % of a busy pod's wall clock and
# bounds how long an external delete can keep the account pessimistic.
RECONCILE_S = 600.0
RECONCILE_OPS = 2000

_lock = threading.Lock()
_locks: dict[str, threading.Lock] = {}
_slot_locks: dict[str, threading.Lock] = {}
# Serialises the read-tally → check-ceiling → adopt sequence, so two concurrent retained outputs cannot both
# measure themselves against a budget the other is about to spend (RETENTION_WHY).
_retain_lock = threading.Lock()

# A sentinel too old to carry a size record: legacy adopt-path bodies were a bare sha256 hex digest.
_LEGACY_NO_CHECK = object()


def _sentinel_body(size: int, sha256: str | None = None) -> str:
    """The sentinel going forward: a SIZE RECORD for O(1) HIT validation, not an integrity check — that lives
    in cp.download's header enforcement, not here."""
    body = {"size": size} if sha256 is None else {"sha256": sha256, "size": size}
    return json.dumps(body, sort_keys=True)


def _sentinel_size(raw: str):
    """int to check against the payload, `_LEGACY_NO_CHECK` (a pre-size-record sha256, always valid), or None
    (empty/corrupt — heal it, this is what let a torn download stay a permanent poisoned HIT)."""
    raw = raw.strip()
    if not raw:
        return None
    if len(raw) == 64 and all(c in "0123456789abcdef" for c in raw):
        return _LEGACY_NO_CHECK
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    size = parsed.get("size") if isinstance(parsed, dict) else None
    return size if isinstance(size, int) else None


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


class _Account:
    """Running (bytes, retained) per COMPLETE slot of one cache root, oldest first (ACCOUNT_WHY)."""

    def __init__(self, base: Path):
        self.root = base
        self.rows: OrderedDict[str, tuple[int, bool]] = OrderedDict()
        self.total = 0
        self.held = 0
        self.ops = 0
        self.scanned_at = 0.0
        self.reconciling = False
        self.touched: set[str] = set()

    def _drop(self, name: str) -> None:
        row = self.rows.pop(name, None)
        if row is not None:
            self.total -= row[0]
            if row[1]:
                self.held -= row[0]

    def put(self, name: str, row: tuple[int, bool] | None) -> None:
        self._drop(name)
        if row is not None:
            self.rows[name] = row      # newest last: insertion order IS the eviction order
            self.total += row[0]
            if row[1]:
                self.held += row[0]
        if self.reconciling:
            self.touched.add(name)

    def load(self, scanned: list[tuple[float, str, int, bool]]) -> None:
        """Replace the account with a fresh walk, keeping whatever changed WHILE it walked."""
        live = {n: self.rows.get(n) for n in self.touched}
        self.rows.clear()
        self.total = self.held = 0
        for _mtime, name, size, retained in sorted(scanned):
            if name not in live:
                self.put(name, (size, retained))
        for name, row in live.items():
            if row is not None:
                self.put(name, row)
        self.touched.clear()
        self.reconciling = False
        self.ops = 0
        self.scanned_at = time.monotonic()


_account_lock = threading.Lock()
_account: _Account | None = None
_scan_lock = threading.Lock()          # one walk at a time: a second one would only repeat the first
# Visible cost (the span): walks taken, prune calls, their summed/max seconds, evictions.
_stats = {"scans": 0, "prunes": 0, "prune_s": 0.0, "prune_max_s": 0.0, "evicted": 0}


def stats() -> dict[str, float]:
    with _account_lock:
        return dict(_stats)


def _read_slot(slot: Path) -> tuple[int, bool] | None:
    """One slot's account row straight off the disk: None unless it is COMPLETE."""
    try:
        if not (slot / DONE).exists():
            return None
        st = (slot / "payload").stat()
    except OSError:
        return None
    return st.st_size, (slot / RETAINED).exists()


def _scan(base: Path) -> list[tuple[float, str, int, bool]]:
    """THE full walk — (mtime, slot name, bytes, retained) per complete slot. Seed and reconcile only."""
    out: list[tuple[float, str, int, bool]] = []
    for slot in base.glob("*"):
        payload = slot / "payload"
        if not (slot / DONE).exists():
            continue
        try:
            st = payload.stat()
        except OSError:
            continue
        if not payload.is_file():
            continue
        out.append((st.st_mtime, slot.name, st.st_size, (slot / RETAINED).exists()))
    with _account_lock:
        _stats["scans"] += 1
    return out


def _ledger() -> _Account:
    """The account for the CURRENT root, seeded by one walk the first time it is asked for."""
    global _account
    base = root()
    acct = _account
    if acct is not None and acct.root == base:
        return acct
    with _scan_lock:
        acct = _account
        if acct is not None and acct.root == base:
            return acct
        fresh = _Account(base)
        fresh.load(_scan(base))
        with _account_lock:
            _account = fresh
        return fresh


def _note(slot: Path) -> None:
    """Re-read ONE slot into the account after this process changed it — never a walk."""
    acct = _ledger()
    if slot.parent != acct.root:
        return
    row = _read_slot(slot)
    with _account_lock:
        acct.put(slot.name, row)


def reconcile() -> None:
    """Re-sync the account with the disk, absorbing changes this process did not make (ACCOUNT_WHY)."""
    acct = _ledger()
    with _scan_lock:
        with _account_lock:
            acct.reconciling = True
            acct.touched.clear()
        scanned = _scan(acct.root)
        with _account_lock:
            acct.load(scanned)


def _reconcile_if_due(acct: _Account) -> None:
    with _account_lock:
        acct.ops += 1
        due = acct.ops >= RECONCILE_OPS or time.monotonic() - acct.scanned_at >= RECONCILE_S
        if not due or acct.reconciling:
            return
        acct.reconciling = True      # single flight: claims the slot before the thread exists
        acct.touched.clear()

    def _run():
        try:
            with _scan_lock:
                scanned = _scan(acct.root)
                with _account_lock:
                    acct.load(scanned)
        except Exception:
            with _account_lock:
                acct.reconciling = False
                acct.ops = 0
                acct.scanned_at = time.monotonic()

    threading.Thread(target=_run, name="inputcache-reconcile", daemon=True).start()


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
    _note(slot)


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
                sentinel.write_text(_sentinel_body(payload.stat().st_size, before), encoding="utf-8")
                _note(slot)
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


class LeaseIntegrityError(RuntimeError):
    """A workspace lease's byte size does not match the cache slot it was copied from — a refusal, not a
    heal, because the slot payload's own size is what this check trusts."""


def _generation(key: str) -> tuple[str, str]:
    """(work prefix, basename) for an object key. The prefix IS the run — a re-run mints a new one — and the
    basename is the artifact within it, so "same artifact, newer run" is readable off the key's own shape."""
    scope, _, name = key.rpartition("/")
    return scope, name


def _read_marker(slot: Path) -> dict[str, str] | None:
    try:
        raw = json.loads((slot / RETAINED).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {"holder": str(raw.get("holder", "")), "key": str(raw.get("key", ""))} if isinstance(raw, dict) \
        else None


def _retained_rows() -> list[tuple[int, Path, dict[str, str]]]:
    """(bytes, slot, marker) for every COMPLETE retained entry. Never a pruning candidate; always counted."""
    rows: list[tuple[int, Path, dict[str, str]]] = []
    acct = _ledger()
    with _account_lock:
        names = [name for name, (_size, retained) in acct.rows.items() if retained]
    for name in names:                   # the account's retained slots, never a walk of the whole cache
        slot = acct.root / name
        payload = slot / "payload"
        if not (slot / DONE).exists() or not payload.is_file() or not (slot / RETAINED).exists():
            continue
        marker = _read_marker(slot)
        if marker is None:
            continue
        try:
            rows.append((payload.stat().st_size, slot, marker))
        except OSError:
            continue
    return rows


def retained_bytes() -> int:
    """THE TALLY, and it is the number a refusal quotes. Not derived from anything a caller passes in: a
    budget you cannot read is a budget you cannot be refused against."""
    if not enabled():
        return 0
    return sum(size for size, _slot, _m in _retained_rows())


def retained_cap(keep_bytes: int | None = None) -> int:
    """The hard ceiling retention may hold — a share of the ONE cap, never a second budget."""
    total = keep_bytes if keep_bytes is not None else int(
        float(os.environ.get(MAX_GB_ENV) or MAX_GB_DEFAULT) * 1e9)
    return int(max(0, total) * RETAINED_SHARE)


def _release_slot(slot: Path, why: str, log=None) -> int:
    """Drop a retained entry that nothing can read any more. Marker FIRST: if the rmtree half fails, what is
    left is an ordinary evictable entry, never a slot that is both unreadable and unprunable."""
    import shutil
    lock = _slot_lock(slot)
    if not lock.acquire(blocking=False):
        return 0                      # in use by an adoption/read right now; the next sweep gets it
    try:
        try:
            size = (slot / "payload").stat().st_size
        except OSError:
            size = 0
        (slot / RETAINED).unlink(missing_ok=True)
        shutil.rmtree(slot, ignore_errors=True)
    except OSError:
        return 0
    finally:
        _note(slot)
        lock.release()
    if log:
        log(f"[input-cache] released retained {slot.name} ({size / 1e6:.0f} MB) — {why}")
    return size


def release_superseded(key: str, holder: str, log=None) -> int:
    """Free every retained entry the arrival of `key` under `holder` makes unreadable, and say why.

    TWO WAYS AN ENTRY DIES, both of which mean nobody can ever claim it again:
      · a NEWER RUN of the same artifact — same basename, different work prefix. Its readers bind their own
        run's address, so the old generation is bytes with no address pointing at them.
      · a CHANGED LEASE — the marker names a worker this process is not. A reader declaring `retained_on`
        for that worker would be refused here anyway, so the bytes are already unreachable.
    """
    if not enabled():
        return 0
    scope, name = _generation(key)
    freed = 0
    for _size, slot, marker in _retained_rows():
        if marker["key"] == key:
            continue                  # this very object; adoption replaces it in place under its own lock
        if marker["holder"] != holder:
            freed += _release_slot(slot, f"held by {marker['holder'] or 'nobody'}, this worker is {holder}",
                                   log=log)
            continue
        old_scope, old_name = _generation(marker["key"])
        if old_name == name and old_scope != scope:
            freed += _release_slot(slot, f"superseded by a newer run of {name} under {scope}", log=log)
    return freed


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
    try:
        incoming = src.stat().st_size
    except OSError as exc:
        raise RetentionUnavailable(f"retained output for {key} could not be weighed "
                                   f"({type(exc).__name__})") from exc
    # ONE adoption at a time, so the tally a refusal quotes is the tally the next adoption is measured
    # against. Nothing takes this while holding an object lock, so the order below cannot deadlock.
    with _retain_lock:
        release_superseded(key, holder, log=log)
        held = retained_bytes()
        # This object's own previous generation (same key) is replaced in place, so it is not new weight.
        already = next((size for size, _s, m in _retained_rows() if m["key"] == key), 0)
        ceiling = retained_cap()
        if held - already + incoming > ceiling:
            raise RetentionUnavailable(
                f"retained output for {key} ({incoming / 1e9:.2f} GB) would put this worker's retained set at "
                f"{(held - already + incoming) / 1e9:.2f} GB against a ceiling of {ceiling / 1e9:.2f} GB "
                f"({RETAINED_SHARE:.0%} of {MAX_GB_ENV}). REFUSING the adoption instead of filling the disk: "
                f"a failed step is recoverable, a full box takes every chain on it down. Re-run this chain "
                f"without retain, or let the older runs' masters go.")
        with _key_lock(key):
            snapshot = slot / f"payload.{os.getpid()}.{threading.get_ident()}.retain-snapshot"
            try:
                slot.mkdir(parents=True, exist_ok=True)
                _copy_or_reflink(src, snapshot)
                snapshot.chmod(0o444)
                digest = _sha256(snapshot)
                sentinel.unlink(missing_ok=True)
                snapshot.replace(payload)
                # Marker BEFORE the sentinel: a reader that sees a usable entry must already see that it is
                # the only copy, or the pruner could evict what nothing can re-download. It carries the KEY
                # as well as the holder, because a slot name is a digest and the generation this supersedes
                # can only be read off the key's own shape.
                marker.write_text(json.dumps({"holder": holder, "key": key}, sort_keys=True),
                                  encoding="utf-8")
                sentinel.write_text(_sentinel_body(payload.stat().st_size, digest), encoding="utf-8")
                _note(slot)
            except Exception as exc:
                _invalidate_locked(slot, log=log)
                raise RetentionUnavailable(
                    f"retained output for {key} could not be adopted ({type(exc).__name__}); nothing "
                    f"uploaded it, so this object now exists nowhere") from exc
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
    marker = _read_marker(slot)
    return (marker["holder"] or None) if marker else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def prune(keep_bytes: int | None = None, log=None) -> int:
    """Evict oldest-first until the cache fits. A pod's disk is smaller than the media it sees in a shift,
    so an unbounded cache trades one bug for a fuller one.

    RETAINED BYTES ARE COUNTED BUT NEVER EVICTED (RETENTION_WHY). Skipping them from the total too was the
    bug: an all-retained cache summed to ZERO, compared clean against the cap and freed nothing while the
    disk filled. Counting them means retention squeezes the ordinary cache, which is the pressure that
    SHOULD arrive first — a re-download is what a cache is for.

    The totals come from the in-memory account, never a walk (ACCOUNT_WHY): under the cap this is O(1).
    """
    t0 = time.perf_counter()
    cap = keep_bytes if keep_bytes is not None else int(
        float(os.environ.get(MAX_GB_ENV) or MAX_GB_DEFAULT) * 1e9)
    acct = _ledger()
    try:
        return _evict(acct, cap, log)
    finally:
        _reconcile_if_due(acct)
        took = time.perf_counter() - t0
        with _account_lock:
            _stats["prunes"] += 1
            _stats["prune_s"] += took
            _stats["prune_max_s"] = max(_stats["prune_max_s"], took)
        if log and took >= 0.25:
            log(f"[input-cache] prune took {took:.2f} s ({len(acct.rows)} entries, "
                f"{acct.total / 1e9:.2f} GB accounted)")


def _oldest_batch(acct: _Account, cap: int, tried: set[str]) -> list[tuple[str, int]]:
    """Just enough of the oldest evictable rows to cover the excess — never a copy of the whole index.
    Caller holds `_account_lock`."""
    excess = acct.total - cap
    batch: list[tuple[str, int]] = []
    for name, (size, retained) in acct.rows.items():
        if excess <= 0:
            break
        if retained or name in tried:
            continue
        batch.append((name, size))
        excess -= size
    return batch


def _evict(acct: _Account, cap: int, log) -> int:
    with _account_lock:
        if acct.total <= cap:
            return 0
        held = acct.held
    if log and held:
        log(f"[input-cache] {held / 1e9:.2f} GB retained (unevictable) of a {cap / 1e9:.2f} GB cap — "
            f"evicting cached objects against it")
    import shutil
    freed = 0
    tried: set[str] = set()
    while True:
        # Oldest first: the account's insertion order, taken in batches so no lock is held across disk work.
        with _account_lock:
            batch = _oldest_batch(acct, cap, tried)
        if not batch:
            return freed
        for name, size in batch:
            tried.add(name)
            slot = acct.root / name
            slot_lock = _slot_lock(slot)
            if not slot_lock.acquire(blocking=False):
                continue
            try:
                # Re-check under the same per-object lock used by get/upload_and_adopt: the account may be
                # stale (an external delete, a replace since the batch), and an incomplete transfer is never
                # ours to evict.
                row = _read_slot(slot)
                if row is None or row[1]:
                    # gone, half-written, or adopted as retained meanwhile: correct the account, delete nothing
                    with _account_lock:
                        acct.put(name, row)
                    continue
                shutil.rmtree(slot, ignore_errors=True)
                with _account_lock:
                    acct.put(name, _read_slot(slot))
                    _stats["evicted"] += 1
                freed += row[0]
            except OSError:
                continue
            finally:
                slot_lock.release()
            if log:
                log(f"[input-cache] evicted {name} ({size / 1e6:.0f} MB)")


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
        if (slot / RETAINED).exists():
            # Retention gates HIT/MISS before the sentinel is even read — no remote fallback (RETENTION_WHY).
            if not payload.is_file():
                raise RetentionUnavailable(
                    f"retained object for {key} has no payload on disk — nothing to serve and no remote "
                    f"copy to fall back to")
            hit = True
        else:
            hit = sentinel.exists() and payload.exists()
            if hit:
                try:
                    raw = sentinel.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    raw = ""  # unreadable is exactly as trustworthy as empty
                recorded = _sentinel_size(raw)
                healthy = recorded is _LEGACY_NO_CHECK
                if not healthy and recorded is not None:
                    try:
                        healthy = payload.stat().st_size == recorded
                    except OSError:
                        healthy = False
                if not healthy:
                    if log:
                        try:
                            actual = payload.stat().st_size
                        except OSError:
                            actual = -1
                        log(f"[input-cache] healing {key}: sentinel recorded {recorded!r}, payload is "
                            f"{actual} bytes — re-downloading")
                    shutil.rmtree(slot, ignore_errors=True)
                    _note(slot)
                    hit = False
        if not hit:
            slot.mkdir(parents=True, exist_ok=True)
            tmp = slot / f"payload.{os.getpid()}.{threading.get_ident()}.part"
            ok = False
            try:
                download(url, tmp)
                size = tmp.stat().st_size
                # sentinel AFTER the rename: a fetch killed mid-write must read as ABSENT, never as usable
                tmp.replace(payload)
                ok = True
            finally:
                if not ok:
                    tmp.unlink(missing_ok=True)
            sentinel.write_text(_sentinel_body(size), encoding="utf-8")
            _note(slot)
            if log:
                log(f"[input-cache] stored {key} ({payload.stat().st_size / 1e6:.0f} MB)")
        elif log:
            log(f"[input-cache] hit {key} ({payload.stat().st_size / 1e6:.0f} MB) — no transfer")
        # Read under the same per-object lock the copy runs under, so nothing between this stat and the
        # lease copy below can move the slot the comparison trusts.
        slot_size = payload.stat().st_size
        lease.parent.mkdir(parents=True, exist_ok=True)
        tmp_lease = lease.with_name(f".{lease.name}.{os.getpid()}.{threading.get_ident()}.lease-part")
        try:
            _copy_or_reflink(payload, tmp_lease)
            tmp_lease.replace(lease)
        finally:
            tmp_lease.unlink(missing_ok=True)
        lease_size = lease.stat().st_size
        if lease_size != slot_size:
            lease.unlink(missing_ok=True)
            raise LeaseIntegrityError(
                f"input-cache lease for {key} (slot={slot.name}) is {lease_size} bytes but the cache slot's "
                f"payload is {slot_size} bytes — refusing a short/partial lease; the slot payload is untouched")
    prune(log=log)      # outside the per-object lock: eviction must not hold up another object's transfer
    return lease
