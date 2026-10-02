"""TRK-86 — the input cache prunes off an in-memory account, never a walk per operation (ACCOUNT_WHY).

2026-10-02, v0.20.153: a 23,914-entry cache cost 1.75 s per `prune`, and `prune` ran after every fetch and
every PUT — 187 cache operations, ~327 s of pure directory scanning in one smoke envelope."""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from podagent.ops import inputcache

SEED = 20_000
BODY = b"0123456789"


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv(inputcache.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(inputcache.DISABLE_ENV, raising=False)
    inputcache._locks.clear()
    inputcache._slot_locks.clear()
    monkeypatch.setattr(inputcache, "_account", None)
    # `_stats` is a module-global counter, not per-account — it must not carry counts in from whatever
    # else in the suite called `prune` before this test's fixture ran.
    for key in inputcache._stats:
        inputcache._stats[key] = 0


def _fake_cache(n: int) -> list[Path]:
    """`n` complete entries straight on disk — the warm cache a restarted agent inherits — oldest first."""
    base = inputcache.root()
    base.mkdir(parents=True)
    body = inputcache._sentinel_body(len(BODY))
    now = time.time()
    slots = []
    for i in range(n):
        slot = base / f"{i:032x}"
        slot.mkdir()
        payload = slot / "payload"
        payload.write_bytes(BODY)
        os.utime(payload, (now - n + i, now - n + i))
        (slot / inputcache.DONE).write_text(body, encoding="utf-8")
        slots.append(slot)
    return slots


def _cap_gb(n_entries: int) -> str:
    return repr(n_entries * len(BODY) / 1e9)


def _walks(monkeypatch) -> list[int]:
    """Count every directory walk of the cache root, whoever makes it."""
    seen: list[int] = []
    original = Path.glob

    def _glob(self, pattern, *a, **kw):
        if self == inputcache.root():
            seen.append(1)
        return original(self, pattern, *a, **kw)

    monkeypatch.setattr(Path, "glob", _glob)
    return seen


def _dl(_url, dst):
    dst.write_bytes(BODY)


def test_prune_never_walks_the_cache_per_operation(tmp_path, monkeypatch):
    """1000 cache operations on a 20k-entry cache: ONE walk seeds the account, reconciles are the only other
    walks, and the cap still holds with the OLDEST entries evicted first."""
    seeded = _fake_cache(SEED)
    monkeypatch.setenv(inputcache.MAX_GB_ENV, _cap_gb(SEED))      # full to the byte: every insert must evict
    walks = _walks(monkeypatch)

    ops = 1000
    fetched = []
    for i in range(ops):
        url = f"https://r2.example/monty/new/{i % 600}.mov?sig={i}"   # 600 new objects, 400 hits
        inputcache.get(url, _dl, lease=tmp_path / "leases" / f"{i}")
        fetched.append(url)

    reconciles_allowed = ops // inputcache.RECONCILE_OPS
    assert len(walks) <= 1 + reconciles_allowed, \
        f"{len(walks)} full walks for {ops} operations — prune is rescanning the cache per operation"
    assert inputcache.stats()["prunes"] == ops

    # The cap is still the cap, on DISK — not merely in the account.
    on_disk = sum((s / "payload").stat().st_size for s in inputcache.root().iterdir()
                  if (s / inputcache.DONE).exists())
    assert on_disk <= SEED * len(BODY), "the account let the cache grow past its cap"
    assert inputcache._ledger().total == on_disk, "the account drifted from the disk"
    # Oldest-first: exactly the 600 oldest seeded entries made room for the 600 new objects.
    assert not any(s.exists() for s in seeded[:600]), "an old entry survived while newer ones were evicted"
    assert all(s.exists() for s in seeded[600:]), "a newer seeded entry was evicted before an older one"
    for url in fetched[:600]:
        assert (inputcache._slot(inputcache.object_key(url)) / "payload").exists(), "a fresh fetch was evicted"


def test_an_operation_under_the_cap_costs_no_walk_at_all(tmp_path, monkeypatch):
    _fake_cache(2000)
    inputcache.prune()                                            # the one seeding walk
    walks = _walks(monkeypatch)
    for i in range(50):
        inputcache.get(f"https://r2.example/x/{i}.mov", _dl, lease=tmp_path / f"l{i}")
    assert walks == [], "an under-cap prune walked the cache"


def test_an_external_delete_is_absorbed_by_the_reconcile(tmp_path, monkeypatch):
    """The account only sees what THIS process does. An operator `rm` leaves it pessimistic (it would evict
    for bytes that are already gone) until the reconcile re-reads the disk."""
    slots = _fake_cache(100)
    acct = inputcache._ledger()
    assert acct.total == 100 * len(BODY)
    for slot in slots[:40]:
        shutil.rmtree(slot)
    assert acct.total == 100 * len(BODY), "precondition: nothing but a walk can see an external delete"

    inputcache.reconcile()
    assert inputcache._ledger().total == 60 * len(BODY)
    assert set(inputcache._ledger().rows) == {s.name for s in slots[40:]}

    # And the next over-cap prune evicts the oldest SURVIVOR, not a ghost.
    assert inputcache.prune(keep_bytes=59 * len(BODY)) == len(BODY)
    assert not slots[40].exists() and slots[41].exists()


def test_the_reconcile_runs_by_itself_off_the_hot_path(tmp_path, monkeypatch):
    """Due after RECONCILE_OPS operations, it runs on a background thread — and a write made while it walks
    is not lost to the older picture the walk took."""
    slots = _fake_cache(10)
    monkeypatch.setattr(inputcache, "RECONCILE_OPS", 5)
    inputcache._ledger()
    shutil.rmtree(slots[0])                                         # external: only a reconcile sees it

    gate, walking = threading.Event(), threading.Event()
    original = inputcache._scan

    def _slow_scan(base):
        out = original(base)
        walking.set()
        assert gate.wait(5)
        return out

    monkeypatch.setattr(inputcache, "_scan", _slow_scan)
    for i in range(5):
        inputcache.prune()
    assert walking.wait(5), "a due reconcile never started"
    # Written while the walk is in flight: the walk's picture does not have it.
    late = inputcache.get("https://r2.example/late.mov", _dl, lease=tmp_path / "late")
    assert late.exists()
    gate.set()
    deadline = time.monotonic() + 5
    while inputcache._ledger().reconciling and time.monotonic() < deadline:
        time.sleep(0.01)
    acct = inputcache._ledger()
    assert not acct.reconciling
    assert slots[0].name not in acct.rows, "the reconcile did not absorb the external delete"
    late_slot = inputcache._slot(inputcache.object_key("https://r2.example/late.mov")).name
    assert late_slot in acct.rows, "an insert made during the reconcile was overwritten by its stale walk"
    assert acct.total == 10 * len(BODY)
