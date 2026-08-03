"""A presign signs a capability, not an identity: the same object minted twice is two different strings, and
the per-chain memo keyed on the string. `source_axis` binds the master in three chains — three transfers."""
from __future__ import annotations

import threading

import pytest

from podagent.ops import inputcache, runner

URL_A = "https://r2.example/monty/uploads/local/fleet-s/s.mov?X-Amz-Signature=aaa&X-Amz-Expires=86400"
URL_A2 = "https://r2.example/monty/uploads/local/fleet-s/s.mov?X-Amz-Signature=bbb&X-Amz-Expires=900"
URL_B = "https://r2.example/monty/uploads/local/fleet-s/other.mov?X-Amz-Signature=ccc"


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(inputcache.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(inputcache.DISABLE_ENV, raising=False)
    inputcache._locks.clear()


def _dl(counter: list[str]):
    def _download(url, dest):
        counter.append(url)
        dest.write_bytes(b"payload")
    return _download


def test_a_re_minted_presign_is_the_same_object():
    """NEGATIVE — this is the whole bug. Key on the full URL and this pulls twice."""
    seen: list[str] = []
    p1 = inputcache.get(URL_A, _dl(seen))
    p2 = inputcache.get(URL_A2, _dl(seen))
    assert p1 == p2 and len(seen) == 1, f"a re-signed URL re-downloaded the same object: {seen}"


def test_two_different_objects_do_not_collide():
    seen: list[str] = []
    a = inputcache.get(URL_A, _dl(seen))
    b = inputcache.get(URL_B, _dl(seen))
    assert a != b and len(seen) == 2


def test_the_cache_outlives_the_chain(tmp_path):
    """Two chains = two Workspaces. The master must not cross the wire again for the second one."""
    seen: list[str] = []
    w1, w2 = runner.Workspace(tmp_path / "c1"), runner.Workspace(tmp_path / "c2")
    for w in (w1, w2):
        (w.root / "_in").mkdir(parents=True, exist_ok=True)
    import podagent.ops.runner as R
    orig = R.download
    R.download = _dl(seen)
    try:
        a = w1.fetch(URL_A)
        b = w2.fetch(URL_A2)
    finally:
        R.download = orig
    assert a == b, "the second chain got its own copy"
    assert len(seen) == 1, f"the recording crossed the wire {len(seen)}x for two chains"


def test_a_killed_fetch_reads_as_absent(tmp_path):
    """The sentinel is written AFTER the rename, so a partial payload is never served as whole."""
    def _boom(url, dest):
        dest.write_bytes(b"half")
        raise OSError("connection reset")

    with pytest.raises(OSError):
        inputcache.get(URL_A, _boom)
    seen: list[str] = []
    p = inputcache.get(URL_A, _dl(seen))
    assert p.read_bytes() == b"payload" and len(seen) == 1, "a truncated fetch was served as a hit"


def test_a_non_http_binding_is_not_cached():
    """A local path or a file: URL has no object identity here — the caller keeps doing what it did."""
    assert inputcache.get("/local/path.mov", _dl([])) is None
    assert inputcache.get("file:///x.mov", _dl([])) is None


def test_the_cache_can_be_switched_off(monkeypatch):
    monkeypatch.setenv(inputcache.DISABLE_ENV, "1")
    assert inputcache.get(URL_A, _dl([])) is None


def test_the_query_is_never_part_of_the_identity():
    assert inputcache.object_key(URL_A) == inputcache.object_key(URL_A2)
    assert inputcache.object_key(URL_A) == "r2.example/monty/uploads/local/fleet-s/s.mov"
    assert inputcache.object_key(URL_B) != inputcache.object_key(URL_A)


def test_two_threads_racing_one_object_transfer_it_once():
    """Serialised PER OBJECT: the point is one transfer, not a global lock that would serialise the chain."""
    import time
    seen: list[str] = []
    lock = threading.Lock()

    def _slow(url, dest):
        time.sleep(0.3)                      # wide enough that a second entrant would overlap
        with lock:
            seen.append(url)
        dest.write_bytes(b"payload")

    out: list = []
    ts = [threading.Thread(target=lambda: out.append(inputcache.get(URL_A, _slow))) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    assert len(seen) == 1, f"both threads transferred the same object: {seen}"
    assert out[0] == out[1]


def test_different_objects_are_not_serialised_against_each_other():
    """NEGATIVE on the locking shape: one global lock would make a fan-out of distinct inputs sequential."""
    import time
    inside = threading.Event()
    release = threading.Event()

    def _a(url, dest):
        inside.set()
        assert release.wait(timeout=5), "a fetch of a DIFFERENT object was blocked behind this one"
        dest.write_bytes(b"payload")

    def _b(url, dest):
        dest.write_bytes(b"payload")

    t = threading.Thread(target=lambda: inputcache.get(URL_A, _a))
    t.start()
    assert inside.wait(timeout=5)
    inputcache.get(URL_B, _b)                # must NOT wait on URL_A's transfer
    release.set()
    t.join(timeout=10)


def test_the_cache_is_bounded(tmp_path):
    """A pod's disk is smaller than the media it sees in a shift — unbounded, this trades one bug for a
    fuller one. Oldest goes first, and the entry just stored must survive its own prune."""
    import os
    import time
    for i in range(3):
        p = inputcache.get(f"https://r2.example/b/{i}.mov?sig={i}", _dl([]))
        os.utime(p, (time.time() - (10 - i), time.time() - (10 - i)))
    freed = inputcache.prune(keep_bytes=len(b"payload") * 2)
    assert freed > 0, "nothing was evicted although the cache was over its cap"
    left = {s.name for s in inputcache.root().glob("*") if (s / inputcache.DONE).exists()}
    assert len(left) == 2, f"eviction did not stop at the cap: {left}"
    newest = inputcache._slot(inputcache.object_key("https://r2.example/b/2.mov?sig=2"))
    assert newest.name in left, "the newest entry was evicted before the oldest"


def test_a_half_written_entry_is_never_evicted_as_if_complete(tmp_path):
    """Its slot has no sentinel, so it is not the pruner's to count OR to delete mid-transfer."""
    slot = inputcache._slot("r2.example/x.mov")
    slot.mkdir(parents=True)
    (slot / "payload.123.part").write_bytes(b"half")
    inputcache.prune(keep_bytes=0)
    assert (slot / "payload.123.part").exists(), "the pruner deleted a transfer that was still running"
