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
    inputcache._slot_locks.clear()


def _dl(counter: list[str]):
    def _download(url, dest):
        counter.append(url)
        dest.write_bytes(b"payload")
    return _download


def test_a_re_minted_presign_is_the_same_object(tmp_path):
    """NEGATIVE — this is the whole bug. Key on the full URL and this pulls twice."""
    seen: list[str] = []
    p1 = inputcache.get(URL_A, _dl(seen), lease=tmp_path / "l1")
    p2 = inputcache.get(URL_A2, _dl(seen), lease=tmp_path / "l2")
    assert p1 != p2 and p1.read_bytes() == p2.read_bytes() == b"payload"
    assert len(seen) == 1, f"a re-signed URL re-downloaded the same object: {seen}"


def test_two_different_objects_do_not_collide(tmp_path):
    seen: list[str] = []
    a = inputcache.get(URL_A, _dl(seen), lease=tmp_path / "a")
    b = inputcache.get(URL_B, _dl(seen), lease=tmp_path / "b")
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
    assert a != b and a.read_bytes() == b.read_bytes(), "workspaces did not get independent leases"
    assert len(seen) == 1, f"the recording crossed the wire {len(seen)}x for two chains"


def test_a_killed_fetch_reads_as_absent(tmp_path):
    """The sentinel is written AFTER the rename, so a partial payload is never served as whole."""
    def _boom(url, dest):
        dest.write_bytes(b"half")
        raise OSError("connection reset")

    with pytest.raises(OSError):
        inputcache.get(URL_A, _boom, lease=tmp_path / "bad")
    seen: list[str] = []
    p = inputcache.get(URL_A, _dl(seen), lease=tmp_path / "good")
    assert p.read_bytes() == b"payload" and len(seen) == 1, "a truncated fetch was served as a hit"


def test_a_non_http_binding_is_not_cached(tmp_path):
    """A local path or a file: URL has no object identity here — the caller keeps doing what it did."""
    assert inputcache.get("/local/path.mov", _dl([]), lease=tmp_path / "a") is None
    assert inputcache.get("file:///x.mov", _dl([]), lease=tmp_path / "b") is None


def test_the_cache_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv(inputcache.DISABLE_ENV, "1")
    assert inputcache.get(URL_A, _dl([]), lease=tmp_path / "off") is None


def test_the_query_is_never_part_of_the_identity():
    assert inputcache.object_key(URL_A) == inputcache.object_key(URL_A2)
    assert inputcache.object_key(URL_A) == "r2.example/monty/uploads/local/fleet-s/s.mov"
    assert inputcache.object_key(URL_B) != inputcache.object_key(URL_A)


def test_two_threads_racing_one_object_transfer_it_once(tmp_path):
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
    ts = [threading.Thread(target=lambda i=i: out.append(
        inputcache.get(URL_A, _slow, lease=tmp_path / f"lease-{i}"))) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    assert len(seen) == 1, f"both threads transferred the same object: {seen}"
    assert out[0] != out[1] and out[0].read_bytes() == out[1].read_bytes()


def test_different_objects_are_not_serialised_against_each_other(tmp_path):
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

    t = threading.Thread(target=lambda: inputcache.get(URL_A, _a, lease=tmp_path / "a"))
    t.start()
    assert inside.wait(timeout=5)
    inputcache.get(URL_B, _b, lease=tmp_path / "b")  # must NOT wait on URL_A's transfer
    release.set()
    t.join(timeout=10)


def test_the_cache_is_bounded(tmp_path):
    """A pod's disk is smaller than the media it sees in a shift — unbounded, this trades one bug for a
    fuller one. Oldest goes first, and the entry just stored must survive its own prune."""
    import os
    import time
    for i in range(3):
        url = f"https://r2.example/b/{i}.mov?sig={i}"
        inputcache.get(url, _dl([]), lease=tmp_path / f"lease-{i}")
        payload = inputcache._slot(inputcache.object_key(url)) / "payload"
        os.utime(payload, (time.time() - (10 - i), time.time() - (10 - i)))
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


def test_put_snapshot_is_exact_even_if_workspace_mutates_same_size(tmp_path):
    src = tmp_path / "workspace.mp4"
    src.write_bytes(b"before")
    put: list[bytes] = []

    def _upload(snapshot, _url):
        src.write_bytes(b"after!")  # same size, after the immutable snapshot was taken
        put.append(snapshot.read_bytes())

    inputcache.upload_and_adopt(URL_A, src, _upload)
    assert put == [b"before"], "PUT read the mutable workspace rather than its snapshot"
    seen: list[str] = []
    lease = inputcache.get(URL_A2, _dl(seen), lease=tmp_path / "lease")
    assert lease.read_bytes() == put[0] == b"before" and not seen
    assert lease.stat().st_ino != src.stat().st_ino, "workspace was hardlinked into the lease"


def test_a_killed_snapshot_falls_back_to_put_without_adoption(tmp_path, monkeypatch):
    src = tmp_path / "workspace.mp4"
    src.write_bytes(b"whole")

    def _partial(_src, dst):
        dst.write_bytes(b"ha")
        raise OSError("disk full")

    monkeypatch.setattr(inputcache, "_copy_or_reflink", _partial)
    sent: list[bytes] = []
    inputcache.upload_and_adopt(URL_A, src, lambda path, _url: sent.append(path.read_bytes()))
    slot = inputcache._slot(inputcache.object_key(URL_A))
    assert sent == [b"whole"]
    assert not (slot / inputcache.DONE).exists(), "partial adopted bytes were advertised as complete"


def test_a_failed_put_never_reaches_adoption(tmp_path, monkeypatch):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"src")
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.pack, "resolve", lambda _h: (
        lambda *, params, inputs, outputs: outputs["dst"].write_bytes(b"out")))
    monkeypatch.setattr(runner, "upload", lambda *_a: (_ for _ in ()).throw(OSError("PUT failed")))
    from podagent.models import OpChain, OpsPackRef
    chain = OpChain(job_id="j", pack=OpsPackRef(url="https://x/p.tgz", sha256="a" * 64), steps=[{
        "id": "s", "op": "media.scale", "needs": [],
        "params": {"height": 960, "encode_profile": "browser"},
        "inputs": [{"port": "src", "path": str(src)}],
        "outputs": [{"port": "dst", "url": URL_A}],
    }])
    with pytest.raises(OSError, match="PUT failed"):
        runner._run_step(chain.steps[0], runner.Workspace(tmp_path / "ws"), {})
    slot = inputcache._slot(inputcache.object_key(URL_A))
    assert not (slot / inputcache.DONE).exists(), "a failed PUT poisoned the input cache"


def test_snapshot_cleanup_cannot_mask_the_put_failure(tmp_path, monkeypatch):
    src = tmp_path / "out.mp4"
    src.write_bytes(b"out")
    original_unlink = inputcache.Path.unlink

    def _unlink(path, *args, **kwargs):
        if path.name.endswith("put-snapshot"):
            raise OSError("cleanup failed too")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(inputcache.Path, "unlink", _unlink)
    with pytest.raises(RuntimeError, match="authoritative PUT failure"):
        inputcache.upload_and_adopt(
            URL_A, src,
            lambda _path, _url: (_ for _ in ()).throw(RuntimeError("authoritative PUT failure")))


def _seed_remote_a(tmp_path):
    remote = {"body": b"A"}
    inputcache.get(
        URL_A, lambda _url, dest: dest.write_bytes(remote["body"]), lease=tmp_path / "seed-a")
    return remote


def test_successful_fallback_put_invalidates_prior_complete_a(tmp_path, monkeypatch):
    remote = _seed_remote_a(tmp_path)
    src = tmp_path / "b.mp4"
    src.write_bytes(b"B")
    original_copy = inputcache._copy_or_reflink

    def _snapshot_fails(source, dest):
        if dest.name.endswith("put-snapshot"):
            raise OSError("snapshot disk unavailable")
        return original_copy(source, dest)

    monkeypatch.setattr(inputcache, "_copy_or_reflink", _snapshot_fails)
    inputcache.upload_and_adopt(URL_A2, src,
                                lambda path, _url: remote.__setitem__("body", path.read_bytes()))
    slot = inputcache._slot(inputcache.object_key(URL_A))
    assert not (slot / inputcache.DONE).exists() and not (slot / "payload").exists()
    downloads: list[bytes] = []

    def _download(_url, dest):
        downloads.append(remote["body"])
        dest.write_bytes(remote["body"])

    lease = inputcache.get(URL_A, _download, lease=tmp_path / "fresh-b")
    assert downloads == [b"B"] and lease.read_bytes() == b"B", "stale COMPLETE A survived remote=B"


@pytest.mark.parametrize("after_failure", ["exception", "mismatch"])
def test_post_put_checksum_failure_invalidates_prior_complete_a(after_failure, tmp_path, monkeypatch):
    remote = _seed_remote_a(tmp_path)
    src = tmp_path / "b.mp4"
    src.write_bytes(b"B")
    real_sha = inputcache._sha256
    calls = 0

    def _sha(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_sha(path)
        if after_failure == "exception":
            raise OSError("checksum read failed after PUT")
        return "0" * 64

    monkeypatch.setattr(inputcache, "_sha256", _sha)
    inputcache.upload_and_adopt(URL_A2, src,
                                lambda path, _url: remote.__setitem__("body", path.read_bytes()))
    slot = inputcache._slot(inputcache.object_key(URL_A))
    assert not (slot / inputcache.DONE).exists() and not (slot / "payload").exists()
    # Restore hashing for the subsequent ordinary GET; it must cross the wire for B, never hit A.
    monkeypatch.setattr(inputcache, "_sha256", real_sha)
    downloads: list[bytes] = []

    def _download(_url, dest):
        downloads.append(remote["body"])
        dest.write_bytes(remote["body"])

    lease = inputcache.get(URL_A, _download, lease=tmp_path / f"fresh-{after_failure}")
    assert downloads == [b"B"] and lease.read_bytes() == b"B"


def test_failed_put_preserves_prior_complete_a(tmp_path):
    _seed_remote_a(tmp_path)
    src = tmp_path / "b.mp4"
    src.write_bytes(b"B")
    with pytest.raises(RuntimeError, match="PUT rejected"):
        inputcache.upload_and_adopt(
            URL_A2, src, lambda _path, _url: (_ for _ in ()).throw(RuntimeError("PUT rejected")))
    downloads: list[str] = []
    lease = inputcache.get(URL_A, _dl(downloads), lease=tmp_path / "still-a")
    assert not downloads and lease.read_bytes() == b"A", "failed PUT discarded still-authoritative A"


def test_post_put_payload_replace_failure_invalidates_prior_complete_a(tmp_path, monkeypatch):
    remote = _seed_remote_a(tmp_path)
    src = tmp_path / "b.mp4"
    src.write_bytes(b"B")
    original_replace = inputcache.Path.replace

    def _replace(path, target):
        if path.name.endswith("put-snapshot"):
            raise OSError("payload replace failed")
        return original_replace(path, target)

    monkeypatch.setattr(inputcache.Path, "replace", _replace)
    inputcache.upload_and_adopt(URL_A2, src,
                                lambda path, _url: remote.__setitem__("body", path.read_bytes()))
    downloads: list[bytes] = []

    def _download(_url, dest):
        downloads.append(remote["body"])
        dest.write_bytes(remote["body"])

    lease = inputcache.get(URL_A, _download, lease=tmp_path / "replace-fresh-b")
    assert downloads == [b"B"] and lease.read_bytes() == b"B"


def test_adoption_failure_never_changes_a_successful_put_into_a_failed_step(tmp_path, monkeypatch):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"src")
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.pack, "resolve", lambda _h: (
        lambda *, params, inputs, outputs: outputs["dst"].write_bytes(b"out")))
    sent: list[str] = []
    monkeypatch.setattr(runner, "upload", lambda _path, url: sent.append(url))
    original_replace = inputcache.Path.replace

    def _replace(path, target):
        if path.name.endswith("put-snapshot"):
            raise OSError("cache disk failed")
        return original_replace(path, target)

    monkeypatch.setattr(inputcache.Path, "replace", _replace)
    from podagent.models import OpChain, OpsPackRef
    chain = OpChain(job_id="j", pack=OpsPackRef(url="https://x/p.tgz", sha256="a" * 64), steps=[{
        "id": "s", "op": "media.scale", "needs": [],
        "params": {"height": 960, "encode_profile": "browser"},
        "inputs": [{"port": "src", "path": str(src)}],
        "outputs": [{"port": "dst", "url": URL_A}],
    }])
    runner._run_step(chain.steps[0], runner.Workspace(tmp_path / "ws"), {})
    assert sent == [URL_A]


def test_prune_cannot_delete_payload_during_copy_out_lease(tmp_path, monkeypatch):
    inputcache.get(URL_A, _dl([]), lease=tmp_path / "seed")
    slot = inputcache._slot(inputcache.object_key(URL_A))
    # Simulate a warm disk inherited by a restarted agent: no in-memory lock table exists yet.
    inputcache._locks.clear()
    inputcache._slot_locks.clear()
    entered = threading.Event()
    release = threading.Event()
    original = inputcache._copy_or_reflink

    def _blocked(src, dst):
        if src == slot / "payload":
            entered.set()
            assert release.wait(5)
        original(src, dst)

    monkeypatch.setattr(inputcache, "_copy_or_reflink", _blocked)
    out: list = []
    thread = threading.Thread(target=lambda: out.append(
        inputcache.get(URL_A2, _dl([]), lease=tmp_path / "live")))
    thread.start()
    assert entered.wait(5)
    assert inputcache.prune(keep_bytes=0) == 0
    assert (slot / "payload").exists(), "prune deleted an actively copied payload"
    release.set()
    thread.join(5)
    assert not thread.is_alive() and out[0].read_bytes() == b"payload"
    inputcache.prune(keep_bytes=0)
    assert not slot.exists() and out[0].read_bytes() == b"payload", "lease depended on evicted slot"


def test_get_may_evict_its_own_slot_but_never_its_returned_lease(tmp_path, monkeypatch):
    monkeypatch.setenv(inputcache.MAX_GB_ENV, "0")
    lease = inputcache.get(URL_A, _dl([]), lease=tmp_path / "live")
    slot = inputcache._slot(inputcache.object_key(URL_A))
    assert not slot.exists(), "zero-cap cache should evict the completed persistent entry"
    assert lease.read_bytes() == b"payload", "get returned a path owned by the evicted cache"


def test_prune_cannot_evict_a_slot_while_new_put_is_being_bound_to_it(tmp_path):
    inputcache.get(URL_A, _dl([]), lease=tmp_path / "old-lease")
    slot = inputcache._slot(inputcache.object_key(URL_A))
    src = tmp_path / "new.mp4"
    src.write_bytes(b"new")
    inside_put = threading.Event()
    release = threading.Event()

    def _upload(_snapshot, _url):
        inside_put.set()
        assert release.wait(5)

    thread = threading.Thread(target=lambda: inputcache.upload_and_adopt(URL_A2, src, _upload))
    thread.start()
    assert inside_put.wait(5)
    assert inputcache.prune(keep_bytes=0) == 0
    assert (slot / "payload").exists(), "prune removed a slot during its PUT/adoption transaction"
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    lease = inputcache.get(URL_A, _dl([]), lease=tmp_path / "new-lease")
    assert lease.read_bytes() == b"new"


def test_overlapping_same_object_puts_and_adoptions_have_one_order(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    first_inside = threading.Event()
    release = threading.Event()
    put_order: list[bytes] = []

    def _upload(snapshot, _url):
        body = snapshot.read_bytes()
        if body == b"first":
            first_inside.set()
            assert release.wait(5)
        put_order.append(body)

    t1 = threading.Thread(target=lambda: inputcache.upload_and_adopt(URL_A, first, _upload))
    t2 = threading.Thread(target=lambda: inputcache.upload_and_adopt(URL_A2, second, _upload))
    t1.start()
    assert first_inside.wait(5)
    t2.start()
    import time
    time.sleep(0.05)
    assert not put_order, "second same-object PUT overtook the first while its cache adoption was pending"
    release.set()
    for thread in (t1, t2):
        thread.join(5)
        assert not thread.is_alive()
    assert put_order == [b"first", b"second"]
    lease = inputcache.get(URL_A, _dl([]), lease=tmp_path / "final")
    assert lease.read_bytes() == put_order[-1] == b"second"
