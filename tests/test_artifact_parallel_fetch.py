"""artifact.fetch_verified — the whole 4.6 GB weights tar rode ONE connection over a long RTT (measured
10.6 MB/s on a 772 Mbps host); this is the ranged-parallel path and its single-stream fallback guarantees.
"""
from __future__ import annotations

import hashlib
import threading

import pytest
import requests

from podagent import artifact


class _Ref:
    def __init__(self, sha: str, size=None, url: str = "https://example/x.tar") -> None:
        self.url, self.sha256, self.size = url, sha, size


class _FakeResp:
    def __init__(self, status: int, headers: dict, body: bytes) -> None:
        self.status_code, self.headers, self._body = status, headers, body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


def _parse_range(headers):
    if not headers or "Range" not in headers:
        return None
    start, end = headers["Range"].split("=", 1)[1].split("-")
    return int(start), int(end)


def _origin(body, *, honor_range=True, content_range=True, fail_range=None,
           lose_206_for=None, short_for=None, bad_offset_for=None):
    """A fake presigned-URL origin: records every Range asked for, degrades exactly like a real one would,
    and can misbehave on ONE specific part (lost 206 / short body / wrong offset) while the rest is fine."""
    calls = []
    total = len(body)

    def _get(url, headers=None, stream=True, timeout=None):
        rng = _parse_range(headers)
        calls.append(rng)
        if rng is not None and fail_range is not None and rng == fail_range:
            raise requests.exceptions.ConnectionError("dropped mid-flight")
        if rng is not None and lose_206_for is not None and rng == lose_206_for:
            return _FakeResp(200, {"Content-Length": str(total)}, body)
        if rng is not None and honor_range:
            start, end = rng
            out_start = 0 if (bad_offset_for is not None and rng == bad_offset_for) else start
            out = {"Accept-Ranges": "bytes"}
            if content_range:
                out["Content-Range"] = f"bytes {out_start}-{end}/{total}"
            slice_ = body[start:end + 1]
            if short_for is not None and rng == short_for and slice_:
                slice_ = slice_[:-1]
            return _FakeResp(206, out, slice_)
        return _FakeResp(200, {"Content-Length": str(total)}, body)

    return _get, calls


def _body(n=3000):
    return bytes(i % 251 for i in range(n))


def test_a_ranged_origin_is_pulled_by_more_than_one_connection_and_lands_in_order(tmp_path, monkeypatch):
    body = _body()
    sha = hashlib.sha256(body).hexdigest()
    get, calls = _origin(body)
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 100)
    monkeypatch.setenv("ARTIFACT_FETCH_WORKERS", "4")
    dst = tmp_path / "out.tar"
    total = artifact.fetch_verified(_Ref(sha, size=len(body)), dst)
    assert total == len(body)
    assert dst.read_bytes() == body
    part_calls = [c for c in calls if c is not None and c != (0, 0)]
    assert len({c for c in part_calls}) > 1, "the object must be split across more than one range"


def test_an_origin_that_ignores_range_still_produces_a_correct_file(tmp_path, monkeypatch):
    body = _body(500)
    sha = hashlib.sha256(body).hexdigest()
    get, calls = _origin(body, honor_range=False)
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 10)
    dst = tmp_path / "out.tar"
    total = artifact.fetch_verified(_Ref(sha, size=len(body)), dst)
    assert total == len(body)
    assert dst.read_bytes() == body
    assert all(c == (0, 0) or c is None for c in calls[:1]), "only the probe may ask for a range"


def test_an_origin_with_no_content_length_falls_back_to_single_stream(tmp_path, monkeypatch):
    body = _body(500)
    sha = hashlib.sha256(body).hexdigest()
    get, calls = _origin(body, honor_range=True, content_range=False)
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 10)
    dst = tmp_path / "out.tar"
    total = artifact.fetch_verified(_Ref(sha, size=len(body)), dst)
    assert total == len(body)
    assert dst.read_bytes() == body
    assert len(calls) == 2, "one probe, then exactly the single-stream fallback GET"


def test_a_part_that_fails_mid_flight_fails_the_fetch_loudly(tmp_path, monkeypatch):
    body = _body(3000)
    sha = hashlib.sha256(body).hexdigest()
    get, _ = _origin(body, fail_range=(750, 1499))
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 100)
    monkeypatch.setenv("ARTIFACT_FETCH_WORKERS", "4")
    dst = tmp_path / "out.tar"
    with pytest.raises(requests.exceptions.ConnectionError):
        artifact.fetch_verified(_Ref(sha, size=len(body)), dst)


def test_a_part_that_loses_206_after_the_probe_fails_loudly(tmp_path, monkeypatch):
    """The probe getting 206 does not guarantee a PART gets one too — an origin that flips to 200 mid-fetch
    for one range must not be accepted as if it were the requested slice."""
    body = _body(3000)
    sha = hashlib.sha256(body).hexdigest()
    get, _ = _origin(body, lose_206_for=(750, 1499))
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 100)
    monkeypatch.setenv("ARTIFACT_FETCH_WORKERS", "4")
    dst = tmp_path / "out.tar"
    with pytest.raises(artifact.TransferStalled, match="lost 206"):
        artifact.fetch_verified(_Ref(sha, size=len(body)), dst)


def test_a_part_delivering_fewer_bytes_than_declared_fails_loudly(tmp_path, monkeypatch):
    body = _body(3000)
    sha = hashlib.sha256(body).hexdigest()
    get, _ = _origin(body, short_for=(750, 1499))
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 100)
    monkeypatch.setenv("ARTIFACT_FETCH_WORKERS", "4")
    dst = tmp_path / "out.tar"
    with pytest.raises(artifact.TransferStalled, match="delivered"):
        artifact.fetch_verified(_Ref(sha, size=len(body)), dst)


def test_a_part_answering_from_the_wrong_offset_is_a_bad_host_not_corrupt_bytes(tmp_path, monkeypatch):
    """A 206 with the right LENGTH but the wrong START would otherwise sail through and only die at
    _verify as a sha256 mismatch — after the whole object moved, misreported as a bad artifact."""
    body = _body(3000)
    sha = hashlib.sha256(body).hexdigest()
    get, _ = _origin(body, bad_offset_for=(750, 1499))
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 100)
    monkeypatch.setenv("ARTIFACT_FETCH_WORKERS", "4")
    dst = tmp_path / "out.tar"
    with pytest.raises(artifact.TransferStalled, match="offset"):
        artifact.fetch_verified(_Ref(sha, size=len(body)), dst)


def test_a_small_ranged_capable_origin_still_uses_the_single_stream_path(tmp_path, monkeypatch):
    """The size floor: an origin that WOULD honour Range is still not worth splitting below it — the
    fallback GET must carry no Range header, not a width-1 ranged GET, or the floor is not actually gating."""
    body = _body(50)
    sha = hashlib.sha256(body).hexdigest()
    get, calls = _origin(body)
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 1000)
    dst = tmp_path / "out.tar"
    total = artifact.fetch_verified(_Ref(sha, size=len(body)), dst)
    assert total == len(body)
    assert dst.read_bytes() == body
    assert calls == [(0, 0), None], f"expected probe then a bare fallback GET, got {calls}"


def test_sha256_mismatch_still_raises_on_the_ranged_path(tmp_path, monkeypatch):
    body = _body(3000)
    get, _ = _origin(body)
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 100)
    monkeypatch.setenv("ARTIFACT_FETCH_WORKERS", "4")
    dst = tmp_path / "out.tar"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        artifact.fetch_verified(_Ref("0" * 64, size=len(body)), dst)


def test_progress_reports_aggregate_bytes_and_rate_across_workers(tmp_path, monkeypatch):
    body = _body(3000)
    sha = hashlib.sha256(body).hexdigest()
    get, _ = _origin(body)
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 100)
    monkeypatch.setenv("ARTIFACT_FETCH_WORKERS", "4")
    seen = []
    dst = tmp_path / "out.tar"
    artifact.fetch_verified(_Ref(sha, size=len(body)), dst, progress=seen.append)
    assert seen, "aggregate progress must fire at least once across the parallel parts"
    last = seen[-1]
    assert "MB/s avg" in last
    assert f"{len(body) / 1e6:.0f} MB" in last


class _StubFuture:
    def __init__(self, exc):
        self._exc = exc

    def exception(self, timeout=None):
        return self._exc

    def cancel(self):
        return False


class _HangDetectingExecutor:
    """A `shutdown(wait=True)` while any stub future is still outstanding IS the BLOCK-1 hang made
    observable without a real thread ever blocking: it raises instead of actually joining forever."""

    def __init__(self, futures):
        self._futures = list(futures)
        self.shutdown_calls = []
        self.counter = None   # captured from the first submit(), for tests that drive it directly
        self.abort = None

    def submit(self, fn, *a, **kw):
        if self.counter is None:
            self.counter, self.abort = a[4], a[6]
        return self._futures.pop(0)

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))
        if wait and any(f._exc is None for f in self._futures):
            raise AssertionError("shutdown(wait=True) joined a still-outstanding part — the BLOCK-1 hang")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(wait=True)
        return False


def test_a_raised_part_does_not_join_a_sibling_still_streaming(tmp_path, monkeypatch):
    """BLOCK 1: range A raises at once, range B is still "streaming" (never completes). The failure path
    must not join B — driven by stub futures/executor so no real thread ever blocks; a revert to
    `with cf.ThreadPoolExecutor(...) as ex:` makes `__exit__` try to join B and this goes red."""
    body = _body(3000)
    sha = hashlib.sha256(body).hexdigest()
    get, _ = _origin(body)
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_MIN_PART_BYTES", 100)
    monkeypatch.setenv("ARTIFACT_FETCH_WORKERS", "2")

    boom = ConnectionError("range A dropped mid-flight")
    futures = [_StubFuture(boom), _StubFuture(None)]
    fake_ex = _HangDetectingExecutor(futures)
    monkeypatch.setattr(artifact.cf, "ThreadPoolExecutor", lambda *a, **kw: fake_ex)
    monkeypatch.setattr(artifact.cf, "wait", lambda pending, timeout=None, return_when=None:
                         ({f for f in pending if f._exc is not None}, {f for f in pending if f._exc is None}))

    dst = tmp_path / "out.tar"
    with pytest.raises(ConnectionError, match="dropped mid-flight"):
        artifact.fetch_verified(_Ref(sha, size=len(body)), dst)

    assert fake_ex.shutdown_calls == [(False, True)], \
        f"the failure path must call shutdown(wait=False) exactly once, got {fake_ex.shutdown_calls}"
    assert fake_ex.abort.is_set(), \
        "a fatal part must ALSO tell its live siblings to stop — shutdown(wait=False) only stops the " \
        "orchestrator waiting, it leaves an orphan thread streaming the rented pod's bandwidth into a " \
        "staging file that is about to be deleted"


def test_a_part_stops_between_chunks_once_a_sibling_has_failed(tmp_path, monkeypatch):
    """The abort flag's OTHER half: the worker must actually READ it. Set-but-never-checked leaves the
    orphan pulling its whole range long after the fetch it belonged to has already failed."""
    body = _body(3000)
    get, _ = _origin(body)
    monkeypatch.setattr(artifact.requests, "get", get)
    monkeypatch.setattr(artifact, "_CHUNK", 100)

    dst = tmp_path / "part.bin"
    with dst.open("wb") as fh:
        fh.truncate(len(body))

    class _SetAfterOneLook(threading.Event):
        """False on the first look, true on every later one — one chunk gets through, the rest must not."""

        def is_set(self):
            was = threading.Event.is_set(self)
            self.set()
            return was

    counter, lock = [0], threading.Lock()
    artifact._fetch_range("https://example/x.tar", dst, 0, len(body) - 1, counter, lock, _SetAfterOneLook())

    assert counter[0] == 100, f"the worker read past the abort: {counter[0]} of {len(body)} bytes"


def test_the_ranged_stall_floor_still_fires_after_the_first_byte(tmp_path, monkeypatch):
    """Test-gap pin for the floor itself: deleting it must not go unnoticed. Fully deterministic — a
    scripted fake clock and a stub future, no real thread or wait, per this repo's no-real-wait law."""
    fake_ex = _HangDetectingExecutor([_StubFuture(None)])
    monkeypatch.setattr(artifact.cf, "ThreadPoolExecutor", lambda *a, **kw: fake_ex)

    seen = {"n": 0}

    def _wait(pending, timeout=None, return_when=None):
        seen["n"] += 1
        if seen["n"] == 2:
            fake_ex.counter[0] += 1500   # first byte lands, then nothing more ever arrives
        return (set(), pending)
    monkeypatch.setattr(artifact.cf, "wait", _wait)

    ticks = iter([0.0, 25.0, 25.0, 55.0])   # t0; TTFB (0 bytes); first-byte (baseline reset); dead window
    monkeypatch.setattr(artifact.time, "monotonic", lambda: next(ticks))

    dst = tmp_path / "out.tar"
    with pytest.raises(artifact.TransferStalled, match="stalled"):
        artifact._download_ranged(_Ref("0" * 64, size=3000), dst, 3000, None, "artifact")


def test_a_slow_first_byte_does_not_condemn_a_healthy_ranged_transfer(tmp_path, monkeypatch):
    """BLOCK 2 regression: TTFB is not a rate. Same scripted-clock technique as the test above, ending in
    a full, healthy delivery instead of a stall — must complete, not raise."""
    done = _StubFuture(None)
    fake_ex = _HangDetectingExecutor([done])
    monkeypatch.setattr(artifact.cf, "ThreadPoolExecutor", lambda *a, **kw: fake_ex)

    def _wait(pending, timeout=None, return_when=None):
        if fake_ex.counter[0] == 0:
            return (set(), pending)          # still connecting: TTFB, 0 bytes
        return ({done}, set())               # the whole object then lands at once, healthily
    monkeypatch.setattr(artifact.cf, "wait", _wait)

    def _mono(_ticks=iter([0.0, 2.5, 2.6])):
        t = next(_ticks)
        if t == 2.5:
            fake_ex.counter[0] = 3000        # first byte(s) observed right after this TTFB tick
        return t
    monkeypatch.setattr(artifact.time, "monotonic", _mono)

    dst = tmp_path / "out.tar"
    got = artifact._download_ranged(_Ref("0" * 64, size=3000), dst, 3000, None, "artifact")
    assert got == 3000


def test_fetch_width_reads_the_dedicated_knob_not_ops_max_transfers(monkeypatch):
    monkeypatch.delenv("ARTIFACT_FETCH_WORKERS", raising=False)
    assert artifact.range_fetch_width() == artifact._FETCH_WORKERS_DEFAULT
    monkeypatch.setenv("ARTIFACT_FETCH_WORKERS", "3")
    assert artifact.range_fetch_width() == 3
