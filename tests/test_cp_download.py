"""podagent.cp.download — the HTTP body path had zero coverage (`_pump`/`ShortBody`/`If-Range` untouched),
and the parallel-ranged fast path layered on top has named corruption vectors from two reviews. Style
follows test_artifact_parallel_fetch.py: a fake `_store` origin, not a real socket."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest
import requests

from podagent import cp


class _FakeResp:
    """A `requests` response shaped just enough for cp.py: a context manager, `raise_for_status`,
    `iter_content` that can be told to die partway through (a torn body, or a real transport drop)."""

    def __init__(self, status: int, headers: dict[str, str], body: bytes, *,
                fail_after: int | None = None) -> None:
        self.status_code, self.headers, self._body = status, headers, body
        self._fail_after = fail_after

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size: int):
        sent = 0
        for i in range(0, len(self._body), chunk_size):
            if self._fail_after is not None and sent >= self._fail_after:
                raise requests.exceptions.ConnectionError("dropped mid-flight")
            chunk = self._body[i:i + chunk_size]
            yield chunk
            sent += len(chunk)


def _parse_range(headers: dict[str, str] | None) -> tuple[int, int] | None:
    if not headers or "Range" not in headers:
        return None
    spec = headers["Range"].split("=", 1)[1]
    start_s, end_s = spec.split("-", 1)
    return int(start_s), (int(end_s) if end_s else -1)


def _ranged_temp(dest: Path) -> Path:
    """The exact temp path `_download_ranged` computes for a call made on THIS (the test's) thread."""
    return dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.ranged-part")


def _origin(payload: bytes, etag: str, *, honor_range: bool = True,
           worker_response: Callable[[int, int], _FakeResp] | None = None,
           single_stream_response: Callable[[], _FakeResp] | None = None):
    """A fake presigned-URL object store: records every (Range, If-Range) asked for, thread-safe (parallel
    ranged workers call it concurrently from a real ThreadPoolExecutor)."""
    total = len(payload)
    calls: list[tuple[str | None, str | None]] = []
    lock = threading.Lock()

    def get(url: str, *, headers: dict[str, str] | None = None, stream: bool = True,
           timeout: Any = None, allow_redirects: bool = True) -> _FakeResp:
        headers = headers or {}
        with lock:
            calls.append((headers.get("Range"), headers.get("If-Range")))
        rng = _parse_range(headers)
        if rng is None:
            if single_stream_response is not None:
                return single_stream_response()
            return _FakeResp(200, {"ETag": etag}, payload)
        start, end_raw = rng
        end = total - 1 if end_raw < 0 else min(end_raw, total - 1)
        is_probe = (start, end) == (0, 1)
        if not is_probe and worker_response is not None:
            return worker_response(start, end)
        if not honor_range:
            return _FakeResp(200, {"ETag": etag}, payload)
        return _FakeResp(206, {"Content-Range": f"bytes {start}-{end}/{total}", "ETag": etag},
                         payload[start:end + 1])

    return get, calls


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry backoff is production behavior worth keeping, not worth PAYING for in every test."""
    monkeypatch.setattr(cp.time, "sleep", lambda _s: None)


# Characterization: today's single-stream path, exercised over a fake origin for the first time ever.

def test_single_stream_full_download_matches_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = bytes((i * 37) % 256 for i in range(100_000))
    get, calls = _origin(payload, '"etag-1"')
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    got = cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert got == dest
    assert dest.read_bytes() == payload
    assert calls[0][0] == "bytes=0-1", "attempt 0 always probes for ranged eligibility first"
    assert calls[1][0] is None, "below threshold: the real fetch carries no Range header"


def test_single_stream_resumes_via_range_after_a_dropped_connection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = bytes((i * 7) % 256 for i in range(200_000))
    etag = '"stable-etag"'
    drop_at = 50_000
    monkeypatch.setattr(cp, "_CHUNK", 20_000)

    def single_stream_response() -> _FakeResp:
        return _FakeResp(200, {"ETag": etag}, payload, fail_after=drop_at)

    get, calls = _origin(payload, etag, single_stream_response=single_stream_response)
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    got = cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert got == dest
    assert dest.read_bytes() == payload, "the resumed tail must splice onto exactly the dropped prefix"
    resume_calls = [c for c in calls if c[0] not in (None, "bytes=0-1")]
    assert resume_calls, "a dropped single-stream body must be resumed via a Range request"
    assert resume_calls[0][1] == etag, "the resume must pin If-Range to the ETag seen on the failed attempt"


def test_single_stream_short_body_raises_and_deletes_dest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Content-Length the body didn't deliver, with NO transport error: `_pump`'s iterator just ends
    early. `ShortBody` must fire and `dest` must not survive with a torn prefix on disk."""
    payload = b"y" * 500_000

    def single_stream_response() -> _FakeResp:
        return _FakeResp(200, {"Content-Length": "999999", "ETag": '"e"'}, payload)

    get, calls = _origin(payload, '"e"', single_stream_response=single_stream_response)
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    with pytest.raises(cp.ShortBody, match="expected 999999"):
        cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert not dest.exists()
    assert len(calls) == 1 + cp._XFER_ATTEMPTS, "one probe, then every single-stream attempt exhausted"


# Ranged fast path.

@pytest.fixture
def _small_ranged_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real objects trigger ranged at 64 MiB with 32 MiB parts; tests use bytes, not gigabytes."""
    monkeypatch.setattr(cp, "_RANGE_THRESHOLD_BYTES", 40)
    monkeypatch.setattr(cp, "_RANGE_MIN_PART_BYTES", 8)
    monkeypatch.setenv("OPS_RANGE_WORKERS", "3")


def test_ranged_happy_path_assembles_exact_content(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _small_ranged_thresholds: None) -> None:
    payload = bytes(range(96)) * 2  # 192 bytes, well over the 40-byte test threshold
    get, calls = _origin(payload, '"etag-ranged"')
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert dest.read_bytes() == payload, "assembled bytes must equal the known payload exactly"
    assert not _ranged_temp(dest).exists()
    worker_calls = [c for c in calls if c[0] not in (None, "bytes=0-1")]
    assert len({c[0] for c in worker_calls}) > 1, "the object must be split across more than one range"
    assert all(c[0] is None or c[1] == '"etag-ranged"' for c in worker_calls), \
        "every ranged worker must pin If-Range to the probed ETag"


def test_ranged_etag_change_mid_fetch_falls_back_and_leaves_dest_correct(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _small_ranged_thresholds: None) -> None:
    payload = bytes(range(96)) * 2
    total = len(payload)

    def worker_response(start: int, end: int) -> _FakeResp:
        return _FakeResp(206, {"Content-Range": f"bytes {start}-{end}/{total}", "ETag": '"replaced"'},
                         payload[start:end + 1])

    get, calls = _origin(payload, '"original"', worker_response=worker_response)
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert dest.read_bytes() == payload, "the fallback single-stream fetch must still land correct bytes"
    assert not _ranged_temp(dest).exists(), "ranged temp must be cleaned up"
    assert any(c[0] is None for c in calls), "a failed ranged attempt must fall through to single-stream"


def test_ranged_malformed_content_range_falls_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _small_ranged_thresholds: None) -> None:
    payload = bytes(range(96)) * 2

    def worker_response(start: int, end: int) -> _FakeResp:
        return _FakeResp(206, {"Content-Range": "not-a-content-range", "ETag": '"etag"'},
                         payload[start:end + 1])

    get, calls = _origin(payload, '"etag"', worker_response=worker_response)
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert dest.read_bytes() == payload
    assert not _ranged_temp(dest).exists()
    assert any(c[0] is None for c in calls)


def test_ranged_part_short_body_falls_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _small_ranged_thresholds: None) -> None:
    """Makes explicit what `_ranged_worker`'s own `got == expected` check already guards: a 206 body
    shorter than its declared Content-Range span must never be assembled, only fall back."""
    payload = bytes(range(96)) * 2
    total = len(payload)

    def worker_response(start: int, end: int) -> _FakeResp:
        full = payload[start:end + 1]
        return _FakeResp(206, {"Content-Range": f"bytes {start}-{end}/{total}", "ETag": '"etag"'}, full[:-1])

    get, calls = _origin(payload, '"etag"', worker_response=worker_response)
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert dest.read_bytes() == payload
    assert not _ranged_temp(dest).exists()
    assert any(c[0] is None for c in calls)


def test_ranged_content_encoding_on_a_part_falls_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _small_ranged_thresholds: None) -> None:
    """A Range offset addresses the DECODED stream; a transcoded ranged part can't honour that (mirrors the
    single-stream resume guard at cp.py's Content-Encoding check)."""
    payload = bytes(range(96)) * 2
    total = len(payload)

    def worker_response(start: int, end: int) -> _FakeResp:
        return _FakeResp(206, {"Content-Range": f"bytes {start}-{end}/{total}", "ETag": '"etag"',
                               "Content-Encoding": "gzip"}, payload[start:end + 1])

    get, calls = _origin(payload, '"etag"', worker_response=worker_response)
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert dest.read_bytes() == payload
    assert any(c[0] is None for c in calls), "Content-Encoding on a ranged part must fall back"


def test_ranged_fallback_wipes_stale_dest_and_starts_the_retry_from_zero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _small_ranged_thresholds: None) -> None:
    """A pre-existing `dest` (a prior crashed attempt, same deterministic runner.py path) must never survive
    into the fallback's resume math — "attempt 0 always truncates" must hold for this path too."""
    payload = bytes(range(96)) * 2

    def worker_response(start: int, end: int) -> _FakeResp:
        return _FakeResp(206, {"Content-Range": "garbage", "ETag": '"etag"'}, payload[start:end + 1])

    get, calls = _origin(payload, '"etag"', worker_response=worker_response)
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"
    dest.write_bytes(b"STALE-LEFTOVER-FROM-A-PRIOR-CRASHED-ATTEMPT" * 10)

    cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert dest.read_bytes() == payload, "stale bytes must never be spliced onto the real object"
    fallback_calls = [c for c in calls if c[0] is None]
    assert fallback_calls == [(None, None)], \
        "the fallback's first request must be a fresh full GET (no Range/If-Range) — proof `have == 0`"


def test_ranged_oserror_during_presize_falls_back_to_single_stream(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _small_ranged_thresholds: None) -> None:
    """ANY failure escaping the ranged path — not just requests.RequestException — must still degrade to
    the proven single-stream path (an ENOSPC on the temp pre-size must not lose the whole download)."""
    payload = bytes(range(96)) * 2
    get, calls = _origin(payload, '"etag"')
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    real_open = Path.open

    def failing_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if self.name.endswith(".ranged-part") and mode == "wb":
            raise OSError(28, "No space left on device")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    dest = tmp_path / "obj.bin"

    got = cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert got == dest
    assert dest.read_bytes() == payload
    assert any(c[0] is None for c in calls), "the OSError must still degrade to a single-stream fallback"


def test_two_concurrent_downloads_to_the_same_dest_use_distinct_temps_and_both_land_correct(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _small_ranged_thresholds: None) -> None:
    """ops/runner.py sanctions two steps racing the SAME dest for the same url; a temp keyed only by
    `dest.name` would let one call's cleanup yank the inode out from under the other's open fds."""
    payload = bytes(range(96)) * 2
    get, calls = _origin(payload, '"etag"')
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"
    errors: list[BaseException] = []

    def run() -> None:
        try:
            cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)
        except BaseException as exc:  # noqa: BLE001 - captured for the assertion, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not any(t.is_alive() for t in threads), "a concurrent download() must not hang"
    assert not errors, f"concurrent download() to the same dest must not raise: {errors}"
    assert dest.read_bytes() == payload
    assert not list(tmp_path.glob(f"{dest.name}.*.ranged-part")), "no ranged temp may survive both winners"
    assert len(calls) >= 2, "both threads must have actually issued probes, not one short-circuiting"


def test_ranged_200_only_origin_uses_single_stream(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _small_ranged_thresholds: None) -> None:
    payload = bytes(range(96)) * 2
    get, calls = _origin(payload, '"etag"', honor_range=False)
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert dest.read_bytes() == payload
    assert len(calls) == 2, "probe (200, not 206) then exactly one single-stream fallback GET"
    assert calls[0][0] == "bytes=0-1" and calls[1][0] is None


def test_ranged_threshold_boundary_just_below_uses_single_stream(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cp, "_RANGE_THRESHOLD_BYTES", 100)
    payload = b"z" * 99  # one byte under the threshold, origin DOES honor ranges
    get, calls = _origin(payload, '"etag"')
    monkeypatch.setattr(cp, "_store", type("S", (), {"get": staticmethod(get)}))
    dest = tmp_path / "obj.bin"

    cp.download("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest)

    assert dest.read_bytes() == payload
    assert len(calls) == 2, "an eligible-but-too-small origin must still take the single-stream path"
    assert calls[1][0] is None


def test_ranged_byte_count_mismatch_skips_rename_and_cleans_temp(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt-and-suspenders on the sum-of-parts invariant: a worker that writes the right bytes but LIES
    about how many it moved must still block the rename — never repeat artifact.py's bug of trusting the
    probed total instead of bytes actually moved."""
    total = 64
    etag = '"etag"'

    def lying_worker(url: str, temp: Path, start: int, end: int, tot: int, et: str,
                     abort: threading.Event) -> int:
        expected = end - start + 1
        with temp.open("r+b") as fh:
            fh.seek(start)
            fh.write(b"x" * expected)
        return expected - 1  # under-reports by one byte

    monkeypatch.setattr(cp, "_ranged_worker", lying_worker)
    dest = tmp_path / "obj.bin"

    with pytest.raises(cp.ShortBody, match="workers moved"):
        cp._download_ranged("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest, total, etag)

    assert not dest.exists()
    assert not _ranged_temp(dest).exists()


def test_aggregate_deadline_trip_drains_running_workers_then_falls_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Injected clock, no real sleeps: the deadline must trip, and the worker must be observed to LAND
    (drained) before the temp file is torn down — never a teardown under a still-writing thread."""
    clock = {"now": 0.0}
    monkeypatch.setattr(cp.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(cp, "_RANGE_TICK_S", 0.01)
    landed = threading.Event()

    def fake_worker(url: str, temp: Path, start: int, end: int, tot: int, et: str,
                    abort: threading.Event) -> int:
        clock["now"] = 10_000.0  # jump the wall clock past the deadline the instant a worker starts
        while not abort.is_set():
            time.sleep(0.002)
        landed.set()
        return end - start + 1

    monkeypatch.setattr(cp, "_ranged_worker", fake_worker)
    dest = tmp_path / "obj.bin"

    with pytest.raises(cp.TransferTimeout, match="aggregate wall"):
        cp._download_ranged("https://store.example/obj?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6", dest, 64, '"etag"')

    assert landed.is_set(), "the worker must have been drained (observed to stop) before the raise unwound"
    assert not dest.exists()
    assert not _ranged_temp(dest).exists()
