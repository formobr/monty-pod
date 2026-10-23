"""A presigned PUT's deadline is sized to the object and the uplink it shares, and a 4xx is never re-sent.

Prod 2026-10-01: three concurrent previews spent 680/811/460 s in `put` because every object under 30 MiB
got a flat 30 s read window, timed out on a shared uplink and was re-sent from byte 0 — and an expired
presign (403) was retried with backoff although it can only fail again.
"""
from __future__ import annotations

import http.client
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
import requests

from podagent import cp

_URL = ("https://r2.example/object?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request"
        "&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6")
_BUSY_POD_TRANSFERS = 64


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error for url: {_URL}", response=self)


class _Link:
    """A slow-but-steady uplink: an attempt whose read window is shorter than size/rate times out, exactly
    like a socket send blocked on a shared uplink. Answers with `statuses` in order once the body is sent."""

    def __init__(self, rate_bytes_per_s: float, statuses: list[int] | None = None) -> None:
        self.rate = rate_bytes_per_s
        self.statuses = list(statuses or [200])
        self.timeouts: list[Any] = []

    def put(self, url: str, data: Any, headers: dict[str, str], timeout: Any) -> _Resp:
        self.timeouts.append(timeout)
        read_s = timeout[1] if isinstance(timeout, tuple) else timeout
        needed_s = len(data) / self.rate
        if read_s < needed_s:
            raise requests.exceptions.ReadTimeout(f"read timed out ({read_s:.0f}s < {needed_s:.0f}s)")
        sent = sum(len(chunk) for chunk in data)
        assert sent == len(data)
        return _Resp(self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0])


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(cp.time, "sleep", calls.append)
    monkeypatch.setattr(cp, "_PUT_FLOOR_BYTES_PER_S", cp._MEASURED_STREAM_BYTES_PER_S / _BUSY_POD_TRANSFERS)
    cp.put_rate.reset()
    cp.put_trace.reset()
    yield calls
    cp.put_rate.reset()


def _obj(tmp_path: Path, size: int) -> Path:
    src = tmp_path / "out.mp4"
    with src.open("wb") as f:
        f.truncate(size)
    return src


def test_put_deadline_scales_and_4xx_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    sleeps: list[float]) -> None:
    # (1) 40 MiB on a busy pod's slice of the uplink (~0.38 MB/s, ~110 s of wire) completes in ONE attempt —
    # the old flat `max(30, size // 1 MiB)` = 40 s window cut it off and re-sent it from byte 0.
    size = 40 << 20
    link = _Link(rate_bytes_per_s=0.38e6)
    monkeypatch.setattr(cp, "_store", link)
    old_window = max(cp._TIMEOUT, size // (1 << 20))
    assert old_window < size / link.rate, "the scenario must be one the flat window could not survive"
    cp.upload(_obj(tmp_path, size), _URL)
    assert len(link.timeouts) == 1 and sleeps == []
    assert [r["outcome"] for r in cp.put_trace.collect()] == ["ok"]

    # The deadline grows with size, and with how many PUTs share the measured uplink right now.
    cp.put_rate.reset()
    assert cp._put_deadline_s(400 << 20) > cp._put_deadline_s(40 << 20) > cp._put_deadline_s(4 << 20)
    cp.put_rate.end(100 << 20, 4.0)  # one PUT alone measured 25 MiB/s
    alone = cp._put_deadline_s(size)
    for _ in range(16):
        cp.put_rate.begin()
    shared = cp._put_deadline_s(size)
    assert shared > alone
    assert shared == pytest.approx(
        cp._PUT_CONNECT_MARGIN_S + size / ((100 << 20) / 4.0 / 16 / cp._PUT_RATE_SLACK))
    cp.put_rate.reset()

    # (2) An expired presign (403) fails after ONE attempt, no sleep, status named, no url leaked.
    cp.put_trace.reset()
    link = _Link(rate_bytes_per_s=100e6, statuses=[403])
    monkeypatch.setattr(cp, "_store", link)
    with pytest.raises(cp.PutRejected, match="HTTP 403") as exc:
        cp.upload(_obj(tmp_path, 1 << 20), _URL)
    assert isinstance(exc.value, requests.HTTPError)
    assert "Signature" not in str(exc.value)
    assert len(link.timeouts) == 1 and sleeps == []
    assert [r["outcome"] for r in cp.put_trace.collect()] == ["error"]

    # 408/429 are the 4xx answers that DO mean "try again".
    for status in (408, 429):
        sleeps.clear()
        link = _Link(rate_bytes_per_s=100e6, statuses=[status, 200])
        monkeypatch.setattr(cp, "_store", link)
        cp.upload(_obj(tmp_path, 1 << 20), _URL)
        assert len(link.timeouts) == 2 and sleeps == [1]

    # (3) A 503 keeps the bounded retry: every attempt, 2**attempt backoff, then the error.
    sleeps.clear()
    cp.put_trace.reset()
    link = _Link(rate_bytes_per_s=100e6, statuses=[503])
    monkeypatch.setattr(cp, "_store", link)
    with pytest.raises(requests.HTTPError) as exc:
        cp.upload(_obj(tmp_path, 1 << 20), _URL)
    assert not isinstance(exc.value, cp.PutRejected)
    assert len(link.timeouts) == cp._XFER_ATTEMPTS
    assert sleeps == [2**a for a in range(cp._XFER_ATTEMPTS - 1)]
    assert [r["outcome"] for r in cp.put_trace.collect()] == ["retry", "retry", "error"]


def test_a_timeout_keeps_the_bounded_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           sleeps: list[float]) -> None:
    link = _Link(rate_bytes_per_s=1.0)  # nothing fits any deadline
    monkeypatch.setattr(cp, "_store", link)
    with pytest.raises(requests.exceptions.ReadTimeout):
        cp.upload(_obj(tmp_path, 1 << 20), _URL)
    assert len(link.timeouts) == cp._XFER_ATTEMPTS
    assert sleeps == [1, 2]
    assert cp.put_rate.per_put() == 0.0  # failed attempts never pose as a measured rate


# The floor and the pool are sized from ONE width recorded at import (`cp._STORE_POOL_WIDTH`); the tests compare
# against that record, never a fresh `_store_pool()` read, whose answer depends on the environment at test time.
def test_floor_is_the_measured_stream_split_across_the_transfer_width() -> None:
    assert cp._MEASURED_STREAM_BYTES_PER_S == 25e6
    assert cp._PUT_FLOOR_BYTES_PER_S == pytest.approx(cp._MEASURED_STREAM_BYTES_PER_S / cp._STORE_POOL_WIDTH)


def test_the_floor_and_the_pool_share_the_recorded_width() -> None:
    assert isinstance(cp._STORE_POOL_WIDTH, int) and cp._STORE_POOL_WIDTH >= 4
    assert cp._PUT_FLOOR_BYTES_PER_S == pytest.approx(cp._MEASURED_STREAM_BYTES_PER_S / cp._STORE_POOL_WIDTH)
    for scheme in ("http://", "https://"):
        adapter = cp._store.get_adapter(scheme + "store.example/object")
        assert adapter._pool_connections == cp._STORE_POOL_WIDTH
        assert adapter._pool_maxsize == cp._STORE_POOL_WIDTH


def test_a_later_env_change_moves_neither_the_floor_nor_the_width(monkeypatch: pytest.MonkeyPatch) -> None:
    width, floor = cp._STORE_POOL_WIDTH, cp._PUT_FLOOR_BYTES_PER_S
    other = width + 7
    monkeypatch.setenv("OPS_MAX_TRANSFERS", str(other))
    assert cp._store_pool() == other, "the env change is real: a fresh read would see it"
    assert cp._STORE_POOL_WIDTH == width
    assert cp._PUT_FLOOR_BYTES_PER_S == floor
    assert cp._PUT_FLOOR_BYTES_PER_S == pytest.approx(cp._MEASURED_STREAM_BYTES_PER_S / cp._STORE_POOL_WIDTH)


# Prod 2026-10-13 (MISC-306): urllib3 2.x sends the request BODY under the CONNECT timeout, so every PUT body on
# a slow pod was cut at the 10 s connect leg whatever its size-derived deadline (see `cp._PUT_SEND_TIMEOUT_S`).
# The PUT below goes through the REAL `cp._store` -> requests -> urllib3 -> socket path to a local HTTP server;
# only the uplink's pace is modelled, at `http.client.HTTPConnection.send` (what urllib3 sends headers and body
# chunks through): a send that the modelled link needs longer for than the socket timeout in force raises the
# exact error the kernel would ("The write operation timed out"). No wall clock, no sleeps: the verdict is a
# comparison of the socket timeout urllib3 actually set against the modelled need.
_SLOW_BODY_NEEDS_S = 15.0


class _Store(BaseHTTPRequestHandler):
    received: list[bytes]

    def do_PUT(self) -> None:  # noqa: N802 — http.server's naming
        want = int(self.headers["Content-Length"])
        body = self.rfile.read(want)
        if len(body) == want:
            self.received.append(body)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


class _ThrottledUplink:
    """`rate` bytes/s, steady, for the first `stall_after` bytes of each connection; nothing after that."""

    def __init__(self, rate_bytes_per_s: float, stall_after: float = math.inf) -> None:
        self.rate, self.stall_after = rate_bytes_per_s, stall_after
        self.sends: list[tuple[int, float | None]] = []  # (bytes, socket timeout in force)
        self._sent: dict[int, int] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_send = http.client.HTTPConnection.send
        link = self

        def send(conn: http.client.HTTPConnection, data: Any) -> None:
            if conn.sock is None:
                conn.connect()
            n = len(data)
            timeout = conn.sock.gettimeout()
            link.sends.append((n, timeout))
            sent = link._sent.get(id(conn.sock), 0)
            need = math.inf if sent + n > link.stall_after else n / link.rate
            if timeout is not None and need > timeout:
                raise TimeoutError("The write operation timed out")
            link._sent[id(conn.sock)] = sent + n
            real_send(conn, data)

        monkeypatch.setattr(http.client.HTTPConnection, "send", send)


@pytest.fixture
def store() -> Iterator[tuple[str, list[bytes]]]:
    received: list[bytes] = []
    handler = type("_StoreHandler", (_Store,), {"received": received})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/object?{_URL.split('?', 1)[1]}"
    try:
        yield url, received
    finally:
        server.shutdown()
        server.server_close()
        cp._store.close()  # no keep-alive socket into a server that is gone


def test_a_slow_uplink_body_completes_within_its_size_derived_deadline(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float],
        store: tuple[str, list[bytes]]) -> None:
    url, received = store
    size = cp._CHUNK  # one body send: the whole object rides one socket op
    payload = os.urandom(size)
    src = tmp_path / "v3.audio.mp3"
    src.write_bytes(payload)
    link = _ThrottledUplink(rate_bytes_per_s=size / _SLOW_BODY_NEEDS_S)
    link.install(monkeypatch)
    monkeypatch.setattr(cp.time, "monotonic", lambda: 0.0)  # frozen: no real-clock dependency in the verdict
    deadline = cp._put_deadline_s(size)
    assert cp._PUT_CONNECT_MARGIN_S < _SLOW_BODY_NEEDS_S <= deadline, \
        "the scenario: a body the 10 s connect leg cannot carry but its own deadline can"

    cp.upload(src, url)

    assert received == [payload]
    assert sleeps == [] and [r["outcome"] for r in cp.put_trace.collect()] == ["ok"]
    body_sends = [t for n, t in link.sends if n == size]
    assert body_sends == [deadline], "the body is sent under the size-derived deadline, not the connect leg"
    assert cp._PUT_SEND_TIMEOUT_S.get() is None, "the send bound never leaks past the PUT"


def test_a_stalled_body_still_fails_at_the_deadline(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float],
        store: tuple[str, list[bytes]]) -> None:
    url, received = store
    size = cp._CHUNK
    link = _ThrottledUplink(rate_bytes_per_s=1e9, stall_after=64 << 10)  # headers go out, the body never does
    link.install(monkeypatch)
    monkeypatch.setattr(cp.time, "monotonic", lambda: 0.0)  # frozen: no real-clock dependency in the verdict
    deadline = cp._put_deadline_s(size)

    with pytest.raises(requests.ConnectionError, match="The write operation timed out"):
        cp.upload(_obj(tmp_path, size), url)

    assert received == []
    assert sleeps == [1, 2], "the attempt count and backoff are unchanged"
    assert [r["outcome"] for r in cp.put_trace.collect()] == ["retry", "retry", "error"]
    stalled = [t for n, t in link.sends if n == size]
    assert stalled == [deadline] * cp._XFER_ATTEMPTS, "each stalled body is cut at its deadline, no later"
    assert cp.put_rate.per_put() == 0.0


class _FakeClock:
    """A monotonic() stand-in the test drives explicitly — no sleeps, no thread timing, no real elapsed time
    anywhere in the verdict."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


class _DummySock:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _DummyConn:
    """Stands in for `http.client.HTTPConnection`: real `request()` sends headers AND every body chunk via
    its own `send()` calls, ALL from within the one `request()` call — `send_plan` reproduces that shape
    (each entry is a clock value to jump to, then a chunk to send) so `_SendBoundMixin`'s deadline stays live
    across them exactly as it would on the real class."""

    def __init__(self, clock: "_FakeClock", send_plan: list[tuple[float, bytes]]) -> None:
        self.sock = _DummySock()
        self._clock = clock
        self._send_plan = send_plan

    def request(self, method: str, url: str, body: Any = None, headers: Any = None, **kw: Any) -> None:
        for t, chunk in self._send_plan:
            self._clock.t = t
            self.send(chunk)

    def send(self, data: Any) -> None:
        pass


class _BoundDummyConn(cp._SendBoundMixin, _DummyConn):
    pass


def test_a_stalled_chunk_does_not_hand_the_next_chunk_a_fresh_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression for the bug a prior fix introduced: `socket.settimeout` bounds ONE syscall, not a running
    # total, so setting `conn.timeout = bound` once (as plain `request()` does for headers) would let a
    # multi-chunk body's SECOND `send()` call start its own fresh `bound`-sized window — a stall after the
    # first chunk would then fail at 2x the size-derived deadline, not at it. `_SendBoundMixin.send()` must
    # instead shrink the window it grants by whatever time the deadline already has behind it, and fail
    # inside the SAME `request()` call once that absolute deadline is reached — not after a fresh window.
    clock = _FakeClock(start=0.0)
    monkeypatch.setattr(cp.time, "monotonic", clock)
    send_plan = [
        (0.0, b"first chunk"),           # remaining = 30.0 - 0.0 = 30.0
        (25.0, b"second chunk"),         # the first chunk alone ate 25s: remaining must be 5.0, not a fresh 30.0
        (29.9, b"third chunk, barely"),  # remaining 0.1 — still inside the ORIGINAL deadline
        (30.0, b"fourth chunk, exhausted"),  # remaining <= 0: must fail here, not after a 4th window
    ]
    conn = _BoundDummyConn(clock, send_plan)
    token = cp._PUT_SEND_TIMEOUT_S.set(30.0)
    try:
        with pytest.raises(TimeoutError, match="The write operation timed out"):
            conn.request("PUT", "/object")
    finally:
        cp._PUT_SEND_TIMEOUT_S.reset(token)

    assert conn.sock.timeouts == pytest.approx([30.0, 5.0, 0.1]), \
        "each send gets the REMAINING time before the absolute deadline, not a fresh `bound` every call"
    assert conn._send_deadline is None, "the deadline is cleared once request() is done, success or failure"
