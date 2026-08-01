"""The CP transport survives a connection its peer already closed — the recycled keep-alive that failed
two production runs on their first call after minutes of work."""
from __future__ import annotations

import socket
import threading
from typing import Any

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from podagent import cp

_OK = (b"HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\n"
       b"Content-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}")


def _read_request(conn: socket.socket) -> None:
    """Drain headers + body: an answer written into a socket the client is still filling gets an RST."""
    conn.settimeout(5)
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            return
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    want = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            want = int(line.split(b":", 1)[1])
    while len(rest) < want:
        chunk = conn.recv(4096)
        if not chunk:
            return
        rest += chunk


class _FlakyCP:
    """A control plane that hangs up on its first `drops` connections without answering, then accepts."""

    def __init__(self, drops: int) -> None:
        self.drops = drops
        self.hits = 0
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(16)
        self.base = f"http://127.0.0.1:{self._srv.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            self.hits += 1
            try:
                _read_request(conn)
                if self.hits > self.drops:
                    conn.sendall(_OK)
                    conn.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            finally:
                conn.close()

    def close(self) -> None:
        self._srv.close()


@pytest.fixture
def instant_backoff(monkeypatch: Any) -> None:
    monkeypatch.setattr(cp, "_CP_RETRY_BACKOFF_S", 0.0)


def test_post_event_survives_one_dropped_connection(instant_backoff: None) -> None:
    server = _FlakyCP(drops=1)
    try:
        plane = cp.ControlPlane(server.base, "job-token")
        plane.post_event({"stage": "ops", "status": "step", "step": "probe"})
    finally:
        server.close()
    assert server.hits == 2, "one drop must cost a socket, not the run behind it"


def test_a_control_plane_that_always_hangs_up_still_fails(instant_backoff: None,
                                                          monkeypatch: Any) -> None:
    monkeypatch.setattr(cp, "_CP_RETRY_TOTAL", 2)
    server = _FlakyCP(drops=10_000)
    try:
        plane = cp.ControlPlane(server.base, "job-token")
        with pytest.raises(requests.RequestException):
            plane.post_event({"stage": "ops", "status": "error", "error": "boom"})
    finally:
        server.close()
    assert server.hits == 3, "bounded by _CP_RETRY_TOTAL — a dead endpoint is never an infinite wait"


def test_poll_job_survives_one_dropped_connection(instant_backoff: None) -> None:
    server = _FlakyCP(drops=1)
    try:
        plane = cp.ControlPlane(server.base, "job-token")
        assert plane.poll_job() == {"ok": True}
    finally:
        server.close()
    assert server.hits == 2


def test_a_post_that_may_have_run_is_never_replayed() -> None:
    retry = cp._LoudRetry(total=3, read=3, allowed_methods=None)
    err = ReadTimeoutError(None, "/pod/event", "read timed out")  # type: ignore[arg-type]
    with pytest.raises(ReadTimeoutError):
        retry.increment("POST", "/pod/event", error=err)


def test_a_read_timeout_on_the_long_poll_is_replayable() -> None:
    retry = cp._LoudRetry(total=3, read=3, allowed_methods=None)
    err = ReadTimeoutError(None, "/pod/job", "read timed out")  # type: ignore[arg-type]
    assert retry.increment("GET", "/pod/job", error=err).total == 2
