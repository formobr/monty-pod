"""The pod has one control-plane lane: a strict, durable, ordered WebSocket.

These are transport tests, not mocks of the happy path.  Each failure shape was capable of losing a
terminal when events and results used different in-memory/HTTP mechanisms.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from websockets.sync.server import serve

from podagent import cp, event_stream
from podagent.event_stream import EventStream, FrameRejected


def _ack(frame: dict[str, Any], status: int = 202, **extra: Any) -> str:
    return json.dumps({
        "type": "ack",
        "stream_id": frame["stream_id"],
        "seq": frame["seq"],
        "status": status,
        **extra,
    })


@contextmanager
def _server(handler: Callable[[Any], None]) -> Iterator[str]:
    server = serve(handler, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.socket.getsockname()[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def _wait_for(predicate: Callable[[], bool], wall: float = 3.0) -> None:
    deadline = time.monotonic() + wall
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _fast_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event_stream, "FRAME_WALL_S", 0.15)
    monkeypatch.setattr(event_stream, "OPEN_WALL_S", 0.15)
    monkeypatch.setattr(event_stream, "REOPEN_BACKOFF_S", 0.01)
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 1)


def test_result_is_a_typed_frame_and_requires_correlation(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame)
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        with pytest.raises(ValueError, match="corr_id or result_key"):
            stream.send_result({"job_id": "j", "status": "ok"})
        assert stream.send_result({"job_id": "j", "status": "ok", "corr_id": "c"})
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert seen[0] == {
        "type": "result",
        "stream_id": seen[0]["stream_id"],
        "seq": 1,
        "result": {"job_id": "j", "status": "ok", "corr_id": "c"},
    }
    assert seen[1]["type"] == "event"
    acked = seen[1]["event"]
    assert acked["phase"] == "result_acked" and acked["corr_id"] == "c"
    assert acked["op"] == "result" and set(acked["timings"]) == {"delivery_s"}


def test_single_sender_preserves_append_order(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame)
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        for n in range(6):
            assert stream.send_event({"n": n}, wait=False)
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert [f["event"]["n"] for f in seen] == list(range(6))
    assert [f["seq"] for f in seen] == list(range(1, 7))
    assert len({f["stream_id"] for f in seen}) == 1


def test_ack_lost_after_send_replays_the_same_identity(tmp_path: Path) -> None:
    seen: list[tuple[str, int]] = []
    first = True

    def handler(ws: Any) -> None:
        nonlocal first
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            if frame["type"] == "result":
                seen.append((frame["stream_id"], frame["seq"]))
                if first:
                    first = False
                    return  # The peer may have committed the frame; the ACK alone was lost.
                ws.send(_ack(frame, duplicate=True))
            else:
                ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        assert stream.send_result({"job_id": "j", "corr_id": "c", "status": "ok"})
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert len(seen) == 2
    assert seen[0] == seen[1], "ambiguity must replay the original dedupe identity"


def test_process_restart_replays_old_identity_and_new_frames_use_a_new_stream(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    rejected_seen: list[dict[str, Any]] = []
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 0)

    def unavailable(ws: Any) -> None:
        frame = json.loads(ws.recv())
        rejected_seen.append(frame)
        ws.send(_ack(frame, status=503, error="try later"))

    with _server(unavailable) as base:
        first = EventStream(base, "token", outbox_path=path)
        assert not first.send_result(
            {"job_id": "j", "stage": "ops", "status": "ok", "corr_id": "c"}, wait=True)
        first.close()

    old_identity = (rejected_seen[0]["stream_id"], rejected_seen[0]["seq"])
    accepted: list[dict[str, Any]] = []

    def accepting(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            accepted.append(frame)
            ws.send(_ack(frame))

    with _server(accepting) as base:
        second = EventStream(base, "token", outbox_path=path)
        _wait_for(lambda: second.pending_count() == 0)
        assert second.send_event({"phase": "received"}, wait=True)
        second.close()

    assert (accepted[0]["stream_id"], accepted[0]["seq"]) == old_identity
    assert accepted[1]["type"] == "event" and accepted[1]["event"]["phase"] == "result_acked"
    assert accepted[1]["stream_id"] != old_identity[0] and accepted[1]["seq"] == 1
    assert accepted[2]["stream_id"] == accepted[1]["stream_id"] and accepted[2]["seq"] == 2


@pytest.mark.parametrize("reply", [
    {"type": "mystery"},
    {"type": "ack", "stream_id": "wrong", "seq": 1, "status": 202, "extra": True},
])
def test_unknown_or_malformed_ack_never_retires_a_frame(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: dict[str, Any]) -> None:
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 0)

    def handler(ws: Any) -> None:
        ws.recv()
        ws.send(json.dumps(reply))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        assert not stream.send_event({"phase": "started"}, wait=True)
        assert stream.pending_count() == 1
        stream.close()


def test_4xx_is_durably_dead_lettered_and_fails_the_caller(tmp_path: Path) -> None:
    path = tmp_path / "outbox.json"

    def handler(ws: Any) -> None:
        frame = json.loads(ws.recv())
        ws.send(_ack(frame, status=422, error="bad result"))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        with pytest.raises(FrameRejected, match="422"):
            stream.send_result({"job_id": "j", "corr_id": "c", "status": "ok"})
        assert stream.pending_count() == 0
        stream.close()

    state = json.loads(path.read_text())
    assert state["frames"] == []
    assert len(state["rejected"]) == 1
    assert state["rejected"][0]["ack"]["error"] == "bad result"


def test_async_4xx_dead_letters_without_accumulating_an_outcome(tmp_path: Path) -> None:
    path = tmp_path / "outbox.json"

    def handler(ws: Any) -> None:
        frame = json.loads(ws.recv())
        ws.send(_ack(frame, status=409, error="duplicate content key"))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        assert stream.send_event({"phase": "started"})
        _wait_for(lambda: bool(json.loads(path.read_text())["rejected"]))
        assert stream.pending_count() == 0
        assert stream._delivery_outcomes == {}
        stream.close()


def test_persist_failure_while_retiring_keeps_frame_and_unblocks_caller(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"

    def handler(ws: Any) -> None:
        frame = json.loads(ws.recv())
        ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        real_persist = stream._persist_locked
        calls = 0

        def fail_retire() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk became read-only")
            real_persist()

        monkeypatch.setattr(stream, "_persist_locked", fail_retire)
        with pytest.raises(OSError, match="read-only"):
            stream.send_event({"phase": "work_finished"}, wait=True)
        assert stream.pending_count() == 1
        assert len(json.loads(path.read_text())["frames"]) == 1
        stream.close()


def test_unwritable_outbox_refuses_construction_before_claim(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        EventStream,
        "_persist_locked",
        lambda _self: (_ for _ in ()).throw(OSError("read-only outbox")),
    )
    with pytest.raises(OSError, match="read-only outbox"):
        EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")


def test_retry_exhaustion_retains_the_frame(tmp_path: Path) -> None:
    path = tmp_path / "outbox.json"
    hits = 0

    def handler(ws: Any) -> None:
        nonlocal hits
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            hits += 1
            ws.send(_ack(frame, status=503, error="overloaded"))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        assert not stream.send_result({"job_id": "j", "corr_id": "c", "status": "ok"})
        assert stream.pending_count() == 1
        stream.close()

    state = json.loads(path.read_text())
    assert hits == 2
    assert len(state["frames"]) == 1
    assert state["rejected"] == []


def test_jobs_are_typed_and_unwrapped(tmp_path: Path) -> None:
    def handler(ws: Any) -> None:
        ws.send(json.dumps({
            "type": "job", "job": {"type": "infer", "session_id": "s", "corr_id": "c"}}))
        time.sleep(0.1)

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        assert stream.claim(1) == {"type": "infer", "session_id": "s", "corr_id": "c"}
        stream.close()


def test_control_plane_delegates_one_result_without_a_second_lane() -> None:
    calls: list[tuple[dict[str, Any], bool]] = []

    class _Stream:
        def send_result(self, payload: dict[str, Any], *, wait: bool = True) -> bool:
            calls.append((payload, wait))
            return True

    plane = object.__new__(cp.ControlPlane)
    plane._stream = _Stream()
    plane.send_result({
        "job_id": "j", "stage": "ops", "status": "ok", "session_id": "s", "corr_id": "c"})

    assert calls == [({
        "job_id": "j", "stage": "ops", "status": "ok", "session_id": "s", "corr_id": "c"}, True)]
