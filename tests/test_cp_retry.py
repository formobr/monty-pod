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
from podagent.event_stream import DeliveryPending, EventStream, FrameRejected, TransportUnhealthy


def _event(**extra: Any) -> dict[str, Any]:
    return {"stage": "test", "status": "step", **extra}


def _result(corr_id: str = "c", **extra: Any) -> dict[str, Any]:
    return {
        "job_id": "j",
        "session_id": "s",
        "corr_id": corr_id,
        "stage": "ops",
        "status": "ok",
        **extra,
    }


def _job(corr_id: str = "c") -> dict[str, Any]:
    return {
        "type": "job",
        "delivery_id": corr_id,
        "job": {
            "type": "infer",
            "session_id": "s",
            "corr_id": corr_id,
            "request": {
                "infer_version": 5,
                "job_id": "j",
                "kind": "face_probe",
                "model": "m",
                "put_url": "https://storage.example/out?sig=x",
                "face_probe": {
                    "video_url": "https://storage.example/in?sig=x",
                    "shots": [[0.0, 1.0]],
                    "stride": 1,
                    "frame_diff": False,
                },
            },
        },
    }


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
        with pytest.raises(TransportUnhealthy, match="invalid result payload"):
            stream.send_result({"job_id": "j", "status": "ok"})
        assert stream.send_result(_result())
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert seen[0] == {
        "type": "result",
        "stream_id": seen[0]["stream_id"],
        "seq": 1,
        "result": _result(),
    }
    assert seen[1]["type"] == "event"
    acked = seen[1]["event"]
    assert acked["phase"] == "result_acked" and acked["corr_id"] == "c"
    assert acked["op"] == "ops" and set(acked["timings"]) == {"delivery_s"}


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
            assert stream.send_event(_event(step=f"n{n}"), wait=False)
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert [f["event"]["step"] for f in seen] == [f"n{n}" for n in range(6)]
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
        assert stream.send_result(_result())
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
        with pytest.raises(DeliveryPending):
            first.send_result(_result(), wait=True)
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
        assert second.send_event(_event(phase="received"), wait=True)
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
        with pytest.raises(DeliveryPending):
            stream.send_event(_event(phase="started"), wait=True)
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
            stream.send_result(_result())
        assert stream.pending_count() == 0
        stream._accept_job(_job())
        assert stream._inbox == {}
        with pytest.raises(FrameRejected):
            stream.claim(0.1)
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
        assert stream.send_event(_event(phase="started"))
        _wait_for(lambda: bool(json.loads(path.read_text())["rejected"]))
        assert stream.pending_count() == 0
        assert stream._delivery_outcomes == {}
        with pytest.raises(FrameRejected):
            stream.claim(0.1)
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
        with pytest.raises(TransportUnhealthy, match="read-only"):
            stream.send_event(_event(phase="work_finished"), wait=True)
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
        with pytest.raises(DeliveryPending):
            stream.send_result(_result())
        assert stream.pending_count() == 1
        with pytest.raises(DeliveryPending):
            stream.claim(0.1)
        stream.close()

    state = json.loads(path.read_text())
    assert hits == 2
    assert len(state["frames"]) == 1
    assert state["rejected"] == []


def test_jobs_are_typed_and_unwrapped(tmp_path: Path) -> None:
    def handler(ws: Any) -> None:
        ws.send(json.dumps(_job()))
        time.sleep(0.1)

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        assert stream.claim(1) == _job()["job"]
        stream.close()


def test_inbox_replays_after_receive_before_run_crash(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    first = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    first._accept_job(_job("work/infer/result.json"))
    first.close()

    second = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    monkeypatch.setattr(second, "_ensure_open", lambda: True)
    assert second.claim(0.2)["corr_id"] == "work/infer/result.json"
    second.close()


def test_reconnect_duplicate_job_is_not_queued_or_run_twice(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)
    stream._accept_job(_job())
    assert stream.claim(0.2)["corr_id"] == "c"
    stream._accept_job(_job())
    assert stream.claim(0.05) is None
    stream.close()


def test_conflicting_duplicate_delivery_latches_before_queued_work_can_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)
    stream._accept_job(_job())
    conflicting = _job()
    conflicting["job"]["request"]["model"] = "different"
    with pytest.raises(TransportUnhealthy, match="different content"):
        stream._accept_job(conflicting)
    with pytest.raises(TransportUnhealthy, match="different content"):
        stream.claim(0.1)
    assert "c" in stream._inbox
    stream.close()


def test_valid_infer_golden_crosses_real_event_stream_with_one_result_identity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from podagent import main as agent_main
    from podagent.cp import ControlPlane

    seen: list[dict[str, Any]] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame)
            ws.send(_ack(frame))

    req = json.loads(
        (Path(__file__).parents[1] / "contracts/examples/infer_request.face_probe.json").read_text())
    corr = "work/session/result.json"
    yunet = tmp_path / "yunet.onnx"

    class _Probe:
        def run(self, _params: Any, _put_url: str) -> float:
            return 0.125

    probe_module = types.ModuleType("podagent.infer_probe")
    probe_module.ProbeService = _Probe
    monkeypatch.setitem(sys.modules, "podagent.infer_probe", probe_module)

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        plane = object.__new__(ControlPlane)
        plane._stream = stream
        assert agent_main._run_infer(
            req, plane, {}, {(yunet, req["model"]): _Probe()}, {}, yunet, False,
            corr_id=corr, session_id="session-1")
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    result = next(frame["result"] for frame in seen if frame["type"] == "result")
    assert result["corr_id"] == corr == result["result_key"]
    assert result["session_id"] == "session-1" and result["kind"] == "face_probe"


def test_result_append_and_inbox_retirement_are_one_durable_transition(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    corr = "work/infer/result.json"
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    stream._accept_job(_job(corr))
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)
    assert stream.claim(0.2)["corr_id"] == corr
    assert stream.send_result(_result(corr, result_key=corr), wait=False)
    state = json.loads(path.read_text())
    assert state["inbox"] == {}
    assert state["frames"][0]["result"]["corr_id"] == corr
    stream.close()


def test_result_append_failure_rolls_back_seq_waiter_and_inbox(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    corr = "work/infer/result.json"
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    stream._accept_job(_job(corr))
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)
    assert stream.claim(0.2)["corr_id"] == corr
    monkeypatch.setattr(
        stream, "_persist_locked",
        lambda: (_ for _ in ()).throw(OSError("fsync failed for https://u:p@store/x?token=secret")),
    )
    with pytest.raises(TransportUnhealthy, match=r"durable append failed.*\[redacted-url\]"):
        stream.send_result(_result(corr, result_key=corr))
    assert stream._next_seq == 1
    assert stream._outbox == [] and stream._delivery_waiters == {}
    assert corr in stream._inbox
    with pytest.raises(TransportUnhealthy):
        stream.claim(0.1)
    stream.close()


def test_replayed_job_before_pending_result_ack_does_not_reconnect_livelock_or_rerun(
        tmp_path: Path) -> None:
    corr = "work/infer/result.json"
    seen: list[str] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame["type"])
            if frame["type"] == "result":
                ws.send(json.dumps(_job(corr)))
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        stream._accept_job(_job(corr))
        assert stream.claim(0.2)["corr_id"] == corr
        assert stream.send_result(_result(corr, result_key=corr))
        _wait_for(lambda: stream.pending_count() == 0)
        assert stream.claim(0.1) is None
        stream.close()
    assert seen == ["result", "event"]


def test_result_acked_persist_failure_replays_result_after_restart(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    accepted: list[str] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            accepted.append(frame["type"])
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        real_persist = stream._persist_locked
        calls = 0

        def fail_result_retire() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("result_acked fsync failed")
            real_persist()

        monkeypatch.setattr(stream, "_persist_locked", fail_result_retire)
        with pytest.raises(TransportUnhealthy, match="result_acked fsync failed"):
            stream.send_result(_result())
        stream.close()

    disk = json.loads(path.read_text())
    assert [f["type"] for f in disk["frames"]] == ["result"]

    with _server(handler) as base:
        replay = EventStream(base, "token", outbox_path=path)
        _wait_for(lambda: replay.pending_count() == 0)
        replay.close()
    assert accepted == ["result", "result", "event"]


def test_stuck_sender_hits_caller_wall_without_leaking_waiter_or_outcome(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    release = threading.Event()
    monkeypatch.setattr(event_stream, "_delivery_wall_s", lambda: 0.05)

    def stuck(_frame: dict[str, Any]) -> tuple[str, None]:
        release.wait(1)
        return "pending", None

    monkeypatch.setattr(stream, "_deliver", stuck)
    assert stream.send_result(_result()) is False
    assert stream._delivery_waiters == {} and stream._delivery_outcomes == {}
    release.set()
    _wait_for(lambda: isinstance(stream._admission_error, DeliveryPending))
    assert stream._delivery_outcomes == {}
    assert stream.pending_count() == 1
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
