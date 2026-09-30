"""MISC-209: why the pod's stream reopened reaches the control plane, not just the pod's own stderr.

A rented pod's stderr is unreadable, so the close reason `_fail_connection` knows is carried as the first
frame on the next socket: an ordinary `event` whose `timings.stream_reopen` names the close.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from podagent import event_stream
from podagent.event_stream import EventStream
from podagent.stream_models import PodStreamFrame
from test_cp_retry import _ack, _server
from wire_fixtures import _bundle


@pytest.fixture(autouse=True)
def _fast_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event_stream, "FRAME_WALL_S", 0.15)
    monkeypatch.setattr(event_stream, "OPEN_WALL_S", 0.5)
    monkeypatch.setattr(event_stream, "REOPEN_BACKOFF_S", 0.01)
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 1)
    monkeypatch.setattr(event_stream, "ACK_WINDOW", 4)


def _event(step: str) -> dict[str, Any]:
    return {"stage": "boot", "status": "step", "phase": "started", "step": step}


def test_a_reopened_stream_sends_the_previous_close_reason_in_its_first_event_timings(tmp_path: Path) -> None:
    lock = threading.Lock()
    sockets: list[list[dict[str, Any]]] = []

    def handler(ws: Any) -> None:
        with lock:
            seen: list[dict[str, Any]] = []
            sockets.append(seen)
            index = len(sockets)
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            with lock:
                seen.append(frame)
            # The first socket ACKs its first frame and then goes silent: the prod shape where ACKs stop
            # arriving while the socket still looks open, so only the pod's ACK wall can end it.
            if index == 1 and len(seen) > 1:
                continue
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        try:
            assert stream.send_event(_event("first"), wait=True) is True
            assert stream.send_event(_event("second"), wait=True) is True
        finally:
            stream.close()

    assert len(sockets) == 2, "one ACK-wall close, one reopen"
    first_on_reopen = sockets[1][0]

    # The frame is an ordinary contract-valid event: the shared schema and the pod's own model accept it.
    schema = _bundle()["schemas"]["pod_stream"]
    assert next(Draft202012Validator(schema).iter_errors(first_on_reopen), None) is None
    PodStreamFrame.model_validate(first_on_reopen)
    assert first_on_reopen["type"] == "event"

    event = first_on_reopen["event"]
    assert event["phase"] == "stream_reopen"
    assert "corr_id" not in event, "a stage=boot frame without corr is the api's fleet-infrastructure shape"
    reopen = event["timings"]["stream_reopen"]
    assert reopen["class"] == "ack_wall"
    assert "1/1 unacknowledged" in reopen["message"] and len(reopen["message"]) <= 200
    assert reopen["unacked"] == 1
    assert reopen["open_s"] >= event_stream.FRAME_WALL_S
    # ACKs STOPPED (the first frame's ACK is at least one wall old), rather than the write stalling.
    assert reopen["since_last_ack_s"] >= event_stream.FRAME_WALL_S
    assert "token" not in json.dumps(reopen)

    # The unacknowledged frame is replayed after the report, with its original identity.
    assert [f["event"].get("step") for f in sockets[1][1:]] == ["second"]
    assert sockets[1][1]["seq"] == sockets[0][1]["seq"]


def test_a_first_open_carries_no_reopen_report(tmp_path: Path) -> None:
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
        try:
            assert stream.send_event(_event("only"), wait=True) is True
        finally:
            stream.close()

    assert [f["event"].get("phase") for f in seen] == ["started"]
