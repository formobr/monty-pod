"""TRK-105: a pod that reconnects mid-chain rejoins and keeps serving — the pod-agent side.

The fake api below is built on `_server`/`_ack` (tests/test_cp_retry.py:81-121) and `valid_wire`
(wire_fixtures.py) and implements EXACTLY the facts the api repo's own tests pin (S1, api branch
trk105-api-facts, cmd/api/pod_stream_test.go):

  (a) TestFleetDuplicateReadyFromOldStreamOpensAdmissionUnderThatStream
  (b) TestFleetReplayIsStampedWithAdmissionStream (+ a job_ack under an unrelated stream gets 409)
  (c) TestFleetOldStreamJobAckAcceptedByDurableFallbackWithinReceiptTTL
  (d) TestPodFrameRefusalTextsArePinned

Nothing here calls into the real api; these are the pod's own reconnect-mid-chain behaviours, proven
against a minimal in-process double that answers exactly the way the real handler is pinned to answer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from podagent import event_stream, main as agent_main
from podagent.cp import ControlPlane
from podagent.event_stream import DeliveryPending, EventStream, FrameRejected
from test_cp_retry import _ack, _server  # reuse the brief's named helpers, not a redefinition
from wire_fixtures import DELETE, valid_wire

_ATTEMPT_ID = "b" * 32


@pytest.fixture(autouse=True)
def _isolated_live_mark(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")


@pytest.fixture(autouse=True)
def _fast_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event_stream, "FRAME_WALL_S", 0.15)
    monkeypatch.setattr(event_stream, "OPEN_WALL_S", 0.15)
    monkeypatch.setattr(event_stream, "REOPEN_BACKOFF_S", 0.01)
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 1)
    monkeypatch.setattr(event_stream, "ACK_WINDOW", 4)
    monkeypatch.setattr(event_stream, "BACKGROUND_RETRY_S", 0.02)


def _job_body(corr_id: str) -> dict[str, Any]:
    return {
        "type": "infer", "session_id": "s", "corr_id": corr_id,
        "request": {
            "infer_version": 6, "job_id": "j", "kind": "face_probe", "model": "m",
            "put_url": "https://storage.example/out?sig=x",
            "face_probe": {"video_url": "https://storage.example/in?sig=x", "shots": [[0.0, 1.0]],
                           "stride": 1, "frame_diff": False},
        },
    }


def _job_wire(corr_id: str, stream_id: str, seq: int, *, replayed: bool = False) -> dict[str, Any]:
    """A server->pod `job` frame, shaped like `_job()` in test_cp_retry.py (same golden template)."""
    return valid_wire("pod_stream_server.job", {
        "delivery_id": corr_id, "attempt_id": _ATTEMPT_ID, "stream_id": stream_id, "seq": seq,
        "replayed": replayed,
        "timeline": {
            "enqueue_min_unix_ns": 1, "enqueue_max_unix_ns": 2,
            "claim_min_unix_ns": 3, "claim_max_unix_ns": 4, "socket_write_min_unix_ns": 5,
        },
        "job": {"chain": DELETE, **_job_body(corr_id)},
    })


def _ready_event(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    event = {"stage": "boot", "status": "step", "phase": "ready", "step": "capability preflight passed"}
    event.update(extra or {})
    return event


class _FakeFleetApi:
    """One worker's admission state — exactly facts (a)-(d), nothing else.

    - A fresh ready ACKs 202 and opens admission under ITS OWN carried stream_id (b: "the reconnect's own
      admission stream, not the stream the job was originally sent under") — this pod never replays an
      inherited ready (PLAN.md §1: dropped at load), so the duplicate-ready-reopens-an-OLD-stream case (a)
      is the api's own behaviour, not something this pod's tests need to drive.
    - job_ack whose corr is in `reassigned` always gets 403 "job_ack corr is not owned by this worker" (d).
    - job_ack under a stream_id that is NOT the live admission stream: accepted if a durable receipt was
      seeded for that exact (stream_id, corr) pair (c), else 409 "pod delivery identity conflict: ..." (b).
    - job_ack under the live admission stream for a job this fake never delivered: 403 not-owned (d).
    - job_ack under the live admission stream for the job it DID deliver: accepted, and the next queued
      job (if any) is delivered next, stamped under the (possibly new) live admission stream (b).
    """

    def __init__(self, *, queue: list[str] | None = None, reassigned: frozenset[str] = frozenset(),
                 receipts: dict[tuple[str, str], bool] | None = None,
                 ack_ready: bool = True) -> None:
        self.admission_stream: str | None = None
        self.owned_corr: str | None = None
        self.queue: list[str] = list(queue or [])
        self.reassigned = reassigned
        self.receipts: dict[tuple[str, str], bool] = dict(receipts or {})
        self.ack_ready = ack_ready
        self.job_ack_verdicts: list[tuple[str, int, str]] = []  # (corr, status, error)
        self.delivered: list[dict[str, Any]] = []

    def handle(self, ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            kind = frame.get("type")
            if kind == "event" and (frame.get("event") or {}).get("stage") == "boot":
                self._handle_ready(ws, frame)
            elif kind == "job_ack":
                self._handle_job_ack(ws, frame)
            else:
                ws.send(_ack(frame))

    def _handle_ready(self, ws: Any, frame: dict[str, Any]) -> None:
        if not self.ack_ready:
            return  # simulate a peer that never confirms readiness at all
        self.admission_stream = frame["stream_id"]
        ws.send(_ack(frame))
        self._maybe_deliver(ws)

    def _maybe_deliver(self, ws: Any) -> None:
        if self.owned_corr is not None or not self.queue:
            return
        corr = self.queue.pop(0)
        wire = _job_wire(corr, self.admission_stream, 1)
        self.owned_corr = corr
        self.delivered.append(wire)
        ws.send(json.dumps(wire))

    def _handle_job_ack(self, ws: Any, frame: dict[str, Any]) -> None:
        corr = frame["job_ack"]["corr_id"]
        if corr in self.reassigned:
            self.job_ack_verdicts.append((corr, 403, "job_ack corr is not owned by this worker"))
            ws.send(_ack(frame, status=403, error="job_ack corr is not owned by this worker"))
            return
        if frame["stream_id"] != self.admission_stream:
            if self.receipts.get((frame["stream_id"], corr)):
                self.job_ack_verdicts.append((corr, 202, ""))
                ws.send(_ack(frame))
                return
            error = "pod delivery identity conflict: frame stream_id does not match the delivered job stream"
            self.job_ack_verdicts.append((corr, 409, error))
            ws.send(_ack(frame, status=409, error=error))
            return
        if self.owned_corr != corr:
            self.job_ack_verdicts.append((corr, 403, "job_ack corr is not owned by this worker"))
            ws.send(_ack(frame, status=403, error="job_ack corr is not owned by this worker"))
            return
        self.receipts[(frame["stream_id"], corr)] = True
        self.owned_corr = None
        self.job_ack_verdicts.append((corr, 202, ""))
        ws.send(_ack(frame))
        self._maybe_deliver(ws)


def _poll_with_retry(cp: ControlPlane, *, attempts: int = 40) -> dict[str, Any] | None:
    """`claim()`'s admission_error check is instantaneous, not a wait — exactly like `_dispatch_loop`'s own
    DeliveryPending handling (main.py), a caller retries with backoff rather than treating one still-in-
    flight background frame (e.g. this incarnation's own fresh job_ack for a just-delivered job) as a
    failure."""
    last: DeliveryPending | None = None
    for _ in range(attempts):
        try:
            return cp.poll_job()
        except DeliveryPending as e:
            last = e
            time.sleep(0.02)
    raise last  # re-raise the last observed ambiguity if it never resolved


def _seed_inherited_state(path: Path, *, job_acks: list[tuple[str, str, int]]) -> None:
    """Durable state exactly as a crashed prior incarnation would leave it: each (corr, old_stream, seq)
    names a job durably received AND durably job_ack'd (never confirmed) — mirrors the crashloop PLAN.md
    §0 diagnoses (a stuck agent's unconfirmed job_ack outlives it)."""
    first = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    try:
        for corr, old_stream, seq in job_acks:
            first._accept_job(_job_wire(corr, old_stream, seq))
            first.send_job_ack(
                {"delivery_id": corr, "corr_id": corr, "attempt_id": _ATTEMPT_ID,
                 "client_recv_mono_ns": seq}, wait=False)
    finally:
        first.close()


def test_restart_with_inherited_ready_and_job_rejoins_and_serves_the_next_job(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    # Two inherited, unconfirmed job_acks from a prior incarnation, under an old stream this new
    # incarnation never chose: one gets 409 (no durable receipt under that old stream — b), one gets 403
    # (the pool already reassigned it elsewhere — d). Both must dead-letter exactly once, not brick admission.
    _seed_inherited_state(path, job_acks=[("jobA", "s-old", 1), ("jobB", "s-old", 2)])

    api = _FakeFleetApi(queue=["jobC"], reassigned=frozenset({"jobB"}))
    monkeypatch.setenv("POD_STREAM_OUTBOX", str(path))

    with _server(api.handle) as base:
        cp = ControlPlane(base, "token")
        try:
            t0 = time.monotonic()
            agent_main._report_ready(cp)  # must return cleanly — no SystemExit, no uncaught DeliveryPending
            elapsed = time.monotonic() - t0
            assert elapsed < 2 * event_stream._delivery_wall_s()

            # jobA's OWN inherited job_ack dead-lettered, but the job itself was KEPT (§1: "the inbox job
            # is kept") — it is still the first thing `claim()` serves, exactly as if nothing crashed.
            resumed = _poll_with_retry(cp)
            assert resumed is not None and resumed["corr_id"] == "jobA", \
                "the inherited job must still be servable after its own job_ack dead-lettered"
            # jobB was reassigned elsewhere (403) and is gone; the freshly delivered jobC proves admission
            # for NEW work stayed open behind both dead-letters, not just for the one job already in hand.
            fresh = _poll_with_retry(cp)
            assert fresh is not None and fresh["corr_id"] == "jobC"
        finally:
            cp.close_stream()

    inherited = [v for v in api.job_ack_verdicts if v[0] in ("jobA", "jobB")]
    verdicts = {corr: (status, error) for corr, status, error in inherited}
    assert verdicts["jobA"][0] == 409 and verdicts["jobA"][1].startswith("pod delivery identity conflict: ")
    assert verdicts["jobB"] == (403, "job_ack corr is not owned by this worker")
    assert len(inherited) == 2, "each inherited job_ack must be dead-lettered exactly once, never retried"


def test_a_ready_that_exhausts_its_cap_is_re_appended_at_the_head(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event_stream, "DELIVERY_PENDING_MAX_ATTEMPTS", 2)
    monkeypatch.setenv("POD_STREAM_OUTBOX", str(tmp_path / "outbox.json"))
    api = _FakeFleetApi(ack_ready=False)  # never confirms readiness — forces the cap

    class _StopTest(Exception):
        pass

    with _server(api.handle) as base:
        cp = ControlPlane(base, "token")
        try:
            keys: list[tuple[str, int]] = []
            real_announce = cp.announce_ready

            def _counting_announce(event: dict[str, Any]) -> tuple[str, int]:
                key = real_announce(event)
                keys.append(key)
                if len(keys) >= 2:
                    raise _StopTest()
                return key

            monkeypatch.setattr(cp, "announce_ready", _counting_announce)
            with pytest.raises(_StopTest):
                agent_main._report_ready(cp)
            assert len(keys) == 2
            assert keys[0][0] == keys[1][0], "the SAME incarnation's stream_id, not a fresh EventStream"
            assert keys[1][1] != keys[0][1], "a fresh seq — the durable head, not the same stuck frame"
        finally:
            cp.close_stream()


def test_a_403_not_owned_job_ack_removes_the_unclaimed_job_from_the_inbox(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POD_STREAM_OUTBOX", str(tmp_path / "outbox.json"))
    api = _FakeFleetApi(reassigned=frozenset({"jobX"}))

    with _server(api.handle) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        try:
            stream._accept_job(_job_wire("jobX", stream._stream_id, 1))
            assert "jobX" in stream._inbox
            with pytest.raises(FrameRejected, match="403"):
                stream.send_job_ack({
                    "delivery_id": "jobX", "corr_id": "jobX", "attempt_id": _ATTEMPT_ID,
                    "client_recv_mono_ns": 1,
                })

            def _no_longer_inboxed() -> bool:
                return "jobX" not in stream._inbox

            deadline = time.monotonic() + 2.0
            while not _no_longer_inboxed() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert "jobX" not in stream._inbox
            assert "jobX" not in stream._delivery_meta
            assert stream.claim(0.1) is None, "the reassigned job must never be served again by this pod"
        finally:
            stream.close()


def test_an_unmatched_403_keeps_the_latch(tmp_path: Path) -> None:
    """PLAN.md §1: a 403 whose text names no verdict in the table keeps today's fail-closed latch."""
    def handler(ws: Any) -> None:
        frame = json.loads(ws.recv())
        ws.send(_ack(frame, status=403, error="a brand new refusal this table has never seen"))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        try:
            with pytest.raises(FrameRejected, match="403"):
                stream.send_event(_ready_event(), wait=True)
            with pytest.raises(FrameRejected):
                stream.claim(0.1)
        finally:
            stream.close()
