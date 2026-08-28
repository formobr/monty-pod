#!/usr/bin/env python3
"""The pod's single bidirectional control-plane channel.

Durable client frames are a closed union; each wire attempt adds
``client_send_mono_ns`` immediately before its socket write::

    {"type":"event",  "stream_id":str, "seq":int, "event":object}
    {"type":"result", "stream_id":str, "seq":int, "attempt_id":str, "result":object}
    {"type":"job_ack","stream_id":str, "seq":int, "job_ack":object}

Every frame is appended to a disk outbox before any socket write.  One sender
owns frame order and may have a bounded window of writes awaiting ACKs.  A
frame leaves the active outbox, in order, only after a matching 2xx ACK; a
deterministic 4xx moves it to the durable dead-letter.  Reconnects replay only
the unacknowledged suffix with the original identities, and process restarts
cannot turn completed work into silence.

Server frames are also closed: typed ``ack`` and v12 claim-attributed ``job`` objects. Treating an
unknown object as an ACK is a protocol error, because retiring the wrong
terminal desynchronizes the ordered durable stream.
"""
from __future__ import annotations

import json
import os
import queue
import re
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .sanitize import safe_endpoint, safe_error, safe_text
from .stream_models import StreamAck, StreamJob, event_payload, job_ack_payload, result_payload

try:  # pragma: no cover - import shape is exercised through the no-module branch
    import websockets.sync.client as _ws_client
    _WS_IMPORT_ERROR: str | None = None
except Exception as e:  # noqa: BLE001 - absence means the only lane cannot open
    _ws_client = None  # type: ignore[assignment]
    _WS_IMPORT_ERROR = f"{type(e).__name__}: {e}"


FRAME_WALL_S = float(os.environ.get("POD_STREAM_FRAME_WALL_S", "10"))
OPEN_WALL_S = float(os.environ.get("POD_STREAM_OPEN_WALL_S", "10"))
MAX_REOPENS = int(os.environ.get("POD_STREAM_MAX_REOPENS", "2"))
REOPEN_BACKOFF_S = float(os.environ.get("POD_STREAM_REOPEN_BACKOFF_S", "0.25"))
BACKGROUND_RETRY_S = float(os.environ.get("POD_STREAM_BACKGROUND_RETRY_S", "5"))
ACK_WINDOW = min(256, max(1, int(os.environ.get("POD_STREAM_ACK_WINDOW", "32"))))
# ~10 min at the default BACKGROUND_RETRY_S: generous, but bounded — an unbounded DeliveryPending retry
# bricks admission on a billed pod forever if the control plane can never accept this exact frame.
DELIVERY_PENDING_MAX_ATTEMPTS = 120  # ~10 min at BACKGROUND_RETRY_S; a frame the CP cannot settle by then is dead-lettered, not a brick
DISABLED = os.environ.get("POD_STREAM", "").strip() == "0"
_DEFAULT_OUTBOX = "/var/cache/monty/pod-stream/outbox.json"
_STATE_VERSION = 3
_ADMISSION_WAKE = object()
# 422 is the ONLY ack status that is a permanent content verdict; every other 4xx (409 readiness/identity
# races included) is a transport race and gets retried like a 5xx, never dead-lettered.
_CONTENT_REJECT_STATUS = 422


def _delivery_wall_s() -> float:
    """Caller wall for one bounded delivery cycle, derived from the sender's declared component walls."""
    attempts = MAX_REOPENS + 1
    return attempts * (OPEN_WALL_S + FRAME_WALL_S) + MAX_REOPENS * REOPEN_BACKOFF_S + 2.0


def _log(msg: str) -> None:
    print(f"[pod-stream] {safe_text(msg)}", file=sys.stderr, flush=True)


class ProtocolError(ValueError):
    """The peer sent a frame that cannot safely drive this state machine."""


class TransportUnhealthy(RuntimeError):
    """The pod must not admit more paid work while its durable voice is unhealthy."""


class DeliveryPending(TransportUnhealthy):
    """A bounded delivery cycle ended ambiguously; the durable sender keeps retrying."""


class FrameRejected(TransportUnhealthy):
    """The control plane made a deterministic 4xx verdict on one client frame."""

    def __init__(self, frame: dict[str, Any], ack: dict[str, Any]) -> None:
        self.frame, self.ack = frame, ack
        super().__init__(
            f"{frame['type']} frame {frame['stream_id']}:{frame['seq']} rejected "
            f"with {ack['status']}: {safe_text(ack.get('error') or 'no reason', 300)}")


class EventStream:
    """Durable ordered client frames plus typed jobs on one WebSocket."""

    def __init__(self, base_url: str, job_token: str, *,
                 outbox_path: str | Path | None = None) -> None:
        self._url = base_url.rstrip("/").replace("https://", "wss://", 1).replace(
            "http://", "ws://", 1) + "/pod/stream"
        self._token = job_token
        self._path = Path(outbox_path or os.environ.get("POD_STREAM_OUTBOX", _DEFAULT_OUTBOX))

        self._state = threading.RLock()
        self._work = threading.Condition(self._state)
        self._open_lock = threading.Lock()
        self._conn: Any = None
        self._reader: threading.Thread | None = None
        self._stop = False
        self._jobs: "queue.Queue[str | object]" = queue.Queue()
        self._queued_ids: set[str] = set()
        self._claimed_ids: set[str] = set()
        self._ack_waiters: dict[tuple[str, int], threading.Event] = {}
        self._acks: dict[tuple[str, int], dict[str, Any]] = {}
        self._wire_attempts: dict[tuple[str, int], int] = {}
        self._wire_sends: dict[tuple[str, int], tuple[int, int]] = {}
        self._clock_samples: list[dict[str, Any]] = []
        self._delivery_meta: dict[str, dict[str, Any]] = {}
        self._delivery_waiters: dict[tuple[str, int], threading.Event] = {}
        self._delivery_outcomes: dict[tuple[str, int], bool | BaseException] = {}
        self._admission_error: TransportUnhealthy | None = None
        self._storage_error: TransportUnhealthy | None = None
        # Admission state that must be replayed on every newly opened socket
        # (for example worker capacity/ready). The payload is not a claim or
        # result; it is appended to the same durable ordered outbox so it
        # cannot race frame sequencing or vanish on reconnect.
        self._bootstrap_event: dict[str, Any] | None = None
        # A not-yet-settled bootstrap frame's identity: a replayed job_ack must never beat it to the wire.
        self._pending_bootstrap_key: tuple[str, int] | None = None
        # Sender-loop exhaustion cycles per frame identity, since only a bounded count of ambiguous
        # cycles should be able to keep any one frame — and the admission it closes — alive forever.
        self._pending_settlement_attempts: dict[tuple[str, int], int] = {}

        # A process incarnation owns one new stream id. Frames restored from disk keep THEIR original id/seq;
        # mixing an old identity into the new stream makes server-side dedupe unable to identify a replay.
        self._stream_id, self._next_seq = uuid.uuid4().hex, 1
        self._outbox, self._rejected, self._inbox, self._delivery_meta = self._load()
        for row in self._rejected:
            _log(f"durable dead-letter quarantined at boot: {self._frame_identity_line(row.get('frame', {}))} "
                 "— that one identity stays refused, the rest of the agent is not bricked by it")
        if self._outbox:
            self._admission_error = DeliveryPending("startup replay must clear before admitting work")
        # Establish durability at construction, before claiming work. Discovering an unwritable volume only
        # after a paid job finishes would recreate the silent-terminal failure in a different layer.
        with self._state:
            self._persist_locked()
        self._unavailable = DISABLED or _ws_client is None
        if DISABLED:
            _log("POD_STREAM=0 — this pod can neither report results nor claim work; no fallback exists")
        elif _ws_client is None:
            _log(f"websockets unavailable ({_WS_IMPORT_ERROR}) — the pod has no control-plane channel")

        self._sender = threading.Thread(target=self._sender_loop, name="pod-stream-sender", daemon=True)
        self._sender.start()
        with self._work:
            for delivery_id in self._inbox:
                self._queued_ids.add(delivery_id)
                self._jobs.put(delivery_id)
        if self._outbox:
            _log(f"startup replay: {len(self._outbox)} durable frame(s), "
                 f"new_stream={self._stream_id}")
            with self._work:
                self._work.notify_all()

    # ── durable state ──────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _frame_identity_line(frame: dict[str, Any]) -> str:
        """One human line naming a frame for a boot-time log — never raises on a malformed row."""
        kind = frame.get("type")
        body = frame.get(kind) if isinstance(kind, str) else None
        corr = str((body or {}).get("corr_id") or "") if isinstance(body, dict) else ""
        return f"{kind} {frame.get('stream_id')}:{frame.get('seq')} corr={corr!r}"

    @staticmethod
    def _validate_client_frame(frame: Any) -> dict[str, Any]:
        if not isinstance(frame, dict):
            raise ProtocolError("outbox frame is not an object")
        kind = frame.get("type")
        if kind not in ("event", "result", "job_ack"):
            raise ProtocolError(f"outbox frame type must be event|result|job_ack, got {kind!r}")
        keys = {"type", "stream_id", "seq", kind}
        if kind == "result":
            keys.add("attempt_id")
        if set(frame) != keys:
            raise ProtocolError(f"{kind} frame keys must be {sorted(keys)}, got {sorted(frame)}")
        if (not isinstance(frame.get("stream_id"), str) or not frame["stream_id"]
                or len(frame["stream_id"]) > 128
                or re.fullmatch(r"[A-Za-z0-9_.-]+", frame["stream_id"]) is None):
            raise ProtocolError("outbox frame carries no stream_id")
        seq = frame.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ProtocolError(f"outbox seq must be a positive integer, got {seq!r}")
        if kind == "result" and re.fullmatch(r"[0-9a-f]{32}", str(frame.get("attempt_id") or "")) is None:
            raise ProtocolError("result frame carries no v12 attempt_id")
        checked = dict(frame)
        if kind == "event":
            checked[kind] = event_payload(frame.get(kind))
        elif kind == "result":
            checked[kind] = result_payload(frame.get(kind))
        else:
            checked[kind] = job_ack_payload(frame.get(kind))
        return checked

    def _load(self) -> tuple[
            list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]],
    ]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return [], [], {}, {}
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"durable pod state {self._path} is unreadable; refusing to discard it: {safe_error(e)}") from e
        if not isinstance(raw, dict) or set(raw) != {
                "version", "frames", "rejected", "inbox", "deliveries"}:
            raise RuntimeError(f"durable pod outbox {self._path} has an unknown shape; refusing to discard it")
        if raw.get("version") != _STATE_VERSION:
            raise RuntimeError(f"durable pod outbox version {raw.get('version')!r} is unsupported")
        frames = raw.get("frames")
        rejected = raw.get("rejected")
        inbox = raw.get("inbox")
        deliveries = raw.get("deliveries")
        if (not isinstance(frames, list) or not isinstance(rejected, list)
                or not isinstance(inbox, dict) or not isinstance(deliveries, dict)):
            raise RuntimeError("durable pod state frames/rejected/inbox/deliveries have invalid containers")
        checked = [self._validate_client_frame(f) for f in frames]
        identities = [(str(f["stream_id"]), int(f["seq"])) for f in checked]
        if len(identities) != len(set(identities)):
            raise RuntimeError("durable pod outbox repeats a stream_id/seq identity")
        for row in rejected:
            if not isinstance(row, dict) or not isinstance(row.get("frame"), dict):
                raise RuntimeError("durable pod dead-letter carries a malformed row")
            self._validate_client_frame(row["frame"])
        checked_inbox: dict[str, dict[str, Any]] = {}
        for delivery_id, job in inbox.items():
            delivery = deliveries.get(delivery_id)
            if not isinstance(delivery, dict):
                raise RuntimeError(f"durable pod inbox {delivery_id!r} has no delivery identity")
            decoded = StreamJob.model_validate(
                {
                    "type": "job",
                    "delivery_id": delivery_id,
                    "attempt_id": delivery.get("attempt_id"),
                    "stream_id": delivery.get("pod_clock_id"),
                    "seq": delivery.get("delivery_seq"),
                    "replayed": delivery.get("replayed"),
                    "timeline": {key: delivery.get(key) for key in (
                        "enqueue_min_unix_ns", "enqueue_max_unix_ns", "claim_min_unix_ns",
                        "claim_max_unix_ns", "socket_write_min_unix_ns")},
                    "job": job,
                })
            checked_inbox[delivery_id] = decoded.job.model_dump(exclude_none=True, mode="json")
        if set(deliveries) != set(checked_inbox):
            raise RuntimeError("durable pod deliveries do not exactly match inbox identities")
        result_corrs = {
            str(f["result"]["corr_id"]) for f in checked if f["type"] == "result"
        } | {
            str(row["frame"]["result"]["corr_id"])
            for row in rejected if row["frame"]["type"] == "result"
        }
        overlap = set(checked_inbox) & result_corrs
        if overlap:
            raise RuntimeError(
                f"durable pod state repeats completed delivery identity: {sorted(overlap)}")
        return checked, list(rejected), checked_inbox, {
            str(key): dict(value) for key, value in deliveries.items()}

    def _persist_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": _STATE_VERSION,
            "frames": self._outbox,
            "rejected": self._rejected,
            "inbox": self._inbox,
            "deliveries": self._delivery_meta,
        }
        tmp = self._path.with_name(f".{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        try:
            with tmp.open("w") as fh:
                os.chmod(tmp, 0o600)
                fh.write(encoded)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
            dfd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _latch_locked(self, error: TransportUnhealthy, *, storage: bool = False) -> None:
        before = self._admission_error
        if self._admission_error is None or not isinstance(error, DeliveryPending):
            self._admission_error = error
        if storage:
            self._storage_error = self._storage_error or error
        if self._admission_error is not before or storage:
            self._jobs.put(_ADMISSION_WAKE)
        self._work.notify_all()

    def _result_corrs_locked(self) -> set[str]:
        return {
            str(f["result"]["corr_id"]) for f in self._outbox if f["type"] == "result"
        } | {
            str(row["frame"]["result"]["corr_id"])
            for row in self._rejected if row["frame"]["type"] == "result"
        }

    def _append(
            self, kind: str, payload: dict[str, Any], *, wait: bool, attempt_id: str | None = None,
    ) -> tuple[int, threading.Event | None]:
        try:
            if kind == "event":
                normalized = event_payload(payload)
            elif kind == "result":
                normalized = result_payload(payload)
            elif kind == "job_ack":
                normalized = job_ack_payload(payload)
            else:
                raise ProtocolError(f"unknown client frame kind {kind!r}")
        except BaseException as e:
            error = TransportUnhealthy(f"invalid {kind} payload: {safe_error(e)}")
            with self._work:
                self._latch_locked(error)
            raise error from e
        with self._work:
            if self._storage_error is not None:
                raise self._storage_error
            corr_id = str(normalized.get("corr_id") or "")
            if kind == "result" and corr_id in self._result_corrs_locked():
                # This corr_id alone is refused, not the process: latching here would brick every OTHER
                # corr's admission on the strength of one already-settled duplicate.
                raise TransportUnhealthy(f"result corr_id {corr_id!r} already pending or rejected")
            seq = self._next_seq
            frame = {"type": kind, "stream_id": self._stream_id, "seq": seq, kind: normalized}
            if kind == "result":
                frame["attempt_id"] = attempt_id
            self._validate_client_frame(frame)
            self._outbox.append(frame)
            self._next_seq += 1
            waiter = threading.Event() if wait else None
            if waiter is not None:
                self._delivery_waiters[(self._stream_id, seq)] = waiter
            removed_job = self._inbox.pop(corr_id, None) if kind == "result" else None
            removed_delivery = self._delivery_meta.pop(corr_id, None) if kind == "result" else None
            try:
                self._persist_locked()  # append and matching inbox retirement are ONE durable transition
            except BaseException as e:
                if removed_job is not None:
                    self._inbox[corr_id] = removed_job
                if removed_delivery is not None:
                    self._delivery_meta[corr_id] = removed_delivery
                self._outbox.pop()
                self._next_seq -= 1
                self._delivery_waiters.pop((self._stream_id, seq), None)
                error = TransportUnhealthy(f"durable append failed: {safe_error(e)}")
                self._latch_locked(error, storage=True)
                raise error from e
            if kind == "result":
                self._queued_ids.discard(corr_id)
                self._claimed_ids.discard(corr_id)
            self._work.notify_all()
            return seq, waiter

    def _append_bootstrap_head(self, payload: dict[str, Any]) -> int:
        """Insert durably at the outbox HEAD: this frame must win the wire race against anything pending."""
        normalized = event_payload(payload)
        with self._work:
            seq = self._next_seq
            frame = {"type": "event", "stream_id": self._stream_id, "seq": seq, "event": normalized}
            self._validate_client_frame(frame)
            self._outbox.insert(0, frame)
            self._next_seq += 1
            try:
                self._persist_locked()
            except BaseException as e:
                self._outbox.pop(0)
                self._next_seq -= 1
                error = TransportUnhealthy(f"durable bootstrap append failed: {safe_error(e)}")
                self._latch_locked(error, storage=True)
                raise error from e
            self._work.notify_all()
            return seq

    def pending_count(self) -> int:
        with self._state:
            return len(self._outbox)

    # ── public client frames ───────────────────────────────────────────────────────────────────────

    def send_event(self, payload: dict[str, Any], *, wait: bool = False) -> bool:
        """Persist one event, then optionally wait until its strict ACK."""
        seq, waiter = self._append("event", payload, wait=wait)
        return True if waiter is None else self._await_delivery(self._stream_id, seq, waiter)

    def set_bootstrap_event(self, payload: dict[str, Any]) -> None:
        """Replay this admission event after every reconnect.

        The current connection still receives the caller's ordinary event;
        this hook only guarantees the next socket gets a fresh capacity/ready
        frame after the API resets per-worker credit.
        """
        normalized = event_payload(payload)
        with self._work:
            self._bootstrap_event = normalized
            self._work.notify_all()

    def send_result(self, payload: dict[str, Any], *, wait: bool = True) -> bool:
        """Persist one correlated result, then optionally wait until its strict ACK."""
        payload = dict(payload)
        corr_id = str(payload.get("corr_id") or "")
        with self._state:
            meta = self._delivery_meta.get(corr_id)
        attempt_id = str(payload.pop("attempt_id", "") or (meta or {}).get("attempt_id") or "")
        seq, waiter = self._append("result", payload, wait=wait, attempt_id=attempt_id)
        return True if waiter is None else self._await_delivery(self._stream_id, seq, waiter)

    def send_job_ack(self, payload: dict[str, Any], *, wait: bool = True) -> bool:
        """Persist the pod receipt for one exact claim attempt and wait for its clock-bearing ACK."""
        seq, waiter = self._append("job_ack", payload, wait=wait)
        return True if waiter is None else self._await_delivery(self._stream_id, seq, waiter)

    def timeline_context(self, corr_id: str) -> dict[str, Any]:
        """Claim/delivery identity plus exact clock-offset bounds known before the terminal is built."""
        with self._state:
            delivery = self._delivery_meta.get(corr_id)
            samples = [dict(row) for row in self._clock_samples
                       if row["pod_clock_id"] == self._stream_id]
        reasons: list[str] = []
        if delivery is None:
            reasons.append("delivery_context_missing")
        if not samples:
            reasons.append("clock_sync_missing")
        return {
            "complete": not reasons,
            "incomplete_reasons": reasons,
            "pod_clock_id": self._stream_id,
            **({"delivery": dict(delivery), "attempt_id": delivery["attempt_id"]}
               if delivery is not None else {}),
            "clock_sync": samples,
        }

    def _emit_job_ack(self, delivery: dict[str, Any]) -> None:
        self.send_job_ack({
            "delivery_id": delivery["delivery_id"],
            "corr_id": delivery["delivery_id"],
            "attempt_id": delivery["attempt_id"],
            "client_recv_mono_ns": delivery["client_recv_mono_ns"],
        }, wait=False)

    def _await_delivery(self, stream_id: str, seq: int, waiter: threading.Event) -> bool:
        # Sender normally settles a synchronous waiter. This derived outer wall also covers a bug or an
        # unexpected persistence failure inside the daemon, so a terminal caller itself cannot wait forever.
        if not waiter.wait(_delivery_wall_s()):
            with self._work:
                key = (stream_id, seq)
                self._delivery_waiters.pop(key, None)
                raced = self._delivery_outcomes.pop(key, None)
                if isinstance(raced, BaseException):
                    raise raced
                if raced is not None:
                    return bool(raced)
                self._latch_locked(DeliveryPending(
                    f"{stream_id}:{seq} exceeded its caller delivery wall and remains durable"))
            _log(f"frame seq={seq} sender exceeded derived delivery wall {_delivery_wall_s():.2f}s; "
                 "frame remains durable")
            return False
        with self._state:
            key = (stream_id, seq)
            self._delivery_waiters.pop(key, None)
            outcome = self._delivery_outcomes.pop(key, False)
        if isinstance(outcome, BaseException):
            raise outcome
        return bool(outcome)

    # ── one ordered sender ─────────────────────────────────────────────────────────────────────────

    def _sender_loop(self) -> None:
        # A later ACK may arrive before an earlier one on a non-Go test peer. Keep the verdict in memory,
        # but never retire it from disk or wake its caller until every durable predecessor has settled.
        known: dict[
            tuple[str, int], tuple[str, dict[str, Any] | None, float]
        ] = {}
        while True:
            with self._work:
                self._work.wait_for(lambda: self._stop or self._outbox)
                if self._stop:
                    return
                window = [dict(frame) for frame in self._outbox[:ACK_WINDOW]]
                unsent = [frame for frame in window if self._frame_key(frame) not in known]
            attempted: set[tuple[str, int]] = set()
            try:
                if unsent:
                    attempted = {self._frame_key(frame) for frame in unsent}
                    known.update(self._deliver_window(unsent))

                while True:
                    with self._state:
                        if not self._outbox:
                            break
                        frame = dict(self._outbox[0])
                    key = self._frame_key(frame)
                    verdict = known.get(key)
                    if verdict is None:
                        break
                    disposition, ack, delivery_s = verdict
                    self._settle(frame, disposition, ack, delivery_s)
                    known.pop(key, None)

                with self._state:
                    head = dict(self._outbox[0]) if self._outbox else None
                if head is not None and self._frame_key(head) in attempted:
                    head_key = self._frame_key(head)
                    with self._work:
                        attempts = self._pending_settlement_attempts.get(head_key, 0) + 1
                        self._pending_settlement_attempts[head_key] = attempts
                    if attempts > DELIVERY_PENDING_MAX_ATTEMPTS:
                        _log(f"frame {head_key[0]}:{head_key[1]} dead-lettered after {attempts} "
                             "delivery-pending attempts — that corr is lost, other admission continues")
                        self._settle(head, "abandoned", {
                            "status": 0,
                            "error": f"delivery-pending cap ({DELIVERY_PENDING_MAX_ATTEMPTS}) exceeded",
                        }, 0.0)
                    else:
                        # Exhaustion is ambiguous, not a verdict. It wakes a synchronous caller and closes
                        # job admission, while the frame and any ACKed successors stay ordered and durable.
                        self._settle(head, "pending", None, 0.0)
                    with self._work:
                        self._work.wait(timeout=BACKGROUND_RETRY_S)
            except BaseException as e:  # noqa: BLE001 - sender death would strand a synchronous terminal
                with self._state:
                    frame = dict(self._outbox[0]) if self._outbox else None
                if frame is None:
                    continue
                key = self._frame_key(frame)
                _log(f"sender failed while settling {key[0]}:{key[1]} "
                     f"({safe_error(e)}); frame remains durable")
                with self._work:
                    error = TransportUnhealthy(f"sender settlement failed: {safe_error(e)}")
                    self._latch_locked(error, storage=True)
                    if (waiter := self._delivery_waiters.get(key)) is not None:
                        self._delivery_outcomes[key] = error
                        waiter.set()
                    self._work.wait(timeout=BACKGROUND_RETRY_S)

    def _result_acked_frame(self, frame: dict[str, Any], delivery_s: float) -> dict[str, Any]:
        result = frame["result"]
        stage = str(result.get("stage") or ("infer" if result.get("kind") else "result"))
        event: dict[str, Any] = {
            "job_id": result.get("job_id", "unknown"),
            "corr_id": str(result.get("corr_id") or result.get("result_key") or ""),
            "stage": stage,
            "status": "step",
            "phase": "result_acked",
            "op": str(result.get("op") or result.get("kind") or stage),
            "timings": {"delivery_s": round(delivery_s, 3)},
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if result.get("session_id") is not None:
            event["session_id"] = result["session_id"]
        acked = {
            "type": "event",
            "stream_id": self._stream_id,
            "seq": self._next_seq,
            "event": event,
        }
        return self._validate_client_frame(acked)

    def _settle(self, frame: dict[str, Any], disposition: str,
                ack: dict[str, Any] | None, delivery_s: float) -> None:
        key = (str(frame["stream_id"]), int(frame["seq"]))
        with self._work:
            if self._pending_bootstrap_key == key:
                self._pending_bootstrap_key = None
            outcome: bool | BaseException
            if disposition == "accepted":
                if self._outbox and (str(self._outbox[0]["stream_id"]),
                                     int(self._outbox[0]["seq"])) == key:
                    popped = self._outbox.pop(0)
                    acked: dict[str, Any] | None = None
                    if frame["type"] == "result":
                        # The ACK transition itself is durable. This must live here rather than in
                        # ControlPlane.send_result: startup replay has no waiting caller to manufacture it.
                        acked = self._result_acked_frame(frame, delivery_s)
                        self._outbox.append(acked)
                        self._next_seq += 1
                    try:
                        self._persist_locked()
                    except BaseException as e:
                        if acked is not None:
                            self._outbox.pop()
                            self._next_seq -= 1
                        self._outbox.insert(0, popped)
                        raise TransportUnhealthy(
                            f"durable retire failed: {safe_error(e)}") from e
                outcome = True
                self._wire_attempts.pop(key, None)
                self._wire_sends.pop(key, None)
                self._pending_settlement_attempts.pop(key, None)
                if (isinstance(self._admission_error, DeliveryPending)
                        and not self._outbox and self._storage_error is None):
                    self._admission_error = None
            elif disposition in ("rejected", "abandoned"):
                if self._outbox and (str(self._outbox[0]["stream_id"]),
                                     int(self._outbox[0]["seq"])) == key:
                    popped = self._outbox.pop(0)
                    rejected = {
                        "frame": frame,
                        "rejected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "ack": ack,
                    }
                    self._rejected.append(rejected)
                    try:
                        self._persist_locked()
                    except BaseException as e:
                        self._rejected.pop()
                        self._outbox.insert(0, popped)
                        raise TransportUnhealthy(
                            f"durable dead-letter move failed: {safe_error(e)}") from e
                outcome = FrameRejected(frame, ack or {})
                self._wire_attempts.pop(key, None)
                self._wire_sends.pop(key, None)
                self._pending_settlement_attempts.pop(key, None)
                if disposition == "rejected":
                    self._latch_locked(outcome)
                elif (isinstance(self._admission_error, DeliveryPending)
                        and not self._outbox and self._storage_error is None):
                    # This corr alone is lost (already logged loudly by the caller); every other corr's
                    # admission must not stay closed behind a frame this process has given up retrying.
                    self._admission_error = None
            else:
                # Retry exhaustion is not a verdict. Keep the frame durably at the head and continue bounded
                # background attempts, but close admission until the ordered voice catches up.
                outcome = DeliveryPending(
                    f"{frame['type']} frame {frame['stream_id']}:{frame['seq']} remains unacknowledged")
                self._latch_locked(outcome)
            if (waiter := self._delivery_waiters.get(key)) is not None:
                self._delivery_outcomes[key] = outcome
                waiter.set()
            self._work.notify_all()

    @staticmethod
    def _frame_key(frame: dict[str, Any]) -> tuple[str, int]:
        return str(frame["stream_id"]), int(frame["seq"])

    def _deliver_window(
            self, frames: list[dict[str, Any]],
    ) -> dict[tuple[str, int], tuple[str, dict[str, Any] | None, float]]:
        """Write a bounded ordered window, retrying only frames without a terminal ACK."""
        pending = list(frames)
        started = {self._frame_key(frame): time.monotonic() for frame in frames}
        outcomes: dict[tuple[str, int], tuple[str, dict[str, Any] | None, float]] = {}
        for attempt in range(MAX_REOPENS + 1):
            if not pending:
                break
            if self._unavailable:
                break
            if not self._ensure_open():
                if attempt < MAX_REOPENS:
                    time.sleep(REOPEN_BACKOFF_S)
                continue
            with self._state:
                bootstrap_key = self._pending_bootstrap_key
            if bootstrap_key is not None and (not pending or self._frame_key(pending[0]) != bootstrap_key):
                with self._state:
                    head_frame = next(
                        (dict(f) for f in self._outbox if self._frame_key(f) == bootstrap_key), None)
                if head_frame is not None:
                    pending = [head_frame] + [f for f in pending if self._frame_key(f) != bootstrap_key]
                    started.setdefault(bootstrap_key, time.monotonic())
            acks = self._exchange_window(pending)
            retry: list[dict[str, Any]] = []
            for frame in pending:
                key = self._frame_key(frame)
                ack = acks.get(key)
                status = int(ack["status"]) if ack is not None else 0
                delivery_s = time.monotonic() - started[key]
                if 200 <= status < 300:
                    outcomes[key] = ("accepted", ack, delivery_s)
                elif status == _CONTENT_REJECT_STATUS:
                    _log(f"frame seq={frame['seq']} REJECTED status={status} "
                         f"reason={safe_text(ack.get('error') or '', 200)} "
                         "— moved to durable dead-letter")
                    outcomes[key] = ("rejected", ack, delivery_s)
                else:
                    # Every other 4xx (409 readiness/identity races included), a 5xx, or an absent ACK
                    # made no permanent content verdict — retry transport, do not dead-letter.
                    if 400 <= status < 500:
                        _log(f"frame seq={frame['seq']} transport-retryable status={status} "
                             f"reason={safe_text(ack.get('error') or '', 200)}")
                    retry.append(frame)
            pending = retry
            if pending and attempt < MAX_REOPENS:
                time.sleep(REOPEN_BACKOFF_S)
        if pending:
            first, last = pending[0]["seq"], pending[-1]["seq"]
            _log(f"frame seq={first}..{last} delivery retries exhausted — retained in durable outbox")
        return outcomes

    def _exchange_window(
            self, frames: list[dict[str, Any]],
    ) -> dict[tuple[str, int], dict[str, Any]]:
        # `websockets.sync.send()` has no timeout. The watchdog below owns one wall shared by every write and
        # every ACK in this attempt; on expiry, closing the raw socket interrupts a kernel-blocked send.
        deadline = time.monotonic() + FRAME_WALL_S
        keys = [self._frame_key(frame) for frame in frames]
        ready = {key: threading.Event() for key in keys}
        with self._state:
            conn = self._conn
            for key, waiter in ready.items():
                self._ack_waiters[key] = waiter
        if conn is None:
            with self._state:
                for key in keys:
                    self._ack_waiters.pop(key, None)
            return {}
        write_done = threading.Event()
        write_errors: list[BaseException] = []

        def write_window() -> None:
            try:
                # The sender loop is the sole writer. A timed-out daemon may remain stuck in a broken old
                # connection, so a process-global write lock here would also prevent the fresh socket retry.
                for frame in frames:
                    key = self._frame_key(frame)
                    send_ns = time.monotonic_ns()
                    with self._state:
                        wire_attempt = self._wire_attempts.get(key, 0) + 1
                        self._wire_attempts[key] = wire_attempt
                        self._wire_sends[key] = (send_ns, wire_attempt)
                    wire = {**frame, "client_send_mono_ns": send_ns}
                    conn.send(json.dumps(wire, ensure_ascii=False, separators=(",", ":")))
            except BaseException as e:  # noqa: BLE001 - watchdog owns failure and durable replay
                write_errors.append(e)
            finally:
                write_done.set()

        threading.Thread(
            target=write_window, name="pod-stream-window-write", daemon=True).start()
        while not write_done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _log(f"frame window write exceeded the shared {FRAME_WALL_S}s wall")
                self._fail_connection(conn, "write/ACK wall expired")
                break
            if write_done.wait(min(0.02, remaining)):
                break
            with self._state:
                if self._conn is not conn:
                    break
        if write_errors:
            _log(f"frame window write failed ({safe_error(write_errors[0])})")
            self._fail_connection(conn, f"write: {safe_error(write_errors[0])}")

        for key in keys:
            ready[key].wait(max(0.0, deadline - time.monotonic()))
            with self._state:
                if self._conn is not conn:
                    break
        with self._state:
            acks = {key: self._acks.pop(key) for key in keys if key in self._acks}
            for key in keys:
                self._ack_waiters.pop(key, None)
        if len(acks) != len(keys):
            missing = len(keys) - len(acks)
            _log(f"frame window has {missing}/{len(keys)} unacknowledged after {FRAME_WALL_S}s")
            self._fail_connection(conn, "ack timeout or invalid ACK")
        return acks

    # ── strict server frames ───────────────────────────────────────────────────────────────────────

    def _validate_ack(self, frame: dict[str, Any]) -> dict[str, Any]:
        checked = StreamAck.model_validate(frame).model_dump(exclude_none=True, mode="json")
        stream_id, seq = checked["stream_id"], checked["seq"]
        with self._state:
            if (stream_id, seq) not in self._ack_waiters:
                raise ProtocolError(f"ACK {stream_id}:{seq} has no in-flight frame")
        return checked

    def _record_clock_sample_locked(
            self, key: tuple[str, int], ack: dict[str, Any], client_recv_mono_ns: int) -> None:
        sent = self._wire_sends.get(key)
        if sent is None:
            raise ProtocolError(f"ACK {key[0]}:{key[1]} has no client send boundary")
        client_send_mono_ns, wire_attempt = sent
        theta_min = int(ack["server_send_unix_ns"]) - client_recv_mono_ns
        theta_max = int(ack["server_recv_unix_ns"]) - client_send_mono_ns
        if theta_max < theta_min:
            raise ProtocolError(
                f"ACK {key[0]}:{key[1]} produces reversed clock-offset bounds")
        self._clock_samples.append({
            "pod_clock_id": key[0],
            "seq": key[1],
            "wire_attempt": wire_attempt,
            "client_send_mono_ns": client_send_mono_ns,
            "server_recv_unix_ns": int(ack["server_recv_unix_ns"]),
            "server_send_unix_ns": int(ack["server_send_unix_ns"]),
            "client_recv_mono_ns": client_recv_mono_ns,
            "theta_min_ns": theta_min,
            "theta_max_ns": theta_max,
        })
        # The terminal only needs nearby anchors; bounding this prevents a long-lived pod from retaining
        # transport history forever. Exact frame identity remains in every retained sample.
        del self._clock_samples[:-64]

    @staticmethod
    def _raw_job_identity(frame: Any) -> tuple[str, str, str, str, str]:
        """Extract the CP address before strict Python decoding.

        Go may have durably claimed an envelope which is addressable by the
        transport but not representable by this pod's current Pydantic model.
        The rejection path therefore cannot call ``StreamJob.model_validate``
        first: it must retain the delivery/correlation identity long enough to
        emit the typed terminal that releases the CP credit.
        """
        outer = frame if isinstance(frame, dict) else {}
        raw = outer.get("job") if isinstance(outer.get("job"), dict) else {}
        delivery = str(outer.get("delivery_id") or "")
        attempt = str(outer.get("attempt_id") or "")
        corr = str(raw.get("corr_id") or delivery or "")
        session = str(raw.get("session_id") or "")
        job_id = str(raw.get("job_id") or "")
        for block_name in ("request", "spec", "chain"):
            block = raw.get(block_name)
            if isinstance(block, dict) and block.get("job_id"):
                job_id = str(block["job_id"])
                break
        return delivery, attempt, job_id, session, corr

    def _reject_invalid_job(self, frame: Any, error: BaseException) -> bool:
        """Durably terminalize an addressed-but-Python-invalid server job.

        This runs before latching the stream unhealthy.  A malformed payload
        must stop admission, but it must not strand the already claimed CP
        delivery and its credit.  Missing identity is deliberately not filled
        with a fabricated value; such a frame is logged and the CP's protocol
        guard remains authoritative.
        """
        delivery, attempt, job_id, session, corr = self._raw_job_identity(frame)
        detail = f"invalid server job frame: {safe_error(error)}"
        event: dict[str, Any] = {
            "stage": "dispatch",
            "status": "error",
            "phase": "work_finished",
            "outcome": "error",
            "error_type": type(error).__name__,
            "error": detail,
        }
        if job_id:
            event["job_id"] = job_id
        if session:
            event["session_id"] = session
        if corr:
            event["corr_id"] = corr
        try:
            self.send_event(event, wait=False)
            # A terminal is safe only when the outer delivery and inner
            # correlation agree.  Never invent an address for an untrusted
            # malformed frame.
            if job_id and session and corr and (not delivery or delivery == corr):
                self.send_result({
                    "job_id": job_id,
                    "session_id": session,
                    "corr_id": corr,
                    "attempt_id": attempt,
                    "stage": "dispatch",
                    "status": "error",
                    "error": detail,
                }, wait=False)
                return True
        except BaseException as emit_error:  # noqa: BLE001 - original protocol error remains primary
            _log(f"invalid job terminal could not be persisted ({safe_error(emit_error)})")
        return False

    def _accept_job(
            self, frame: dict[str, Any], *, client_recv_mono_ns: int | None = None,
    ) -> dict[str, Any] | None:
        try:
            checked = StreamJob.model_validate(frame)
        except BaseException as e:
            # An addressed claim has a durable typed terminal.  Keep the
            # socket alive so the CP can ACK it and grant the worker's next
            # credit; closing here would strand the claim and cause a fleet-
            # wide retry loop.  Only an unaddressable malformed frame latches
            # the transport below.
            if self._reject_invalid_job(frame, e):
                _log("invalid addressed job terminalized; keeping stream admission alive")
                return None
            error = TransportUnhealthy(f"invalid server job frame: {safe_error(e)}")
            with self._work:
                self._latch_locked(error)
            raise error from e
        delivery_id = checked.delivery_id
        job = checked.job.model_dump(exclude_none=True, mode="json")
        received_ns = client_recv_mono_ns or time.monotonic_ns()
        delivery = {
            "delivery_id": delivery_id,
            "attempt_id": checked.attempt_id,
            "pod_clock_id": checked.stream_id,
            "delivery_seq": checked.seq,
            "replayed": checked.replayed,
            "client_recv_mono_ns": received_ns,
            **checked.timeline.model_dump(mode="json"),
        }
        with self._work:
            if delivery_id in self._result_corrs_locked():
                _log(f"job delivery={delivery_id} deduplicated: durable result already exists")
                return delivery
            if delivery_id in self._inbox:
                if self._inbox[delivery_id] != job:
                    error = TransportUnhealthy(
                        f"job delivery_id {delivery_id!r} was reused with different content")
                    self._latch_locked(error)
                    raise error
                previous = self._delivery_meta.get(delivery_id)
                if previous is not None and previous["attempt_id"] != checked.attempt_id:
                    error = TransportUnhealthy(
                        f"job delivery_id {delivery_id!r} changed attempt_id before terminal")
                    self._latch_locked(error)
                    raise error
                # Same attempt on a new socket is an explicit replay, not a second unit of work. Retain the
                # newest receipt boundary so the ACK reports the delivery that is actually live now.
                self._delivery_meta[delivery_id] = delivery
                try:
                    self._persist_locked()
                except BaseException as e:
                    if previous is not None:
                        self._delivery_meta[delivery_id] = previous
                    error = TransportUnhealthy(f"durable replay receipt failed: {safe_error(e)}")
                    self._latch_locked(error, storage=True)
                    raise error from e
                return delivery
            # Go replays retained inflight jobs before ACKing our replayed terminal. Ambiguous delivery
            # closes claim admission, not the reader: accepting/deduping the replay keeps the socket alive.
            if (self._admission_error is not None
                    and not isinstance(self._admission_error, DeliveryPending)):
                raise self._admission_error
            self._inbox[delivery_id] = job
            self._delivery_meta[delivery_id] = delivery
            try:
                self._persist_locked()
            except BaseException as e:
                self._inbox.pop(delivery_id, None)
                self._delivery_meta.pop(delivery_id, None)
                error = TransportUnhealthy(f"durable job inbox append failed: {safe_error(e)}")
                self._latch_locked(error, storage=True)
                raise error from e
            self._queued_ids.add(delivery_id)
            self._jobs.put(delivery_id)
            self._work.notify_all()
        return delivery

    def _read_loop(self, conn: Any) -> None:
        while True:
            try:
                raw = conn.recv()
                client_recv_mono_ns = time.monotonic_ns()
                decoded = json.loads(raw)
                if not isinstance(decoded, dict):
                    raise ProtocolError("server frame is not an object")
                kind = decoded.get("type")
                if kind == "job":
                    delivery = self._accept_job(decoded, client_recv_mono_ns=client_recv_mono_ns)
                    if delivery is not None:
                        self._emit_job_ack(delivery)
                    continue
                if kind != "ack":
                    raise ProtocolError(f"unknown server frame type {kind!r}")
                with self._state:
                    if self._conn is not conn:
                        return
                ack = self._validate_ack(decoded)
                seq = int(ack["seq"])
                key = (str(ack["stream_id"]), seq)
                with self._work:
                    self._record_clock_sample_locked(key, ack, client_recv_mono_ns)
                    self._acks[key] = ack
                    waiter = self._ack_waiters.get(key)
                    status = int(ack["status"])
                    if status == _CONTENT_REJECT_STATUS:
                        # Ordered persistence may wait for an earlier ACK. Admission cannot: the peer already
                        # made a deterministic verdict, so no new paid job may enter during that gap.
                        frame = next(
                            (item for item in self._outbox if self._frame_key(item) == key), None)
                        if frame is None:
                            raise ProtocolError(f"ACK {key[0]}:{key[1]} has no durable frame")
                        self._latch_locked(FrameRejected(dict(frame), ack))
            except Exception as e:  # noqa: BLE001 - malformed peer means connection is unsafe
                _log(f"reader rejected server frame ({safe_error(e)})")
                self._fail_connection(conn, safe_error(e))
                return
            if waiter is not None:
                waiter.set()

    # ── connection and jobs ────────────────────────────────────────────────────────────────────────

    def _ensure_open(self) -> bool:
        if self._unavailable:
            return False
        with self._open_lock:
            with self._state:
                if self._conn is not None:
                    return True
            try:
                conn = _ws_client.connect(  # type: ignore[union-attr]
                    self._url,
                    additional_headers={"Authorization": f"Bearer {self._token}"},
                    open_timeout=OPEN_WALL_S,
                    close_timeout=2.0,
                    max_size=1 << 20,
                )
            except Exception as e:  # noqa: BLE001 - only lane unavailable
                _log(f"open failed endpoint={safe_endpoint(self._url)} ({safe_error(e)})")
                return False
            with self._state:
                self._conn = conn
            # The first connection may already have the same event in the
            # outbox (ControlPlane sends it normally). On reconnect append one
            # only when no identical durable frame is pending, preserving the
            # ordered stream while ensuring capacity is replayed.
            with self._work:
                bootstrap = dict(self._bootstrap_event) if self._bootstrap_event else None
                already_pending = bool(bootstrap and any(
                    frame.get("type") == "event" and frame.get("event") == bootstrap
                    for frame in self._outbox
                ))
            if bootstrap is not None and not already_pending:
                try:
                    seq = self._append_bootstrap_head(bootstrap)
                except Exception as e:  # noqa: BLE001 — no admission without durable bootstrap
                    why = f"bootstrap append: {safe_error(e)}"
                    self._fail_connection(conn, why)
                    _log(f"bootstrap append failed ({safe_error(e)}); admission remains closed")
                    return False
                with self._state:
                    self._pending_bootstrap_key = (self._stream_id, seq)
            self._reader = threading.Thread(
                target=self._read_loop, args=(conn,), name="pod-stream-reader", daemon=True)
            self._reader.start()
            _log(f"open ✓ endpoint={safe_endpoint(self._url)} stream={self._stream_id}")
            return True

    @staticmethod
    def _shutdown_conn(conn: Any) -> None:
        """Go straight to the socket: `Connection.close()` takes the same protocol mutex as send(), so a
        wedged writer would make an ordinary close() unbounded too — this is the one bounded exit both the
        failure path and a deliberate shutdown share."""
        raw_socket = getattr(conn, "socket", None)
        if raw_socket is not None:
            try:
                raw_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                raw_socket.close()
            except OSError:
                pass
        else:
            # Test doubles and older transports may expose no socket. Never let their close implementation
            # move the transport wall; it gets a daemon whose only job is to release the blocked writer.
            def close_connection() -> None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 - already dead
                    pass

            threading.Thread(
                target=close_connection, name="pod-stream-force-close", daemon=True).start()

    def _fail_connection(self, conn: Any, why: str) -> None:
        with self._state:
            if self._conn is conn:
                self._conn = None
                waiters = list(self._ack_waiters.values())
            else:
                # A reader from the previous socket may observe its close after a reconnect already installed
                # new ACK waiters under the same frame identities. It must not wake the new connection's window.
                waiters = []
        for waiter in waiters:
            waiter.set()
        self._shutdown_conn(conn)
        _log(f"socket closed ({safe_text(why)}); {self.pending_count()} durable frame(s) pending")

    def claim(self, wall: float) -> dict[str, Any] | None:
        """Return one strictly typed job, or None inside the caller's deadline."""
        with self._work:
            if self._admission_error is not None:
                raise self._admission_error
        if not self._ensure_open():
            with self._work:
                error = DeliveryPending("control-plane EventStream cannot open; refusing job admission")
                self._latch_locked(error)
            raise error
        deadline = time.monotonic() + wall
        while True:
            try:
                item = self._jobs.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return None
            with self._work:
                if item is _ADMISSION_WAKE:
                    if self._admission_error is not None:
                        raise self._admission_error
                    continue
                delivery_id = str(item)
                if self._admission_error is not None:
                    self._jobs.put(delivery_id)
                    raise self._admission_error
                job = self._inbox.get(delivery_id)
                if job is None:
                    continue
                self._queued_ids.discard(delivery_id)
                self._claimed_ids.add(delivery_id)
                return dict(job)

    def close(self) -> None:
        with self._work:
            self._stop = True
            conn = self._conn
            self._conn = None
            pending = len(self._outbox)
            self._work.notify_all()
        if conn is not None:
            self._shutdown_conn(conn)     # bounded — see _shutdown_conn's WHY; never conn.close() here
        self._sender.join(timeout=2.0)
        if pending:
            _log(f"closed with {pending} durable frame(s) pending at {self._path}; startup will replay them")


def make(base_url: str, job_token: str, *,
         outbox_path: str | Path | None = None) -> EventStream:
    return EventStream(base_url, job_token, outbox_path=outbox_path)
