#!/usr/bin/env python3
"""The pod's single bidirectional control-plane channel.

Client frames are a closed union::

    {"type":"event",  "stream_id":str, "seq":int, "event":object}
    {"type":"result", "stream_id":str, "seq":int, "result":object}

Every frame is appended to a disk outbox before any socket write.  One sender
owns frame order and may have a bounded window of writes awaiting ACKs.  A
frame leaves the active outbox, in order, only after a matching 2xx ACK; a
deterministic 4xx moves it to the durable dead-letter.  Reconnects replay only
the unacknowledged suffix with the original identities, and process restarts
cannot turn completed work into silence.

Server frames are also closed: typed ``ack`` and ``job`` objects.  Treating an
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
from .stream_models import StreamAck, StreamJob, event_payload, result_payload

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
DISABLED = os.environ.get("POD_STREAM", "").strip() == "0"
_DEFAULT_OUTBOX = "/var/cache/monty/pod-stream/outbox.json"
_STATE_VERSION = 2
_ADMISSION_WAKE = object()


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
        self._delivery_waiters: dict[tuple[str, int], threading.Event] = {}
        self._delivery_outcomes: dict[tuple[str, int], bool | BaseException] = {}
        self._admission_error: TransportUnhealthy | None = None
        self._storage_error: TransportUnhealthy | None = None

        # A process incarnation owns one new stream id. Frames restored from disk keep THEIR original id/seq;
        # mixing an old identity into the new stream makes server-side dedupe unable to identify a replay.
        self._stream_id, self._next_seq = uuid.uuid4().hex, 1
        self._outbox, self._rejected, self._inbox = self._load()
        if self._rejected:
            self._admission_error = TransportUnhealthy(
                "a deterministic rejected verdict remains in durable dead-letter")
        elif self._outbox:
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
    def _validate_client_frame(frame: Any) -> dict[str, Any]:
        if not isinstance(frame, dict):
            raise ProtocolError("outbox frame is not an object")
        kind = frame.get("type")
        if kind not in ("event", "result"):
            raise ProtocolError(f"outbox frame type must be event|result, got {kind!r}")
        keys = {"type", "stream_id", "seq", kind}
        if set(frame) != keys:
            raise ProtocolError(f"{kind} frame keys must be {sorted(keys)}, got {sorted(frame)}")
        if (not isinstance(frame.get("stream_id"), str) or not frame["stream_id"]
                or len(frame["stream_id"]) > 128
                or re.fullmatch(r"[A-Za-z0-9_.-]+", frame["stream_id"]) is None):
            raise ProtocolError("outbox frame carries no stream_id")
        seq = frame.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ProtocolError(f"outbox seq must be a positive integer, got {seq!r}")
        checked = dict(frame)
        checked[kind] = event_payload(frame.get(kind)) if kind == "event" else result_payload(frame.get(kind))
        return checked

    def _load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return [], [], {}
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"durable pod state {self._path} is unreadable; refusing to discard it: {safe_error(e)}") from e
        if not isinstance(raw, dict) or set(raw) != {"version", "frames", "rejected", "inbox"}:
            raise RuntimeError(f"durable pod outbox {self._path} has an unknown shape; refusing to discard it")
        if raw.get("version") != _STATE_VERSION:
            raise RuntimeError(f"durable pod outbox version {raw.get('version')!r} is unsupported")
        frames = raw.get("frames")
        rejected = raw.get("rejected")
        inbox = raw.get("inbox")
        if not isinstance(frames, list) or not isinstance(rejected, list) or not isinstance(inbox, dict):
            raise RuntimeError("durable pod state frames/rejected/inbox have invalid container types")
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
            decoded = StreamJob.model_validate(
                {"type": "job", "delivery_id": delivery_id, "job": job})
            checked_inbox[delivery_id] = decoded.job.model_dump(exclude_none=True, mode="json")
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
        return checked, list(rejected), checked_inbox

    def _persist_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": _STATE_VERSION,
            "frames": self._outbox,
            "rejected": self._rejected,
            "inbox": self._inbox,
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

    def _append(self, kind: str, payload: dict[str, Any], *, wait: bool) -> tuple[int, threading.Event | None]:
        try:
            normalized = event_payload(payload) if kind == "event" else result_payload(payload)
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
                error = TransportUnhealthy(f"result corr_id {corr_id!r} already pending or rejected")
                self._latch_locked(error)
                raise error
            seq = self._next_seq
            frame = {"type": kind, "stream_id": self._stream_id, "seq": seq, kind: normalized}
            self._validate_client_frame(frame)
            self._outbox.append(frame)
            self._next_seq += 1
            waiter = threading.Event() if wait else None
            if waiter is not None:
                self._delivery_waiters[(self._stream_id, seq)] = waiter
            removed_job = self._inbox.pop(corr_id, None) if kind == "result" else None
            try:
                self._persist_locked()  # append and matching inbox retirement are ONE durable transition
            except BaseException as e:
                if removed_job is not None:
                    self._inbox[corr_id] = removed_job
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

    def pending_count(self) -> int:
        with self._state:
            return len(self._outbox)

    # ── public client frames ───────────────────────────────────────────────────────────────────────

    def send_event(self, payload: dict[str, Any], *, wait: bool = False) -> bool:
        """Persist one event, then optionally wait until its strict ACK."""
        seq, waiter = self._append("event", payload, wait=wait)
        return True if waiter is None else self._await_delivery(self._stream_id, seq, waiter)

    def send_result(self, payload: dict[str, Any], *, wait: bool = True) -> bool:
        """Persist one correlated result, then optionally wait until its strict ACK."""
        seq, waiter = self._append("result", payload, wait=wait)
        return True if waiter is None else self._await_delivery(self._stream_id, seq, waiter)

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
                    # Exhaustion is ambiguous, not a verdict. It wakes a synchronous caller and closes job
                    # admission, while the frame and any ACKed successors remain ordered and durable.
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
                if (isinstance(self._admission_error, DeliveryPending)
                        and not self._outbox and self._storage_error is None):
                    self._admission_error = None
            elif disposition == "rejected":
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
                self._latch_locked(outcome)
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
            acks = self._exchange_window(pending)
            retry: list[dict[str, Any]] = []
            for frame in pending:
                key = self._frame_key(frame)
                ack = acks.get(key)
                status = int(ack["status"]) if ack is not None else 0
                delivery_s = time.monotonic() - started[key]
                if 200 <= status < 300:
                    outcomes[key] = ("accepted", ack, delivery_s)
                elif 400 <= status < 500:
                    _log(f"frame seq={frame['seq']} REJECTED status={status} "
                         f"reason={safe_text(ack.get('error') or '', 200)} "
                         "— moved to durable dead-letter")
                    outcomes[key] = ("rejected", ack, delivery_s)
                else:
                    # A 5xx or absent ACK made no content verdict. Keep the exact identity for replay.
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
                    conn.send(json.dumps(frame, ensure_ascii=False, separators=(",", ":")))
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

    def _accept_job(self, frame: dict[str, Any]) -> None:
        try:
            checked = StreamJob.model_validate(frame)
        except BaseException as e:
            error = TransportUnhealthy(f"invalid server job frame: {safe_error(e)}")
            with self._work:
                self._latch_locked(error)
            raise error from e
        delivery_id = checked.delivery_id
        job = checked.job.model_dump(exclude_none=True, mode="json")
        with self._work:
            if delivery_id in self._result_corrs_locked():
                _log(f"job delivery={delivery_id} deduplicated: durable result already exists")
                return
            if delivery_id in self._inbox:
                if self._inbox[delivery_id] != job:
                    error = TransportUnhealthy(
                        f"job delivery_id {delivery_id!r} was reused with different content")
                    self._latch_locked(error)
                    raise error
                return
            # Go replays retained inflight jobs before ACKing our replayed terminal. Ambiguous delivery
            # closes claim admission, not the reader: accepting/deduping the replay keeps the socket alive.
            if (self._admission_error is not None
                    and not isinstance(self._admission_error, DeliveryPending)):
                raise self._admission_error
            self._inbox[delivery_id] = job
            try:
                self._persist_locked()
            except BaseException as e:
                self._inbox.pop(delivery_id, None)
                error = TransportUnhealthy(f"durable job inbox append failed: {safe_error(e)}")
                self._latch_locked(error, storage=True)
                raise error from e
            self._queued_ids.add(delivery_id)
            self._jobs.put(delivery_id)
            self._work.notify_all()

    def _read_loop(self, conn: Any) -> None:
        while True:
            try:
                raw = conn.recv()
                decoded = json.loads(raw)
                if not isinstance(decoded, dict):
                    raise ProtocolError("server frame is not an object")
                kind = decoded.get("type")
                if kind == "job":
                    self._accept_job(decoded)
                    continue
                if kind != "ack":
                    raise ProtocolError(f"unknown server frame type {kind!r}")
                ack = self._validate_ack(decoded)
                seq = int(ack["seq"])
                key = (str(ack["stream_id"]), seq)
                with self._work:
                    self._acks[key] = ack
                    waiter = self._ack_waiters.get(key)
                    status = int(ack["status"])
                    if 400 <= status < 500:
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
            self._reader = threading.Thread(
                target=self._read_loop, args=(conn,), name="pod-stream-reader", daemon=True)
            self._reader.start()
            _log(f"open ✓ endpoint={safe_endpoint(self._url)} stream={self._stream_id}")
            return True

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
        raw_socket = getattr(conn, "socket", None)
        if raw_socket is not None:
            # `Connection.close()` takes the same protocol mutex as send(). A wedged send owns that mutex, so
            # only shutting down the underlying socket can guarantee that this failure path itself is bounded.
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
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - shutdown is best effort; outbox is already durable
                pass
        self._sender.join(timeout=2.0)
        if pending:
            _log(f"closed with {pending} durable frame(s) pending at {self._path}; startup will replay them")


def make(base_url: str, job_token: str, *,
         outbox_path: str | Path | None = None) -> EventStream:
    return EventStream(base_url, job_token, outbox_path=outbox_path)
