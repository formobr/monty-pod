#!/usr/bin/env python3
"""The pod's single bidirectional control-plane channel.

Client frames are a closed union::

    {"type":"event",  "stream_id":str, "seq":int, "event":object}
    {"type":"result", "stream_id":str, "seq":int, "result":object}

Every frame is appended to a disk outbox before any socket write.  One sender
owns frame order.  A frame leaves the active outbox only after a matching 2xx
ACK; a deterministic 4xx moves it to the durable dead-letter.  Reconnects and
agent process restarts therefore cannot turn completed work into silence.

Server frames are also closed: typed ``ack`` and ``job`` objects.  Treating an
unknown object as an ACK is a protocol error, because retiring the wrong
terminal desynchronizes the ordered durable stream.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

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
DISABLED = os.environ.get("POD_STREAM", "").strip() == "0"
_DEFAULT_OUTBOX = "/var/cache/monty/pod-stream/outbox.json"
_STATE_VERSION = 1

_ACK_KEYS = frozenset({"type", "stream_id", "seq", "status", "duplicate", "error"})
_JOB_KEYS = frozenset({"type", "job"})


def _delivery_wall_s() -> float:
    """Caller wall for one bounded delivery cycle, derived from the sender's declared component walls."""
    attempts = MAX_REOPENS + 1
    return attempts * (OPEN_WALL_S + FRAME_WALL_S) + MAX_REOPENS * REOPEN_BACKOFF_S + 2.0


def _log(msg: str) -> None:
    print(f"[pod-stream] {msg}", file=sys.stderr, flush=True)


class ProtocolError(ValueError):
    """The peer sent a frame that cannot safely drive this state machine."""


class FrameRejected(RuntimeError):
    """The control plane made a deterministic 4xx verdict on one client frame."""

    def __init__(self, frame: dict[str, Any], ack: dict[str, Any]) -> None:
        self.frame, self.ack = frame, ack
        super().__init__(
            f"{frame['type']} frame {frame['stream_id']}:{frame['seq']} rejected "
            f"with {ack['status']}: {ack.get('error') or 'no reason'}")


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
        self._write_lock = threading.Lock()
        self._open_lock = threading.Lock()
        self._conn: Any = None
        self._reader: threading.Thread | None = None
        self._stop = False
        self._paused = False
        self._jobs: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._ack_waiters: dict[tuple[str, int], threading.Event] = {}
        self._acks: dict[tuple[str, int], dict[str, Any]] = {}
        self._delivery_waiters: dict[tuple[str, int], threading.Event] = {}
        self._delivery_outcomes: dict[tuple[str, int], bool | BaseException] = {}

        # A process incarnation owns one new stream id. Frames restored from disk keep THEIR original id/seq;
        # mixing an old identity into the new stream makes server-side dedupe unable to identify a replay.
        self._stream_id, self._next_seq = uuid.uuid4().hex, 1
        self._outbox, self._rejected = self._load()
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
        if not isinstance(frame.get("stream_id"), str) or not frame["stream_id"]:
            raise ProtocolError("outbox frame carries no stream_id")
        seq = frame.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ProtocolError(f"outbox seq must be a positive integer, got {seq!r}")
        if not isinstance(frame.get(kind), dict):
            raise ProtocolError(f"{kind} payload must be an object")
        return dict(frame)

    def _load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return [], []
        except (OSError, ValueError) as e:
            raise RuntimeError(f"durable pod outbox {self._path} is unreadable; refusing to discard it: {e}") from e
        if not isinstance(raw, dict) or set(raw) != {"version", "frames", "rejected"}:
            raise RuntimeError(f"durable pod outbox {self._path} has an unknown shape; refusing to discard it")
        if raw.get("version") != _STATE_VERSION:
            raise RuntimeError(f"durable pod outbox version {raw.get('version')!r} is unsupported")
        frames = raw.get("frames")
        rejected = raw.get("rejected")
        if not isinstance(frames, list) or not isinstance(rejected, list):
            raise RuntimeError("durable pod outbox frames is not a list")
        checked = [self._validate_client_frame(f) for f in frames]
        identities = [(str(f["stream_id"]), int(f["seq"])) for f in checked]
        if len(identities) != len(set(identities)):
            raise RuntimeError("durable pod outbox repeats a stream_id/seq identity")
        for row in rejected:
            if not isinstance(row, dict) or not isinstance(row.get("frame"), dict):
                raise RuntimeError("durable pod dead-letter carries a malformed row")
            self._validate_client_frame(row["frame"])
        return checked, list(rejected)

    def _persist_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": _STATE_VERSION,
            "frames": self._outbox,
            "rejected": self._rejected,
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

    def _append(self, kind: str, payload: dict[str, Any], *, wait: bool) -> tuple[int, threading.Event | None]:
        if kind == "result" and not (str(payload.get("corr_id") or "").strip()
                                     or str(payload.get("result_key") or "").strip()):
            raise ValueError("result frame requires corr_id or result_key")
        with self._work:
            seq = self._next_seq
            frame = {"type": kind, "stream_id": self._stream_id, "seq": seq, kind: dict(payload)}
            self._validate_client_frame(frame)
            self._outbox.append(frame)
            self._next_seq += 1
            self._persist_locked()  # append-before-send: the write below cannot race ahead of durability
            waiter = threading.Event() if wait else None
            if waiter is not None:
                self._delivery_waiters[(self._stream_id, seq)] = waiter
            self._paused = False
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
            with self._state:
                self._delivery_waiters.pop((stream_id, seq), None)
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
        while True:
            with self._work:
                self._work.wait_for(lambda: self._stop or (self._outbox and not self._paused))
                if self._stop:
                    return
                frame = dict(self._outbox[0])
            try:
                delivery_started = time.monotonic()
                disposition, ack = self._deliver(frame)
                self._settle(frame, disposition, ack, time.monotonic() - delivery_started)
            except BaseException as e:  # noqa: BLE001 - sender death would strand a synchronous terminal
                key = (str(frame["stream_id"]), int(frame["seq"]))
                _log(f"sender failed while settling {key[0]}:{key[1]} "
                     f"({type(e).__name__}: {e}); frame remains durable")
                with self._work:
                    self._paused = True
                    if (waiter := self._delivery_waiters.get(key)) is not None:
                        self._delivery_outcomes[key] = e
                        waiter.set()
                    self._work.notify_all()

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
                    except BaseException:
                        if acked is not None:
                            self._outbox.pop()
                            self._next_seq -= 1
                        self._outbox.insert(0, popped)
                        raise
                outcome = True
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
                    except BaseException:
                        self._rejected.pop()
                        self._outbox.insert(0, popped)
                        raise
                outcome = FrameRejected(frame, ack or {})
            else:
                # Retry exhaustion is not a verdict. Keep the frame durably at the head and pause until
                # a later send/claim/startup explicitly kicks the lane again.
                self._paused = True
                outcome = False
            if (waiter := self._delivery_waiters.get(key)) is not None:
                self._delivery_outcomes[key] = outcome
                waiter.set()
            self._work.notify_all()

    def _deliver(self, frame: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        for attempt in range(MAX_REOPENS + 1):
            if self._unavailable:
                break
            if not self._ensure_open():
                if attempt < MAX_REOPENS:
                    time.sleep(REOPEN_BACKOFF_S)
                continue
            ack = self._exchange(frame)
            if ack is None:
                if attempt < MAX_REOPENS:
                    time.sleep(REOPEN_BACKOFF_S)
                continue
            status = int(ack["status"])
            if 200 <= status < 300:
                return "accepted", ack
            if 400 <= status < 500:
                _log(f"frame seq={frame['seq']} REJECTED status={status} "
                     f"reason={str(ack.get('error') or '')[:200]} — moved to durable dead-letter")
                return "rejected", ack
            # A 5xx did not judge content. Retry like a timeout/socket ambiguity, retaining the same identity.
            if attempt < MAX_REOPENS:
                time.sleep(REOPEN_BACKOFF_S)
        _log(f"frame seq={frame['seq']} delivery retries exhausted — retained in durable outbox")
        return "pending", None

    def _exchange(self, frame: dict[str, Any]) -> dict[str, Any] | None:
        seq = int(frame["seq"])
        key = (str(frame["stream_id"]), seq)
        ready = threading.Event()
        with self._state:
            conn = self._conn
            self._ack_waiters[key] = ready
        if conn is None:
            with self._state:
                self._ack_waiters.pop(key, None)
            return None
        try:
            with self._write_lock:
                conn.send(json.dumps(frame, ensure_ascii=False, separators=(",", ":")))
        except Exception as e:  # noqa: BLE001 - transport, frame stays durable
            _log(f"frame seq={seq} write failed ({type(e).__name__}: {e})")
            with self._state:
                self._ack_waiters.pop(key, None)
            self._fail_connection(conn, f"write: {type(e).__name__}: {e}")
            return None
        got = ready.wait(FRAME_WALL_S)
        with self._state:
            self._ack_waiters.pop(key, None)
            ack = self._acks.pop(key, None)
        if not got or ack is None:
            _log(f"frame seq={seq} unacknowledged for {FRAME_WALL_S}s")
            self._fail_connection(conn, "ack timeout or invalid ACK")
            return None
        return ack

    # ── strict server frames ───────────────────────────────────────────────────────────────────────

    def _validate_ack(self, frame: dict[str, Any]) -> dict[str, Any]:
        if not set(frame) <= _ACK_KEYS or not {"type", "stream_id", "seq", "status"} <= set(frame):
            raise ProtocolError(f"ACK keys are invalid: {sorted(frame)}")
        if frame.get("type") != "ack":
            raise ProtocolError(f"expected type=ack, got {frame.get('type')!r}")
        seq, status = frame.get("seq"), frame.get("status")
        stream_id = frame.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            raise ProtocolError("ACK stream_id is empty")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ProtocolError("ACK seq is not a positive integer")
        if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
            raise ProtocolError("ACK status is not an HTTP status integer")
        if "duplicate" in frame and not isinstance(frame["duplicate"], bool):
            raise ProtocolError("ACK duplicate is not boolean")
        if "error" in frame and not isinstance(frame["error"], str):
            raise ProtocolError("ACK error is not a string")
        with self._state:
            if (stream_id, seq) not in self._ack_waiters:
                raise ProtocolError(f"ACK {stream_id}:{seq} has no in-flight frame")
        return frame

    @staticmethod
    def _validate_job(frame: dict[str, Any]) -> dict[str, Any]:
        if set(frame) != _JOB_KEYS or frame.get("type") != "job" or not isinstance(frame.get("job"), dict):
            raise ProtocolError("job frame must be exactly {type:'job', job:object}")
        return dict(frame["job"])

    def _read_loop(self, conn: Any) -> None:
        while True:
            try:
                raw = conn.recv()
                decoded = json.loads(raw)
                if not isinstance(decoded, dict):
                    raise ProtocolError("server frame is not an object")
                kind = decoded.get("type")
                if kind == "job":
                    self._jobs.put(self._validate_job(decoded))
                    continue
                if kind != "ack":
                    raise ProtocolError(f"unknown server frame type {kind!r}")
                ack = self._validate_ack(decoded)
            except Exception as e:  # noqa: BLE001 - malformed peer means connection is unsafe
                _log(f"reader rejected server frame ({type(e).__name__}: {e})")
                self._fail_connection(conn, f"{type(e).__name__}: {e}")
                return
            seq = int(ack["seq"])
            key = (str(ack["stream_id"]), seq)
            with self._state:
                self._acks[key] = ack
                waiter = self._ack_waiters.get(key)
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
                _log(f"open failed ({type(e).__name__}: {e})")
                return False
            with self._state:
                self._conn = conn
            self._reader = threading.Thread(
                target=self._read_loop, args=(conn,), name="pod-stream-reader", daemon=True)
            self._reader.start()
            _log(f"open ✓ {self._url} stream={self._stream_id}")
            return True

    def _fail_connection(self, conn: Any, why: str) -> None:
        with self._state:
            if self._conn is conn:
                self._conn = None
            waiters = list(self._ack_waiters.values())
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - already dead
            pass
        for waiter in waiters:
            waiter.set()
        _log(f"socket closed ({why}); {self.pending_count()} durable frame(s) pending")

    def claim(self, wall: float) -> dict[str, Any] | None:
        """Return one strictly typed job, or None inside the caller's deadline."""
        with self._work:
            self._paused = False
            self._work.notify_all()
        if not self._ensure_open():
            return None
        try:
            return self._jobs.get(timeout=wall)
        except queue.Empty:
            return None

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
