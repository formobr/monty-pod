#!/usr/bin/env python3
"""event_stream.py — ONE socket for the reports the pod used to send as one POST per step.

WHY THIS EXISTS, MEASURED BEFORE IT WAS WRITTEN.

A 575-step run made 575+ separate HTTPS POSTs to `/pod/event`, each with its own 30 s read wall. Read out of
`monty:events` on 2026-08-06: 18 of those walls were hit and 12 connections were closed mid-request — 30 of
the run's failures, all of them on the leg the pod uses to say what it did. Everything downstream was
eliminated first: the Go handler's whole synchronous path is one XADD (plus one FIFO push on a terminal),
Redis SLOWLOG held a single 11.7 ms entry, and the box sat at load 0.07. The stall was never in the work.

THE POD DIALS OUT AND ONLY OUT, and that does not change: this socket is raised from inside the pod exactly
as the POSTs were. What changes is that one connection carries every report instead of one connection per
report — no per-step handshake, no per-step wall.

AND THE FRAME IS THE SAME OBJECT THE POST CARRIED, plus a `seq`. The control plane decodes both into the
same struct, so an old pod on the POST lane and a new pod on the socket are indistinguishable to it. That is
what makes this safe to roll out in the only order available: control plane first, image second.

THE OUTBOX IS THE POINT, not the speed. A report is retired only when the control plane ACKs its `seq`.
A socket that dies takes nothing with it — the unacked frames are re-sent on the next connection, in order.
That is the class of failure this replaces: a pod that finished its work, could not deliver the terminal,
and threw the work away.

FALLING BACK IS LOUD AND IS NOT A FAILURE. A control plane that does not know `/pod/stream` answers 404 to
the upgrade; the pod says so once and uses POST for the rest of its life. Silence here would make a stale
control plane look like a fast one.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, Callable

# The socket is optional at IMPORT time: an image built before the dependency landed must still run, on the
# POST lane, rather than failing to start. `_WS_IMPORT_ERROR` is kept so the reason can be said out loud once.
try:                                              # pragma: no cover - import shape, exercised by both paths
    import websockets.sync.client as _ws_client
    _WS_IMPORT_ERROR: str | None = None
except Exception as e:                            # noqa: BLE001 - any import failure means "no socket here"
    _ws_client = None                             # type: ignore[assignment]
    _WS_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# How long one frame may take to be written and acknowledged. FAR below the 30 s the POST lane spent per
# event: on a live socket an ack is a single round trip, so anything near this is a dead peer, not a slow one.
FRAME_WALL_S = float(os.environ.get("POD_STREAM_FRAME_WALL_S", "10"))
# The handshake gets its own, shorter, wall — a control plane that cannot be reached must not delay the first
# report by more than it would have cost to just POST it.
OPEN_WALL_S = float(os.environ.get("POD_STREAM_OPEN_WALL_S", "10"))
# Consecutive socket failures before the pod stops trying and stays on POST for good. Two, not one: a single
# dropped connection is ordinary, a second in a row means the lane is not there.
MAX_REOPENS = int(os.environ.get("POD_STREAM_MAX_REOPENS", "2"))

DISABLED = os.environ.get("POD_STREAM", "").strip() == "0"


def _log(msg: str) -> None:
    print(f"[pod-stream] {msg}", file=sys.stderr, flush=True)


class EventStream:
    """The socket lane, with an outbox. Thread-safe: the ops runner reports from a pool.

    `send` NEVER raises. It returns True when the control plane acknowledged the frame, and False when the
    caller must fall back to POST — which is what `cp.post_event` does with it.
    """

    def __init__(self, base_url: str, job_token: str) -> None:
        self._url = base_url.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1) \
            + "/pod/stream"
        self._token = job_token
        self._lock = threading.Lock()
        self._conn: Any = None
        self._seq = 0
        self._outbox: list[tuple[int, dict[str, Any]]] = []   # frames sent but not yet acknowledged
        self._reopens = 0
        self._dead = DISABLED or _ws_client is None
        if DISABLED:
            _log("POD_STREAM=0 — the socket lane is off by request; every event goes by POST")
        elif _ws_client is None:
            _log(f"websockets is not importable ({_WS_IMPORT_ERROR}) — this image reports by POST")

    # ── connection ────────────────────────────────────────────────────────────────────────────────────
    def _open(self) -> bool:
        """Raise the socket. Never raises, and — deliberately — sends NOTHING.

        An earlier version re-sent the outbox from here, which double-delivered every frame: `send` appends
        to the outbox and THEN connects, so the frame it was about to write was already in the batch this
        replayed. Connecting and flushing are two jobs and are now two methods; a test pins the duplicate.
        """
        try:
            self._conn = _ws_client.connect(                       # type: ignore[union-attr]
                self._url, additional_headers={"Authorization": f"Bearer {self._token}"},
                open_timeout=OPEN_WALL_S, close_timeout=2.0, max_size=1 << 20)
        except Exception as e:                    # noqa: BLE001 - a lane that will not open is not a crash
            self._reopens += 1
            _log(f"open failed ({type(e).__name__}: {e}) — attempt {self._reopens}/{MAX_REOPENS}")
            if self._reopens >= MAX_REOPENS:
                self._dead = True
                _log("giving up on the socket lane; every event goes by POST from here")
            return False
        self._reopens = 0                     # a lane that opened is not a lane that keeps failing
        _log(f"open ✓ {self._url}")
        return True

    def _flush(self) -> bool:
        """Write every unacknowledged frame, OLDEST FIRST, until the outbox is empty or the socket dies.

        THE ORDER IS LOAD-BEARING, not tidiness: the control plane folds a chain-level terminal into the FIFO
        the box is blocked on, so a terminal delivered ahead of the steps it terminates reads as a chain that
        did nothing. This is also THE RE-SEND — the frames a dead socket never got acknowledged are simply
        still here, which is the whole reason the outbox exists.
        """
        while self._outbox:
            seq, payload = self._outbox[0]
            if not self._write_and_ack(seq, payload):
                return False
        return True

    def _drop(self, why: str) -> None:
        _log(f"socket closed ({why}); {len(self._outbox)} frame(s) still unacknowledged")
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception as e:                    # noqa: BLE001 - closing a dead socket is not an error
            _log(f"close of an already-dead socket raised ({type(e).__name__}: {e}) — ignored")
        self._conn = None

    # ── one frame ─────────────────────────────────────────────────────────────────────────────────────
    def _write_and_ack(self, seq: int, payload: dict[str, Any]) -> bool:
        """Write one frame and wait for its ack. Returns False if the SOCKET failed (caller reopens).

        A frame the control plane REFUSES (a 4xx status in the ack) is not a socket failure: it is the same
        answer the POST lane gave, it is said out loud, and the frame is retired — re-sending a report the
        control plane has already judged malformed would loop forever.
        """
        try:
            self._conn.send(json.dumps({"seq": seq, **payload}))
            raw = self._conn.recv(timeout=FRAME_WALL_S)
        except Exception as e:                    # noqa: BLE001 - transport, not content
            # LOUD IN THE HANDLER ITSELF, not only inside `_drop`. The absorption gate reads this block and
            # cannot follow a call to decide whether anyone was told — and it is right to refuse: a returned
            # False whose announcement lives one frame away is one refactor from being silent.
            _log(f"frame seq={seq} lost the socket ({type(e).__name__}: {e}) — it stays in the outbox")
            self._drop(f"{type(e).__name__}: {e}")
            return False
        try:
            ack = json.loads(raw)
        except ValueError:
            self._drop("the control plane sent a frame that is not JSON")
            return False
        status = int(ack.get("status") or 0)
        if status != 202:
            _log(f"frame seq={seq} REFUSED status={status} reason={str(ack.get('error'))[:200]}")
        # RETIRED EITHER WAY. A 202 is delivery; a 4xx is the control plane's VERDICT on the content, and a
        # second copy earns the same verdict — re-sending it would loop until the pod died.
        self._outbox = [(s, p) for s, p in self._outbox if s != seq]
        return True

    def send(self, payload: dict[str, Any]) -> bool:
        """Deliver one event over the socket. False ⇒ the caller must POST it instead. Never raises."""
        if self._dead:
            return False
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._outbox.append((seq, payload))
            for _ in range(MAX_REOPENS + 1):
                if self._conn is None and not self._open():
                    if self._dead:
                        break
                    continue
                if self._flush():
                    return True
                if self._dead:
                    break
            # The frame stays in the outbox ONLY while the socket may still come back. Once we hand it to the
            # POST lane it is that lane's problem, and keeping a copy would deliver it twice on a reconnect.
            self._outbox = [(s, p) for s, p in self._outbox if s != seq]
            return False

    def close(self) -> None:
        with self._lock:
            if self._outbox:
                # NEVER SILENT: frames still held here were never acknowledged by anyone.
                _log(f"closing with {len(self._outbox)} frame(s) never acknowledged — "
                     f"seq {[s for s, _ in self._outbox]}")
            self._drop("pod shutting down")


def make(base_url: str, job_token: str, post: Callable[[dict[str, Any]], None]) -> Callable[
        [dict[str, Any]], None]:
    """An event sender that prefers the socket and falls back to `post`, announcing the fallback once."""
    stream = EventStream(base_url, job_token)
    said = {"fell_back": False}

    def send(payload: dict[str, Any]) -> None:
        if stream.send(payload):
            return
        if not said["fell_back"]:
            said["fell_back"] = True
            _log("falling back to POST /pod/event for this event and any that follow it")
        post(payload)

    send.stream = stream          # type: ignore[attr-defined]  # so a caller can close it explicitly
    return send
