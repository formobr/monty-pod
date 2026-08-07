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

ONE READER, TWO WRITERS, NO SHARED LOCK — and the first version got this wrong at a measured cost. It used a
single mutex for everything, so `claim` (which waits up to 30 s for the next envelope) held the socket while
it waited, and every step report the pod wanted to send queued behind it. The ledger read
`!transport 535.2 s` over 24 steps against `1.6 s` over 17 the day before, and the box then blamed the CDN:
nine preview batches all hit their 45 s wall while the pod had actually pulled 99.6 MB from pexels in 30.9 s.
A LANE THAT BLOCKS ITS OWN REPORTS IS WORSE THAN THE POSTs IT REPLACED.

So: a dedicated reader thread owns `recv` and nothing else. It routes each frame — an ack wakes the exact
sender waiting on that `seq`, a job goes on a queue. Senders hold a WRITE lock only long enough to write.
`claim` holds nothing at all; it waits on the queue.

THE OUTBOX IS THE POINT, not the speed. A report is retired only when the control plane ACKs its `seq`.
A socket that dies takes nothing with it — the unacked frames are re-sent on the next connection, in order.
That is the class of failure this replaces: a pod that finished its work, could not deliver the terminal,
and threw the work away.

THERE IS NO FALLBACK, AND THAT IS DELIBERATE. An earlier version kept POST as a second lane. Two mechanisms
for one job is how they drift — and the poll that briefly survived beside the socket promptly showed up in
the control plane's access log as 32 fifty-second requests nobody wanted. A socket that will not open is a
FAILURE, said out loud, not a quiet degrade into the shape we were leaving.

AND THE SAME SOCKET CARRIES THE WORK. `GET /pod/job` is deleted: the control plane claims from the session
lane and pushes `{"type":"job"}` frames down this connection. The at-most-once exposure did not change with
the move — the poll ALREADY took the envelope off the queue with a BRPOP before writing its response.
"""
from __future__ import annotations

import json
import os
import queue
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

# POD_STREAM=0 USED TO MEAN "fall back to POST". There is no POST lane any more, so it now means "this pod
# reports and claims NOTHING" — which is only ever useful to a test that wants the dead-lane path without a
# server. It is deliberately NOT an operational revert: reverting a transport by silencing the pod would
# hide the very failures the transport exists to deliver.
DISABLED = os.environ.get("POD_STREAM", "").strip() == "0"


# Returned by `_recv` when the wall expired with nothing to read. A distinct object rather than None so a
# QUIET wire can never be mistaken for a DEAD one — that mistake cost a re-connect every 30 seconds.
_QUIET: dict[str, Any] = {}


def _log(msg: str) -> None:
    print(f"[pod-stream] {msg}", file=sys.stderr, flush=True)


class EventStream:
    """The socket lane: one reader thread, senders that wait on their own ack, and an outbox.

    `send` NEVER raises. It returns True when the control plane acknowledged the frame, False when it could
    not be delivered — and with no POST lane behind it, False is a report that is GONE, which `cp.post_event`
    says out loud.
    """

    def __init__(self, base_url: str, job_token: str) -> None:
        self._url = base_url.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1) \
            + "/pod/stream"
        self._token = job_token
        self._wlock = threading.Lock()        # serialises WRITES only; a reader never takes it
        self._resend = threading.Lock()       # one re-sender at a time, so a reconnect cannot duplicate
        self._state = threading.Lock()        # guards the small maps below
        self._conn: Any = None
        self._reader: threading.Thread | None = None
        self._seq = 0
        self._outbox: list[tuple[int, dict[str, Any]]] = []
        self._waiters: dict[int, threading.Event] = {}
        # Frames a sender has taken responsibility for. Claimed the moment `send` mints the seq, BEFORE any
        # re-send can look — registering it later left a window where another thread saw the frame as
        # orphaned and delivered it a second time.
        self._mine: set[int] = set()
        self._acks: dict[int, dict[str, Any]] = {}
        self._jobs: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._reopens = 0
        self._dead = DISABLED or _ws_client is None
        if DISABLED:
            _log("POD_STREAM=0 — this pod will neither report nor claim. There is no second lane.")
        elif _ws_client is None:
            _log(f"websockets is not importable ({_WS_IMPORT_ERROR}) — this pod CANNOT reach the control "
                 f"plane at all, because the socket is the only lane. Rebuild the image.")

    # ── the reader ────────────────────────────────────────────────────────────────────────────────────
    def _read_loop(self, conn: Any) -> None:
        """Own `recv` for this connection and route what arrives. The ONLY thread that reads."""
        while True:
            try:
                raw = conn.recv()
            except Exception as e:            # noqa: BLE001 — the connection ended; wake everyone waiting
                self._fail_connection(conn, f"{type(e).__name__}: {e}")
                return
            try:
                frame = dict(json.loads(raw))
            except (ValueError, TypeError):
                _log("the control plane sent a frame that is not a JSON object — dropping this socket")
                self._fail_connection(conn, "undecodable frame")
                return
            if str(frame.get("type")) == "job":
                self._jobs.put(frame)
                continue
            seq = int(frame.get("seq") or 0)
            with self._state:
                self._acks[seq] = frame
                ev = self._waiters.get(seq)
            if ev is not None:
                ev.set()

    def _fail_connection(self, conn: Any, why: str) -> None:
        """Tear down `conn` and wake every sender blocked on it, so nobody waits on a socket that is gone."""
        with self._state:
            if self._conn is conn:
                self._conn = None
            waiters = list(self._waiters.values())
        _log(f"socket closed ({why}); {len(self._outbox)} frame(s) still unacknowledged")
        try:
            conn.close()
        except Exception as e:                # noqa: BLE001 — closing a dead socket is not an error
            _log(f"close of an already-dead socket raised ({type(e).__name__}: {e}) — ignored")
        for ev in waiters:
            ev.set()

    # ── connection ────────────────────────────────────────────────────────────────────────────────────
    def _open(self) -> bool:
        """Raise the socket and start its reader. Never raises, and sends NOTHING (flushing is `_flush`)."""
        try:
            conn = _ws_client.connect(                                  # type: ignore[union-attr]
                self._url, additional_headers={"Authorization": f"Bearer {self._token}"},
                open_timeout=OPEN_WALL_S, close_timeout=2.0, max_size=1 << 20)
        except Exception as e:                # noqa: BLE001 - a lane that will not open is not a crash
            self._reopens += 1
            _log(f"open failed ({type(e).__name__}: {e}) — attempt {self._reopens}/{MAX_REOPENS}")
            if self._reopens >= MAX_REOPENS:
                self._dead = True
                _log("giving up on the socket lane; this pod can no longer report or claim")
            return False
        self._reopens = 0
        with self._state:
            self._conn = conn
        self._reader = threading.Thread(target=self._read_loop, args=(conn,),
                                        name="pod-stream-reader", daemon=True)
        self._reader.start()
        _log(f"open ✓ {self._url}")
        return True

    # ── one frame ─────────────────────────────────────────────────────────────────────────────────────
    def _write_and_ack(self, seq: int, payload: dict[str, Any]) -> bool:
        """Write one frame and wait for ITS ack. False ⇒ the SOCKET failed (the caller reopens)."""
        ev = threading.Event()
        with self._state:
            conn = self._conn
            self._waiters[seq] = ev
        if conn is None:
            with self._state:
                self._waiters.pop(seq, None)
            return False
        try:
            with self._wlock:                 # WRITES only — the reader is never behind this
                conn.send(json.dumps({"seq": seq, **payload}))
        except Exception as e:                # noqa: BLE001 - transport, not content
            _log(f"frame seq={seq} lost the socket on write ({type(e).__name__}: {e}) — it stays in the outbox")
            with self._state:
                self._waiters.pop(seq, None)
            self._fail_connection(conn, f"{type(e).__name__}: {e}")
            return False
        got = ev.wait(FRAME_WALL_S)
        with self._state:
            self._waiters.pop(seq, None)
            ack = self._acks.pop(seq, None)
        if not got or ack is None:
            # The control plane answers every frame, so silence inside the wall means the peer is gone even
            # while the socket still looks open.
            _log(f"frame seq={seq} went unacknowledged for {FRAME_WALL_S}s — dropping this socket")
            if conn is not None:
                self._fail_connection(conn, "ack timeout")
            return False
        status = int(ack.get("status") or 0)
        if status != 202:
            _log(f"frame seq={seq} REFUSED status={status} reason={str(ack.get('error'))[:200]}")
        # RETIRED EITHER WAY: a 202 is delivery, a 4xx is a verdict on the content and a second copy earns
        # the same verdict forever.
        with self._state:
            self._outbox = [(s, p) for s, p in self._outbox if s != seq]
            self._mine.discard(seq)
        return True

    def _resend_outbox(self, upto_seq: int) -> bool:
        """Re-send frames a DEAD socket never got acknowledged, oldest first, on the fresh one.

        ONLY THE FRAMES OLDER THAN THE CALLER'S OWN, and only one thread at a time. An earlier version had
        every sender flush the whole outbox from the front, so eight concurrent reports each wrote frame #1
        and the control plane received sixteen frames for eight events — a test caught it. The order still
        matters: a chain terminal delivered ahead of its own steps reads as a chain that did nothing.
        """
        with self._resend:
            while True:
                with self._state:
                    # IN FLIGHT IS NOT UNSENT. A frame stays in the outbox until its ack lands, so its own
                    # sender is still waiting on it — re-sending it here delivered the same event twice
                    # (measured: eight concurrent reports became nineteen frames). Only frames nobody is
                    # waiting for are genuinely orphaned by a dead socket.
                    older = [(s, pl) for s, pl in self._outbox
                             if s < upto_seq and s not in self._mine]
                if not older:
                    return True
                seq, payload = older[0]
                if not self._write_and_ack(seq, payload):
                    return False

    def send(self, payload: dict[str, Any]) -> bool:
        """Deliver one event. NEVER raises. False ⇒ the report is GONE — there is no second lane."""
        if self._dead:
            return False
        with self._state:
            self._seq += 1
            seq = self._seq
            self._outbox.append((seq, payload))
            self._mine.add(seq)
        for _ in range(MAX_REOPENS + 1):
            if self._conn is None and not self._open():
                if self._dead:
                    break
                continue
            # Anything the previous socket never got acknowledged goes first, then THIS frame — never the
            # whole outbox per sender, which is how eight reports became sixteen frames.
            if self._resend_outbox(seq) and self._write_and_ack(seq, payload):
                return True
            if self._dead:
                break
        with self._state:
            self._outbox = [(s, p) for s, p in self._outbox if s != seq]
            self._mine.discard(seq)
        return False

    def claim(self, wall: float) -> dict[str, Any] | None:
        """One envelope of work, or None if none arrived inside `wall`. NEVER raises, and HOLDS NO LOCK —
        that is the whole point: a claim that blocked the socket cost 535 s of `!transport` in one run."""
        if self._dead:
            return None
        if self._conn is None and not self._open():
            return None
        try:
            frame = self._jobs.get(timeout=wall)
        except queue.Empty:
            return None                        # no work this window; the socket is untouched
        return frame.get("job")

    def close(self) -> None:
        with self._state:
            held = list(self._outbox)
            conn = self._conn
            self._conn = None
        if held:
            _log(f"closing with {len(held)} frame(s) never acknowledged — seq {[s for s, _ in held]}")
        if conn is not None:
            try:
                conn.close()
            except Exception as e:            # noqa: BLE001
                _log(f"close raised ({type(e).__name__}: {e}) — ignored")


def make(base_url: str, job_token: str) -> EventStream:
    """The lane for one job. There is no fallback to wrap any more — the socket is it."""
    return EventStream(base_url, job_token)
