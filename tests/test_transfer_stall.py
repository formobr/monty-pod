"""artifact.download_verified — a transfer that crawls is not a transfer that is working.

MEASURED 2026-08-06 on a warm-up that never finished: the 3 GB weights pull ran at 101 MB/s for its first
minute, then decayed to 17 MB/s average — 9 MB moved in the last eight minutes. Nothing in this module
stopped it; the only thing that did was the caller's 600 s checkpoint wall, from outside, which cannot say
WHY and cannot take a different host.
"""
from __future__ import annotations

import time

import pytest

from podagent import artifact


class _Ref:
    def __init__(self, sha: str, size: int | None = None) -> None:
        self.url, self.sha256, self.size = "https://example/x.tar", sha, size


def _serve(chunks, monkeypatch, tick=None):
    """Feed `chunks` through the reader, optionally advancing a fake clock per chunk."""
    monkeypatch.setattr(artifact, "_chunks", lambda _url: iter(chunks))
    if tick is not None:
        t = {"now": 0.0}

        def _mono():
            return t["now"]
        monkeypatch.setattr(artifact.time, "monotonic", _mono)
        return t
    return None


def test_a_stalled_source_is_refused_by_name(tmp_path, monkeypatch):
    """THE DEFECT. A stream delivering kilobytes per second is still "making progress" and will hold a rented
    box until somebody else's deadline fires. Remove the floor and this hangs on instead of failing."""
    t = _serve([b"x" * 1024] * 6, monkeypatch, tick=True)
    orig = artifact.time.monotonic

    def _chunks_slow(_url):
        for _ in range(6):
            t["now"] += 30.0          # 1 KB per 30 s — three orders below any healthy link
            yield b"x" * 1024
    monkeypatch.setattr(artifact, "_chunks", _chunks_slow)
    with pytest.raises(artifact.TransferStalled, match="stalled"):
        artifact.download_verified(_Ref("0" * 64), tmp_path / "out.tar")
    assert orig is not None


def test_a_HEALTHY_transfer_is_not_refused(tmp_path, monkeypatch):
    """The floor must be far below any working link, or it turns a merely slow provider into an outage.
    ~1 MB per tick over the window is ordinary and must pass."""
    t = _serve([], monkeypatch, tick=True)
    payload = b"y" * (1024 * 1024)

    def _chunks_ok(_url):
        for _ in range(3):
            t["now"] += 1.0           # 1 MB/s — slow, but four times the floor
            yield payload
    monkeypatch.setattr(artifact, "_chunks", _chunks_ok)
    import hashlib
    sha = hashlib.sha256(payload * 3).hexdigest()
    assert artifact.download_verified(_Ref(sha), tmp_path / "ok.tar") == 3 * len(payload)


def test_the_rate_is_measured_over_a_WINDOW_not_from_the_start(tmp_path, monkeypatch):
    """THE REASON THE STALL HID FOR EIGHT MINUTES. An average from t0 is dragged down slowly by the very
    stall it should catch — the pull still read '17 MB/s' while moving nothing. A fast start followed by a
    dead window must fail on the window."""
    t = _serve([], monkeypatch, tick=True)
    fast = b"z" * (8 * 1024 * 1024)

    def _chunks_fast_then_dead(_url):
        t["now"] += 1.0
        yield fast                    # 8 MB in 1 s — a great average to hide behind
        t["now"] += 60.0
        yield b"z" * 512              # then 512 bytes in a minute
    monkeypatch.setattr(artifact, "_chunks", _chunks_fast_then_dead)
    with pytest.raises(artifact.TransferStalled):
        artifact.download_verified(_Ref("0" * 64), tmp_path / "dead.tar")


def test_the_floor_is_a_knob_because_the_right_number_belongs_to_the_link(monkeypatch):
    """Env-overridable on purpose: the correct floor is a property of the provider's network, not of this
    code, and a rented box's link is not the one this was measured on."""
    assert artifact._MIN_BYTES_PER_S > 0
    assert artifact._MIN_BYTES_PER_S < 5 * 1024 * 1024, \
        "a floor near a healthy rate turns a slow provider into an outage"


def test_progress_never_runs_on_the_transfer_thread(monkeypatch):
    """A REPORT MAY NEVER COST THE WORK IT IS REPORTING ON. `download_verified` calls its hook from inside
    the chunk loop, so when events moved to a socket that waits for an ack, a progress ping could hold the
    download for as long as the ack took. `cp.note` now hands the send to a thread; make it synchronous
    again and this reddens."""
    from podagent import cp as CP

    seen: list[str] = []
    started = time.monotonic()

    class _Slow:
        base = "http://x"

        def post_event(self, payload):
            time.sleep(0.4)           # stand-in for an ack that is not instant
            seen.append("sent")

        note = CP.ControlPlane.note

    _Slow().note({"stage": "ops", "status": "step", "step": "probe"})
    assert time.monotonic() - started < 0.2, "note() blocked its caller — the transfer would pay for it"
    time.sleep(0.8)
    assert seen == ["sent"], "the event must still actually go, just not on the caller's thread"
