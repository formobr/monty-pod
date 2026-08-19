"""finalize.declared_grid / finalize.grid_verdict — the ONE shared grid check both render doors call
(render.render_spec, render_onepass.render_body). declared_grid is the only refusal this whole wave
allows (before any subprocess); grid_verdict runs AFTER an already-paid-for encode and never raises."""
from __future__ import annotations

import math
import subprocess
from fractions import Fraction

import pytest

from podagent import finalize


@pytest.mark.parametrize("fps", [0.0, -1.0, math.nan, math.inf, None])
def test_declared_grid_refuses_before_any_subprocess(monkeypatch, fps) -> None:
    monkeypatch.setattr(finalize.subprocess, "run", lambda *_a, **_kw: pytest.fail("subprocess ran"))
    with pytest.raises(ValueError):
        finalize.declared_grid(fps)


def test_declared_grid_emits_the_exact_rational_not_a_rounded_float() -> None:
    assert finalize.declared_grid(30) == "30"
    assert finalize.declared_grid(30000 / 1001) == "30000/1001"


@pytest.mark.parametrize("fps, expected", [
    (29.97, "30000/1001"), (23.976, "24000/1001"), (59.94, "60000/1001"),
    (30.0, "30"), (25.0, "25"),
])
def test_declared_grid_snaps_ntsc_decimals_to_the_broadcast_rational(fps, expected) -> None:
    """NEGATIVE (finding 5): plain limit_denominator(1001) round-trips 29.97 to 2997/100, a rational a
    real NTSC master (30000/1001) never measures as — the verdict was blind to the exact drift it exists
    to catch."""
    assert finalize.declared_grid(fps) == expected


@pytest.mark.parametrize("fps", [29.9701, 29.96995])
def test_declared_grid_does_not_snap_a_rate_merely_near_ntsc(fps) -> None:
    """NEGATIVE (HIGH): 29.9701/29.96995 are legitimately different measured rates, not float round-trips
    of 30000/1001 — snapping them would make grid_verdict compare a declared lie against the truth."""
    grid = finalize.declared_grid(fps)
    assert grid not in ("30000/1001", "24000/1001", "60000/1001")
    assert grid == str(Fraction(fps).limit_denominator(1001))


def test_grid_verdict_emits_nothing_on_a_clean_match(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 10.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: True)
    monkeypatch.setattr(finalize, "_probe_audio", lambda _p: (48000, 10.0))
    assert finalize.grid_verdict(tmp_path / "m.mp4", 30.0) is None


def test_grid_verdict_flags_a_video_rate_mismatch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 25.0, 25, 1, 10.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: False)
    defect = finalize.grid_verdict(tmp_path / "m.mp4", 30.0)
    assert defect == {"video_rate": {"declared": "30", "measured": "25"}}


def test_grid_verdict_flags_an_audio_rate_mismatch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 10.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: True)
    monkeypatch.setattr(finalize, "_probe_audio", lambda _p: (44100, 10.0))
    defect = finalize.grid_verdict(tmp_path / "m.mp4", 30.0)
    assert defect == {"audio_rate": {"declared": 48000, "measured": 44100}}


def test_grid_verdict_flags_an_av_duration_drift_past_tolerance(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 10.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: True)
    monkeypatch.setattr(finalize, "_probe_audio", lambda _p: (48000, 9.0))
    defect = finalize.grid_verdict(tmp_path / "m.mp4", 30.0)
    assert defect == {"av_duration_delta_ms": 1000.0}


def test_grid_verdict_tolerates_a_sub_frame_av_duration_drift(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 10.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: True)
    monkeypatch.setattr(finalize, "_probe_audio", lambda _p: (48000, 10.02))
    assert finalize.grid_verdict(tmp_path / "m.mp4", 30.0) is None


def test_grid_verdict_never_raises_on_a_probe_failure(monkeypatch, tmp_path) -> None:
    """NEGATIVE: the render is already paid for — a broken probe on the finished file must become a
    defect entry, never an exception that would cost it."""
    def boom(_p):
        raise RuntimeError("ffprobe exploded")
    monkeypatch.setattr(finalize, "_probe", boom)
    defect = finalize.grid_verdict(tmp_path / "m.mp4", 30.0)
    assert defect is not None and "probe_failed" in defect


def test_grid_verdict_never_raises_on_an_audio_probe_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 10.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: True)

    def boom(_p):
        raise RuntimeError("ffprobe exploded")
    monkeypatch.setattr(finalize, "_probe_audio", boom)
    defect = finalize.grid_verdict(tmp_path / "m.mp4", 30.0)
    assert defect is not None and "audio_probe_failed" in defect


def test_probe_carries_a_deadline(monkeypatch, tmp_path) -> None:
    """A hang here holds a ~20-GPU-minute master's delivery tail open forever; the probe must bound it."""
    timeouts = []

    def fake_run(cmd, **kw):
        timeouts.append(kw.get("timeout"))
        if "width,height,r_frame_rate" in cmd[cmd.index("-show_entries") + 1]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1080 1920 30/1", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="10.0", stderr="")
    monkeypatch.setattr(finalize.subprocess, "run", fake_run)
    finalize._probe(tmp_path / "master.mp4")
    assert timeouts and all(timeouts)


def test_probe_audio_carries_a_deadline(monkeypatch, tmp_path) -> None:
    timeouts = []

    def fake_run(cmd, **kw):
        timeouts.append(kw.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, stdout="48000 10.0", stderr="")
    monkeypatch.setattr(finalize.subprocess, "run", fake_run)
    finalize._probe_audio(tmp_path / "master.mp4")
    assert timeouts and all(timeouts)


def test_probe_timeout_is_the_same_refusal_as_any_other_probe_failure(monkeypatch, tmp_path) -> None:
    """_probe already just propagates on failure (check=True -> CalledProcessError); a timeout must
    propagate the same way, not hang — TimeoutExpired IS a raise, the existing refusal path."""
    def timeout(cmd, **_kw):
        raise subprocess.TimeoutExpired(cmd, 20)
    monkeypatch.setattr(finalize.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        finalize._probe(tmp_path / "hung.mp4")


def test_probe_audio_timeout_is_the_same_refusal_as_any_other_probe_failure(monkeypatch, tmp_path) -> None:
    def timeout(cmd, **_kw):
        raise subprocess.TimeoutExpired(cmd, 20)
    monkeypatch.setattr(finalize.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        finalize._probe_audio(tmp_path / "hung.mp4")


def test_grid_verdict_never_raises_when_probe_times_out(monkeypatch, tmp_path) -> None:
    """The critical case: grid_verdict runs after a paid-for encode and must NEVER raise — a video probe
    that hangs and finally times out must become a defect record, exactly like any other probe failure."""
    def timeout(_p):
        raise subprocess.TimeoutExpired(["ffprobe"], 20)
    monkeypatch.setattr(finalize, "_probe", timeout)
    defect = finalize.grid_verdict(tmp_path / "m.mp4", 30.0)
    assert defect is not None and "probe_failed" in defect


def test_grid_verdict_never_raises_when_audio_probe_times_out(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 10.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: True)

    def timeout(_p):
        raise subprocess.TimeoutExpired(["ffprobe"], 20)
    monkeypatch.setattr(finalize, "_probe_audio", timeout)
    defect = finalize.grid_verdict(tmp_path / "m.mp4", 30.0)
    assert defect is not None and "audio_probe_failed" in defect


def test_grid_verdict_never_raises_when_has_audio_itself_raises(monkeypatch, tmp_path) -> None:
    """NEGATIVE (finding 3): _has_audio carries no guard of its own — it is the whole-body wrap around
    grid_verdict, not a per-call try/except, that keeps the docstring's NEVER-raises promise."""
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 10.0))

    def boom(_p):
        raise RuntimeError("ffprobe exploded")
    monkeypatch.setattr(finalize, "_has_audio", boom)
    defect = finalize.grid_verdict(tmp_path / "m.mp4", 30.0)
    assert defect is not None and "probe_failed" in defect
