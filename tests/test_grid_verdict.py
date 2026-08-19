"""finalize.declared_grid / finalize.grid_verdict — the ONE shared grid check both render doors call
(render.render_spec, render_onepass.render_body). declared_grid is the only refusal this whole wave
allows (before any subprocess); grid_verdict runs AFTER an already-paid-for encode and never raises."""
from __future__ import annotations

import math

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
