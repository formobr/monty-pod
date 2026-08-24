"""NEGATIVE: an unmapped `-f null -` loudnorm measure decodes the VIDEO too — both measure argvs
must pin the exact `-map 0:a:0?` (not a loose `-map a`), and only the measure passes carry it."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from podagent import finalize, render

LN_JSON = ('{ "input_i" : "-19.4", "input_tp" : "-2.1", "input_lra" : "7.2", '
           '"input_thresh" : "-29.8", "target_offset" : "0.0" }')


def _has_seq(hay: list[str], needle: list[str]) -> bool:
    return any(hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))


def _audio_only_measure(cmd: list[str]) -> None:
    assert _has_seq(cmd, ["-map", "0:a:0?"]), f"measure pass decodes video too: {cmd}"
    assert cmd.count("-map") == 1
    assert cmd.index("-map") < cmd.index("-af"), "the map must bind before the audio filter"
    assert _has_seq(cmd, ["-f", "null", "-"])


def test_the_voice_measure_pass_maps_the_first_audio_stream_only(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake(cmd, **_kw):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr=LN_JSON)
    monkeypatch.setattr(render.subprocess, "run", fake)
    vln = render._measure_loudnorm(Path("/w/voice.mp4"), "highpass=f=80")
    assert len(calls) == 1
    _audio_only_measure(calls[0])
    assert "measured_I=-19.4" in vln and "linear=true" in vln


def test_the_master_measure_pass_maps_the_first_audio_stream_only(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake(cmd, **_kw):
        calls.append(list(cmd))
        if "null" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr=LN_JSON)
        (tmp_path / "out.mp4").write_bytes(b"v")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(finalize.subprocess, "run", fake)
    fin = SimpleNamespace(loudnorm=SimpleNamespace(i=-14.0, tp=-1.0, lra=11.0, attenuate_only=False))
    out = finalize.apply_loudnorm(fin, tmp_path / "master.mp4", tmp_path / "out.mp4")
    assert out == tmp_path / "out.mp4"
    measure, apply = calls
    _audio_only_measure(measure)
    # the APPLY pass still carries the video (`-c:v copy`) — the audio-only map is measure-only
    assert "-map" not in apply and _has_seq(apply, ["-c:v", "copy"])
