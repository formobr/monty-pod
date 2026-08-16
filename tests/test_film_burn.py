from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from podagent import accents, finalize, render
from podagent.models import RenderSpec


def _fin(*items):
    return SimpleNamespace(accents=list(items))


def _burn(at=1.75, burn="burn.mp4", intensity=0.7):
    return SimpleNamespace(kind="film_burn", at=at, intensity=intensity,
                           burn=burn, clicks="clicks.wav")


@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "gpu"])
def test_film_burn_command_uses_script_and_copies_audio(monkeypatch, tmp_path, gpu):
    commands = []
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 60000 / 1001, 60000, 1001, 10.0))
    monkeypatch.setattr(accents, "detect_flares", lambda _p: [0.3])
    monkeypatch.setattr(finalize, "_run", lambda cmd, _what, **_kw: commands.append(cmd))

    src = tmp_path / "master.mp4"
    out = tmp_path / "accents.mp4"
    burn = tmp_path / "burn.mp4"
    finalize.apply_accents(_fin(_burn()), src, out, {"burn.mp4": burn}, gpu)

    cmd = commands[0]
    assert cmd.count("-i") == 2 and str(burn) in cmd
    burn_i = cmd.index(str(burn))
    assert cmd[burn_i - 3:burn_i] == ["-stream_loop", "-1", "-i"]
    assert "-filter_complex_script" in cmd
    graph = Path(cmd[cmd.index("-filter_complex_script") + 1]).read_text()
    assert finalize._BT709_SET_PARAMS in graph
    assert graph.index("[jstk]crop") < graph.index("[1:v]scale")
    assert "aa=0.7" in graph
    assert "-af" not in cmd and "clicks" not in graph
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert ("-init_hw_device" in cmd) is gpu
    if gpu:
        idx = cmd.index("-init_hw_device")
        assert cmd[idx:idx + 2] == ["-init_hw_device", "vulkan"]
    profile = finalize._MID_GPU if gpu else finalize._MID_CPU
    for token in profile:
        assert token in cmd
    assert "-color_primaries" in cmd
    assert cmd[cmd.index("-color_primaries") + 1] == "bt709"
    for token in finalize._BT709:
        assert token in cmd


def test_film_burn_never_enters_single_input_builders(monkeypatch, tmp_path):
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 60000 / 1001, 60000, 1001, 10.0))
    monkeypatch.setattr(accents, "detect_flares", lambda _p: [0.3])
    monkeypatch.setattr(finalize, "_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(accents, "build_chain_filter", lambda *_a, **_k: pytest.fail("film_burn reached BUILDERS"))
    finalize.apply_accents(_fin(_burn()), tmp_path / "m.mp4", tmp_path / "o.mp4",
                           {"burn.mp4": tmp_path / "b.mp4"}, False)


@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "gpu"])
def test_finalize_video_reencodes_are_bt709_tagged(monkeypatch, tmp_path, gpu):
    commands = []
    monkeypatch.setattr(finalize, "_run", lambda cmd, *_a, **_kw: commands.append(cmd))
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 10.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: False)
    monkeypatch.setattr(finalize._accents, "build_chain_filter", lambda *_a, **_kw: "[0:v]null[vout]")
    monkeypatch.setattr(finalize, "watermark_filter", lambda **_kw: ("[0:v]null[v]", "v", None))

    finalize.apply_accents(_fin(SimpleNamespace(kind="pixelate")), tmp_path / "m.mp4",
                           tmp_path / "a.mp4", {}, gpu)
    finalize.apply_logo(SimpleNamespace(logo=SimpleNamespace(
        asset="logo.png", corner="tr", width=100, opacity=0.5, margin=10, cover_hold=0.6,
    )), tmp_path / "a.mp4", tmp_path / "l.mp4", {"logo.png": tmp_path / "logo.png"}, gpu)
    finalize.apply_watermark(SimpleNamespace(watermark=SimpleNamespace(
        sting="sting.webm", idle="idle.webm", width=100, x=None, y=None,
        position="bottom-right", margin=10, chime=False, chime_volume=1.0, delay=0.0,
    )), tmp_path / "l.mp4", tmp_path / "w.mp4",
    {"sting.webm": tmp_path / "sting.webm", "idle.webm": tmp_path / "idle.webm"}, gpu)

    assert len(commands) == 3
    assert all(all(token in cmd for token in finalize._BT709) for cmd in commands)
    assert all(finalize._BT709_SET_PARAMS in cmd[cmd.index("-filter_complex") + 1]
               for cmd in commands)


def test_different_burn_ids_refuse_before_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(finalize, "_probe", lambda *_a: pytest.fail("probe ran before refusal"))
    with pytest.raises(RuntimeError, match="share one burn input id"):
        finalize.apply_accents(
            _fin(_burn(1.0, "a.mp4"), _burn(2.0, "b.mp4")),
            tmp_path / "m.mp4", tmp_path / "o.mp4",
            {"a.mp4": tmp_path / "a", "b.mp4": tmp_path / "b"}, False,
        )


def test_different_burn_intensities_refuse_before_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(finalize, "_probe", lambda *_a: pytest.fail("probe ran before refusal"))
    with pytest.raises(RuntimeError, match="share one intensity"):
        finalize.apply_accents(
            _fin(_burn(1.0, intensity=0.4), _burn(2.0, intensity=0.8)),
            tmp_path / "m.mp4", tmp_path / "o.mp4", {"burn.mp4": tmp_path / "b"}, False,
        )


def test_film_burn_boundary_cap_refuses_fanout():
    with pytest.raises(RuntimeError, match=r"boundary count 9.*RARE accent, 2-3/video"):
        accents.add_filmburn([], "[v]", 1, list(range(9)), [0.1])


def test_film_burn_boundary_cap_refuses_before_any_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(finalize, "_probe", lambda *_a: pytest.fail("probe ran before refusal"))
    monkeypatch.setattr(accents, "detect_flares", lambda *_a: pytest.fail("burn was probed before refusal"))
    with pytest.raises(RuntimeError, match=r"boundary count 9.*RARE accent, 2-3/video"):
        finalize.apply_accents(
            _fin(*(_burn(float(i)) for i in range(9))),
            tmp_path / "m.mp4", tmp_path / "o.mp4", {"burn.mp4": tmp_path / "b"}, False,
        )


def test_film_burn_render_uses_named_timeout(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 10.0))
    monkeypatch.setattr(accents, "detect_flares", lambda _p: [0.3])
    monkeypatch.setattr(finalize, "_run", lambda _cmd, _what, **kw: seen.update(kw))
    finalize.apply_accents(_fin(_burn()), tmp_path / "m.mp4", tmp_path / "o.mp4",
                           {"burn.mp4": tmp_path / "b"}, False)
    assert seen == {"timeout_s": finalize.FILM_BURN_RENDER_TIMEOUT_S}


def test_run_refuses_film_burn_render_timeout(monkeypatch):
    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(finalize.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match=r"film burn accents ffmpeg timed out after 1800s"):
        finalize._run(["ffmpeg"], "film burn accents", timeout_s=finalize.FILM_BURN_RENDER_TIMEOUT_S)


@pytest.mark.parametrize("which", ["probe", "decode"])
def test_detect_flares_refuses_subprocess_failure(monkeypatch, which):
    bad = SimpleNamespace(returncode=1, stdout="", stderr=b"bad media")
    good_probe = SimpleNamespace(returncode=0, stdout="1.0", stderr=b"")
    def fake_run(command, **_kwargs):
        return bad if which == "probe" or command[0] == "ffmpeg" else good_probe

    monkeypatch.setattr(accents.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="film_burn"):
        accents.detect_flares("burn.mp4")


@pytest.mark.parametrize("which, timeout", [
    ("probe", accents.FILM_BURN_PROBE_TIMEOUT_S),
    ("decode", accents.FILM_BURN_DECODE_TIMEOUT_S),
])
def test_detect_flares_timeouts_refuse_loudly(monkeypatch, which, timeout):
    def fake_run(command, **kwargs):
        expected_timeout = accents.FILM_BURN_PROBE_TIMEOUT_S if command[0] == "ffprobe" else timeout
        assert kwargs["timeout"] == expected_timeout
        if which == "probe" or command[0] == "ffmpeg":
            raise accents.subprocess.TimeoutExpired(command, timeout)
        return SimpleNamespace(returncode=0, stdout="1.0", stderr=b"")

    monkeypatch.setattr(accents.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="film_burn .*timed out"):
        accents.detect_flares("burn.mp4")


def test_render_accepts_film_burn_spec_and_reaches_execution_gate(monkeypatch):
    spec = RenderSpec.model_validate({
        "spec_version": 6, "job_id": "j", "slug": "s", "mode": "final",
        "inputs": [
            {"id": "base", "kind": "video", "sha256": "0" * 64, "url": "u"},
            {"id": "burn.mp4", "kind": "video", "sha256": "1" * 64, "url": "u"},
            {"id": "clicks.wav", "kind": "audio", "sha256": "2" * 64, "url": "u"},
        ],
        "timeline": {"fps": 30, "width": 1080, "height": 1920,
                     "segments": [{"src": "base", "in": 0, "out": 1, "speed": 1}]},
        "encode": {"video": "libx264", "preset": "medium", "cq": 23, "pix_fmt": "yuv420p",
                   "audio": "aac", "audio_bitrate": "192k"},
        "outputs": [{"id": "master", "kind": "master", "put_url": "u"}],
        "overlays": {"finalize": {"accents": [{"kind": "film_burn", "at": 0.5,
                                                   "intensity": 1, "burn": "burn.mp4",
                                                   "clicks": "clicks.wav"}]}},
    })

    class CP:
        def send_event(self, *_a, **_k):
            return True

    monkeypatch.setattr(render, "_gpu_available", lambda: (_ for _ in ()).throw(
        AssertionError("execution gate reached")))
    with pytest.raises(AssertionError, match="execution gate reached"):
        render.render_spec(spec, CP())
