"""cover.compose — the Pillow still. Colour-token resolution + a real (small) draw over a base frame
using the system fallback font (no brand font needed). ffmpeg weld is a smoke concern, not here."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from podagent import cover


def test_color_resolution():
    colors = {"white": (242, 242, 240), "accent": (214, 255, 58), "black": (10, 10, 10)}
    assert cover._color("accent", colors) == (214, 255, 58)
    assert cover._color([1, 2, 3], colors) == (1, 2, 3)          # literal RGB passthrough
    assert cover._color("lime", colors) == (242, 242, 240)       # unknown token → default white
    assert cover._color("x", {}) == (255, 255, 255)              # empty map → hard white


def test_compose_writes_sized_png(tmp_path: Path):
    from PIL import Image
    base = tmp_path / "frame.jpg"
    Image.new("RGB", (400, 700), (30, 30, 30)).save(base)
    spec = {
        "frame_at": 5.0,
        "headline": {"lines": [[{"t": "НЕ", "c": "white"}], [{"t": "ЛЕНЬ", "c": "accent"}]],
                     "pos": "bottom", "y": 0.72, "size": 90.0, "box": True},
        "colors": {"white": [242, 242, 240], "accent": [214, 255, 58], "black": [10, 10, 10]},
        "elements": [{"type": "badge", "x": 0.5, "y": 0.2, "t": "LIVE", "bg": "accent", "fg": "black"}],
    }
    out = tmp_path / "cover.png"
    cover.compose(base, spec, {}, out, 216, 384)  # small canvas; fallback font
    assert out.is_file()
    with Image.open(out) as im:
        assert im.size == (216, 384)


def test_compose_no_headline_still_renders(tmp_path: Path):
    from PIL import Image
    base = tmp_path / "frame.jpg"
    Image.new("RGB", (216, 384), (0, 0, 0)).save(base)
    out = tmp_path / "c.png"
    cover.compose(base, {"frame_at": 0.0, "headline": {"lines": []}, "colors": {}}, {}, out, 216, 384)
    assert out.is_file()


@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "gpu"])
def test_weld_video_reencodes_set_bt709(monkeypatch, tmp_path: Path, gpu):
    commands = []
    monkeypatch.setattr(cover, "_probe", lambda _p: ("30", "2"))
    monkeypatch.setattr(cover.subprocess, "run", lambda cmd, **_kw: commands.append(cmd))
    cover.weld(tmp_path / "master.mp4", tmp_path / "cover.png", tmp_path / "out.mp4", 0.6, gpu, 216, 384)

    assert len(commands) == 2
    assert all(any("setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709" in token
                   for token in cmd) for cmd in commands)


@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "gpu"])
def test_weld_declares_grid_and_delivery_sample_rate(monkeypatch, tmp_path: Path, gpu):
    """Finding 2/6: neither output clause may pass an inherited sample rate through, and both must
    pin a CFR grid rather than leaving the rate to whatever ffmpeg negotiates."""
    commands = []
    monkeypatch.setattr(cover, "_probe", lambda _p: ("30000/1001", "2"))
    monkeypatch.setattr(cover.subprocess, "run", lambda cmd, **_kw: commands.append(cmd))
    cover.weld(tmp_path / "master.mp4", tmp_path / "cover.png", tmp_path / "out.mp4", 0.6, gpu, 216, 384)

    assert len(commands) == 2
    for cmd in commands:
        assert "-r" in cmd and cmd[cmd.index("-r") + 1] == "30000/1001"
        assert "-fps_mode" in cmd and cmd[cmd.index("-fps_mode") + 1] == "cfr"
        assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "48000"
    assert "aresample=48000" in commands[1][commands[1].index("-filter_complex") + 1]


def test_probe_refuses_an_unmeasurable_fps_instead_of_defaulting(monkeypatch, tmp_path: Path):
    """Finding 2: the old '30' literal fallback is gone — an unmeasurable rate must raise."""
    monkeypatch.setattr(cover.subprocess, "run",
                        lambda *_a, **_kw: subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="could not measure the delivery grid"):
        cover._probe(tmp_path / "silent.mp4")


def test_probe_carries_a_deadline(monkeypatch, tmp_path: Path):
    """A hang here holds a finished weld open forever; the probe must bound its wait."""
    timeouts = []

    def fake_run(cmd, **kw):
        timeouts.append(kw.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, stdout="30\n", stderr="")
    monkeypatch.setattr(cover.subprocess, "run", fake_run)
    cover._probe(tmp_path / "clip.mp4")
    assert timeouts and all(timeouts)


def test_probe_fps_query_timeout_hits_the_same_unmeasurable_refusal(monkeypatch, tmp_path: Path):
    """A timeout on the fps query must NOT invent a new failure mode — same refusal as empty stdout."""
    def timeout(cmd, **_kw):
        raise subprocess.TimeoutExpired(cmd, 20)
    monkeypatch.setattr(cover.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="could not measure the delivery grid"):
        cover._probe(tmp_path / "hung.mp4")


def test_probe_channels_query_timeout_keeps_the_existing_stereo_fallback(monkeypatch, tmp_path: Path):
    """The channels query already tolerates empty output by defaulting to stereo; a timeout on THAT
    query must land on the same fallback, not raise."""
    def flaky_run(cmd, **_kw):
        if "a:0" in cmd:
            raise subprocess.TimeoutExpired(cmd, 20)
        return subprocess.CompletedProcess(cmd, 0, stdout="30\n", stderr="")
    monkeypatch.setattr(cover.subprocess, "run", flaky_run)
    fps, ch = cover._probe(tmp_path / "clip.mp4")
    assert fps == "30"
    assert ch == "2"
