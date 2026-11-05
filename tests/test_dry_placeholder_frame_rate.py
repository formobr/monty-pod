"""The dry video placeholder keeps the frame rate of the bound real video input (engine frame-snaps to it)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from podagent.ops import dry, registry

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not on this runner")

_KEEP = [[0.0, 1.0], [1.5, 2.5]]
_CUT_S = 2.0


def _source(dst: Path, rate: str, *, audio: bool = True) -> Path:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"color=c=blue:s=64x64:r={rate}:d=3"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=f=440:r=48000:d=3", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-shortest", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return dst


def _audio_only(dst: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=f=440:r=48000:d=3", str(dst)],
        check=True, capture_output=True, timeout=60)
    return dst


def _rate(path: Path) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True, timeout=20).stdout.strip()


def _duration(path: Path) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True, timeout=20).stdout.strip())


def _cut(tmp_path: Path, src: Path | None) -> Path:
    outputs = {"dst": tmp_path / "cut.mp4", "durs": tmp_path / "durs.json"}
    inputs = {"src": src} if src is not None else {}
    dry.resolve(registry.get("cut.apply"))(params={"keep": _KEEP, "fps_grid": 30.0}, inputs=inputs, outputs=outputs)
    return outputs["dst"]


def _write(tmp_path: Path, src: Path | None, dur: float | None) -> Path:
    dst = tmp_path / "out.mp4"
    dry._write_video(dst, op_name="test.op", audio_src=src, dur=dur)
    return dst


def test_ntsc_source_rate_reaches_the_dur_pinned_cut_placeholder(tmp_path):
    src = _source(tmp_path / "s.mp4", "30000/1001")
    out = _cut(tmp_path, src)
    assert _rate(out) == "30000/1001"
    assert abs(_duration(out) - _CUT_S) <= 1.0 / 29.97


def test_ntsc_source_rate_reaches_the_shortest_placeholder(tmp_path):
    src = _source(tmp_path / "s.mp4", "30000/1001")
    assert _rate(_write(tmp_path, src, None)) == "30000/1001"


def test_ntsc_source_rate_reaches_the_no_audio_placeholder(tmp_path):
    src = _source(tmp_path / "s.mp4", "30000/1001", audio=False)
    assert _rate(_write(tmp_path, src, _CUT_S)) == "30000/1001"
    assert _rate(_write(tmp_path, src, None)) == "30000/1001"


def test_a_25_fps_source_gives_a_25_fps_placeholder(tmp_path):
    src = _source(tmp_path / "s.mp4", "25")
    assert _rate(_cut(tmp_path, src)) == "25/1"
    assert _rate(_write(tmp_path, src, None)) == "25/1"


def test_no_bound_input_keeps_the_synthetic_rates(tmp_path):
    assert _rate(_write(tmp_path, None, _CUT_S)) == "30/1"
    assert _rate(_write(tmp_path, None, None)) == "1/1"


def test_an_input_without_a_video_stream_falls_back_to_the_synthetic_rates(tmp_path):
    src = _audio_only(tmp_path / "a.m4a")
    assert _rate(_write(tmp_path, src, _CUT_S)) == "30/1"
    assert _rate(_write(tmp_path, src, None)) == "25/1"  # -shortest branch: colour source's own default
