"""MISC-756: the contour-dry cut.apply placeholder lasts the cut its own JSON reports, never the source's length."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from podagent.ops import dry, registry

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not on this runner")

_FPS = 30.0
_SOURCE_S = 7.0
# Kept ranges whose total (5.656 s) is shorter than the 7.0 s source, as in contour-smoke-logo-key.
_KEEP = [[0.0, 2.0], [2.5, 5.156], [6.0, 7.0]]


def _source_with_audio(dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c=blue:s=64x64:r={int(_FPS)}:d={_SOURCE_S}",
         "-f", "lavfi", "-i", f"sine=f=440:r=48000:d={_SOURCE_S}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(dst)],
        check=True, capture_output=True, timeout=60)


def _source_without_audio(dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c=blue:s=64x64:r={int(_FPS)}:d={_SOURCE_S}",
         "-c:v", "libx264", "-preset", "ultrafast", str(dst)],
        check=True, capture_output=True, timeout=60)


def _durations(path: Path) -> list[float]:
    out = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,duration",
         "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=20).stdout)
    return [float(out["format"]["duration"])] + [float(s["duration"]) for s in out["streams"]]


def test_the_dry_cut_file_lasts_the_reported_cut_length(tmp_path):
    src = tmp_path / "source.mp4"
    _source_with_audio(src)
    params = {"keep": _KEEP, "fps_grid": _FPS}
    outputs = {"dst": tmp_path / "cut.mp4", "durs": tmp_path / "durs.json"}
    dry.resolve(registry.get("cut.apply"))(params=params, inputs={"src": src}, outputs=outputs)

    reported = sum(json.loads(outputs["durs"].read_text(encoding="utf-8"))["rdurs"])
    assert reported == pytest.approx(5.656)
    probed = _durations(outputs["dst"])
    assert len(probed) == 3, "the placeholder carries one video and one audio stream"
    for got in probed:
        assert abs(got - reported) <= 1.0 / _FPS, f"probed {got:.3f}s, reported cut {reported:.3f}s"
    assert abs(probed[0] - _SOURCE_S) > 1.0, "the placeholder must not run the source's full length"


def test_the_dry_cut_file_lasts_the_reported_cut_length_with_a_silent_source(tmp_path):
    """A bound src with no audio track takes the no-audio branch of _write_video — it must round the
    same way as the with-audio branch, not fall back to the colour source's own (rounded-up) d=."""
    src = tmp_path / "source.mp4"
    _source_without_audio(src)
    params = {"keep": _KEEP, "fps_grid": _FPS}
    outputs = {"dst": tmp_path / "cut.mp4", "durs": tmp_path / "durs.json"}
    dry.resolve(registry.get("cut.apply"))(params=params, inputs={"src": src}, outputs=outputs)

    reported = sum(json.loads(outputs["durs"].read_text(encoding="utf-8"))["rdurs"])
    assert reported == pytest.approx(5.656)
    probed = _durations(outputs["dst"])
    assert len(probed) == 2, "video only — no audio stream was fabricated"
    for got in probed:
        assert abs(got - reported) <= 1.0 / _FPS, f"probed {got:.3f}s, reported cut {reported:.3f}s"
    assert abs(probed[0] - _SOURCE_S) > 1.0, "the placeholder must not run the source's full length"
