"""The 2913bf1a source shape, generated here: speech-shaped bursts at ~-31 LUFS plus two 5 ms clicks, whose
CREST is what made loudnorm drop to dynamic mode and ship +0.41 dBTP. The engine's corpus cannot be imported
from this repo, so the recipe is spelled inline and the delivered true peak is measured, not predicted."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from podagent import finalize

CEIL = -1.0
TARGET = -14.0

RECIPE = [
    "-loglevel", "error",
    "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=8",
    "-f", "lavfi", "-i", "anoisesrc=r=48000:c=pink:a=0.05:d=8:seed=17",
    "-f", "lavfi", "-i",
    r"aevalsrc=0.63*sin(2*PI*1000*t)*(between(t\,2\,2.005)+between(t\,5.5\,5.505)):d=8:s=48000",
    "-filter_complex",
    r"[1:a]highpass=f=200,lowpass=f=3500,volume=between(mod(t\,1.2)\,0\,0.7):eval=frame[sp];"
    r"[sp][2:a]amix=inputs=2:duration=first:normalize=0,aformat=channel_layouts=stereo:sample_rates=48000[a]",
    "-map", "0:v", "-map", "[a]",
    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-ar", "48000", "-shortest",
]


def _measured(path: Path) -> dict:
    """One loudnorm print_format=json pass over the DELIVERED file — the same measure the pod's own verdict
    takes, so the number asserted here is the number the box would receive."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0?",
         "-af", f"loudnorm=I={TARGET}:TP={CEIL}:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    assert m, r.stderr[-2000:]
    return json.loads(m.group(0))


@pytest.mark.integration
def test_a_quiet_source_with_clicks_is_delivered_under_the_true_peak_ceiling(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    src = tmp_path / "quiet_clicks.mp4"
    subprocess.run(["ffmpeg", "-y", *RECIPE, str(src)], check=True, capture_output=True, timeout=300)
    fin = SimpleNamespace(loudnorm=SimpleNamespace(i=TARGET, tp=CEIL, lra=11.0, attenuate_only=False))
    out = finalize.apply_loudnorm(fin, src, tmp_path / "master.mp4")
    assert out == tmp_path / "master.mp4" and out.stat().st_size > 0
    tp = float(_measured(out)["input_tp"])
    assert tp <= CEIL, f"clipping master delivered at {tp} dBTP (ceiling {CEIL})"
