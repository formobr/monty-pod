"""The 2913bf1a source shape, generated here: speech-shaped bursts at ~-31 LUFS plus two 5 ms clicks, whose
CREST is what made loudnorm drop to dynamic mode and ship +0.41 dBTP. Here the clicks carry the integrated
measure outright, so the designed answer is a loud refusal — the engine's corpus cannot be imported."""
from __future__ import annotations

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


@pytest.mark.integration
def test_a_click_carried_quiet_source_is_refused_loud_with_its_numbers(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    src = tmp_path / "quiet_clicks.mp4"
    subprocess.run(["ffmpeg", "-y", *RECIPE, str(src)], check=True, capture_output=True, timeout=300)
    fin = SimpleNamespace(loudnorm=SimpleNamespace(i=TARGET, tp=CEIL, lra=11.0, attenuate_only=False))
    # The bed sits ~10 dB under the clicks, so the integrated measure is theirs and the brickwall removes
    # it: chasing that residual would only feed the limiter the same transients and ship 17-2024 again.
    with pytest.raises(RuntimeError) as exc:
        finalize.apply_loudnorm(fin, src, tmp_path / "master.mp4")
    msg = str(exc.value)
    assert "lufs=-26." in msg and f"target={TARGET}" in msg and "(residual +12.3" in msg, msg
    assert not (tmp_path / "master.mp4").exists(), "nothing may be delivered for a refused master"
