"""final b-roll cutaway composite in build_filtergraph — cover-crop + time-shifted overlay, plus
slide/push (overlay x/y) and dissolve (alpha fade) transitions. Pure graph assertions, no ffmpeg."""
from __future__ import annotations

import pytest

from podagent import render
from podagent.models import RenderSpec

_BASE_INPUT = {"id": "base", "kind": "video", "sha256": "0" * 64, "url": "u"}
_CLIP_INPUT = {"id": "broll/c.mp4", "kind": "video", "sha256": "1" * 64, "url": "u"}
_ENCODE = {"video": "libx264", "preset": "medium", "cq": 23, "pix_fmt": "yuv420p",
           "audio": "aac", "audio_bitrate": "192k"}
_TIMELINE = {"fps": 30, "width": 1080, "height": 1920,
             "segments": [{"src": "base", "in": 0.0, "out": 60.0, "speed": 1.0}]}


def _spec(clip: dict) -> RenderSpec:
    return RenderSpec.model_validate({
        "spec_version": 1, "job_id": "j", "slug": "s", "mode": "final",
        "inputs": [_BASE_INPUT, _CLIP_INPUT], "timeline": _TIMELINE, "encode": _ENCODE,
        "outputs": [{"id": "master", "kind": "master", "put_url": "p"}],
        "overlays": {"broll_final": {"broll": [clip]}},
    })


def _hardcut() -> dict:
    return {"clip": "broll/c.mp4", "start": 23.75, "preset": "in", "dur": 2.4, "in": 0.3}


def test_hardcut_broll_overlays_and_covers():
    g = render.build_filtergraph(_spec(_hardcut()), gpu=False)
    # base lands in [vbase], the cutaway overlay produces [vout]
    assert "[vbase][aout]" in g
    assert "[vout]" in g
    # cover-crop the raw clip to canvas, trim [in,in+dur], seat at start
    assert "trim=start=0.3:duration=2.4" in g
    assert "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920" in g
    assert "setpts=PTS-STARTPTS+23.750/TB" in g
    # hard cut → plain overlay gated to the clip's span, base passes on eof
    assert "overlay=enable='between(t,23.750,26.150)':eof_action=pass" in g


def test_no_broll_keeps_vout_directly():
    spec = RenderSpec.model_validate({
        "spec_version": 1, "job_id": "j", "slug": "s", "mode": "preview",
        "inputs": [_BASE_INPUT], "timeline": _TIMELINE, "encode": _ENCODE,
        "outputs": [{"id": "master", "kind": "master", "put_url": "p"}],
    })
    g = render.build_filtergraph(spec, gpu=False)
    assert "[vout][aout]" in g  # unchanged base path, no [vbase]
    assert "vbase" not in g


def test_slide_transition_rides_overlay_xy():
    clip = _hardcut()
    clip["transition_in"] = {"kind": "slide_wipe", "edge": "entry", "direction": "left", "dur": 0.35}
    g = render.build_filtergraph(_spec(clip), gpu=False)
    # entry window rides an eased overlay x offset; 'left' enters from the right edge (W-W*e)
    assert "overlay=x='" in g
    assert "W-W*" in g
    assert "between(t,23.7500,24.1000)" in g  # [start, start+dur]


def test_push_return_uses_end_window():
    clip = _hardcut()
    clip["transition_out"] = {"kind": "push", "edge": "return", "direction": "up", "dur": 0.4}
    g = render.build_filtergraph(_spec(clip), gpu=False)
    assert "H-H*" in g  # 'up' moves on y
    assert "between(t,25.7500,26.1500)" in g  # [end-dur, end], end=26.15


def test_dissolve_is_alpha_fade_not_overlay_move():
    clip = _hardcut()
    clip["transition_in"] = {"kind": "dissolve", "edge": "entry", "dur": 0.5}
    g = render.build_filtergraph(_spec(clip), gpu=False)
    assert "format=yuva420p" in g
    assert "fade=t=in:st=23.750:d=0.500:alpha=1" in g
    assert "overlay=enable=" in g  # no x/y move for a dissolve


def test_render_spec_fails_loud_on_unimplemented_overlays():
    from podagent.cp import ControlPlane  # noqa: F401  (only for type; we don't build one)
    clip = _hardcut()
    spec = RenderSpec.model_validate({
        "spec_version": 1, "job_id": "j", "slug": "s", "mode": "final",
        "inputs": [_BASE_INPUT, _CLIP_INPUT, {"id": "music/t.mp3", "kind": "audio",
                                              "sha256": "2" * 64, "url": "u"}],
        "timeline": _TIMELINE, "encode": _ENCODE,
        "outputs": [{"id": "master", "kind": "master", "put_url": "p"}],
        "overlays": {"broll_final": {"broll": [clip]},
                     "music": {"track": "music/t.mp3", "start": 30.0, "gain": 0.09}},
    })
    with pytest.raises(NotImplementedError, match="music"):
        render.render_spec(spec, None)  # type: ignore[arg-type]
