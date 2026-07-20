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
        "spec_version": 2, "job_id": "j", "slug": "s", "mode": "final",
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
    # trim [in,in+dur], Ken Burns move (scale-2x cover → zoompan), seat at start
    assert "trim=start=0.3:duration=2.4" in g
    assert "scale=2160:3840:force_original_aspect_ratio=increase:flags=lanczos,crop=2160:3840" in g
    assert "zoompan=z='(1.0+(0.12" in g            # 'in' preset zooms 1.0→1.0+amount (cutaway is NOT frozen)
    assert ":d=1:s=1080x1920:fps=30" in g          # zoompan emits canvas-size frames
    assert "setpts=PTS-STARTPTS+23.750/TB" in g


def test_broll_kenburns_preset_direction():
    # 'out' zooms 1+amount→1.0; a pure pan ('right') keeps a constant pan-zoom and moves x across the clip
    clip = _hardcut(); clip["preset"] = "out"; clip["amount"] = 0.2
    g = render.build_filtergraph(_spec(clip), gpu=False)
    assert "zoompan=z='(1.2+(-0.2" in g            # out: z0=1.2 → z1=1.0
    clip2 = _hardcut(); clip2["preset"] = "right"
    g2 = render.build_filtergraph(_spec(clip2), gpu=False)
    assert "zoompan=z='(1.08+(0.0" in g2           # pan: constant 1+pan_zoom (0.08)
    assert "(iw-iw/zoom)*(0.15+(0.7" in g2         # x pans 0.15→0.85
    # hard cut → plain overlay gated to the clip's span, base passes on eof
    assert "overlay=enable='between(t,23.750,26.150)':eof_action=pass" in g


def test_no_broll_keeps_vout_directly():
    spec = RenderSpec.model_validate({
        "spec_version": 2, "job_id": "j", "slug": "s", "mode": "preview",
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
    # trims are the last overlay still unimplemented on the pod (broll/music/cover/sfx/captions/mograph are done)
    spec = RenderSpec.model_validate({
        "spec_version": 2, "job_id": "j", "slug": "s", "mode": "final",
        "inputs": [_BASE_INPUT], "timeline": _TIMELINE, "encode": _ENCODE,
        "outputs": [{"id": "master", "kind": "master", "put_url": "p"}],
        "overlays": {"trims": [{"a": 1.0, "b": 2.0}]},
    })
    with pytest.raises(NotImplementedError, match="trims"):
        render.render_spec(spec, None)  # type: ignore[arg-type]


def _music_spec() -> RenderSpec:
    return RenderSpec.model_validate({
        "spec_version": 2, "job_id": "j", "slug": "s", "mode": "final",
        "inputs": [_BASE_INPUT, {"id": "music/t.mp3", "kind": "audio", "sha256": "2" * 64, "url": "u"}],
        "timeline": _TIMELINE, "encode": _ENCODE,
        "outputs": [{"id": "master", "kind": "master", "put_url": "p"}],
        "overlays": {"music": {"track": "music/t.mp3", "start": 30.0, "gain": 0.09}},
    })


def test_music_audio_graph_mixes_voice_and_bed():
    spec = _music_spec()
    a = render._AudioMix(voice_idx=0, bed_idx=1, clean="highpass=f=80",
                         vln="loudnorm=I=-20:TP=-1.5:LRA=11", dur=60.0)
    g = render.build_filtergraph(spec, gpu=False, audio=a)
    # video concatenates WITHOUT audio (a=0); the mix owns [aout]
    assert "concat=n=1:v=1:a=0[vout]" in g
    assert "[0:a]highpass=f=80,loudnorm=I=-20:TP=-1.5:LRA=11,apad=whole_dur=60" in g
    assert "sidechaincompress=threshold=0.06:ratio=3" in g  # locked DUCK
    assert "amix=inputs=2:duration=first:dropout_transition=0:normalize=0" in g
    assert "[bg0]" in g and g.strip().endswith("[aout]")


def test_no_audio_keeps_segment_audio_concat():
    spec = _music_spec()
    g = render.build_filtergraph(spec, gpu=False, audio=None)  # audio not resolved → base passthrough
    assert "concat=n=1:v=1:a=1[vout][aout]" in g
    assert "sidechaincompress" not in g


def test_sfx_mix_delays_sounds_over_master_with_limiter():
    a = render._AudioMix(voice_idx=0, bed_idx=1, clean="highpass=f=80",
                         vln="loudnorm=I=-20:TP=-1.5:LRA=11", dur=60.0,
                         sfx=((2, 12.56, 0.4), (3, 52.87, 0.5)))
    g = ";".join(render._audio_mix_chains(a))
    assert "[2:a]adelay=12560:all=1,volume=0.4[sx0]" in g
    assert "[3:a]adelay=52870:all=1,volume=0.5[sx1]" in g
    assert "[amaster][sx0][sx1]amix=inputs=3:normalize=0:duration=first[mx]" in g
    assert "alimiter=limit=0.84" in g and g.endswith("[aout]")


def test_unresolved_sfx_sound_reddens():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="sfx"):
        RenderSpec.model_validate({
            "spec_version": 2, "job_id": "j", "slug": "s", "mode": "final",
            "inputs": [_BASE_INPUT], "timeline": _TIMELINE, "encode": _ENCODE,
            "outputs": [{"id": "master", "kind": "master", "put_url": "p"}],
            "overlays": {"sfx": [{"sound": "sfx/missing.wav", "at": 1.0, "gain": 0.4}]},
        })


def test_sfx_without_music_skips_bed():
    a = render._AudioMix(voice_idx=0, bed_idx=None, clean="highpass=f=80",
                         vln="loudnorm=I=-20:TP=-1.5:LRA=11", dur=60.0, sfx=((2, 1.0, 0.4),))
    g = ";".join(render._audio_mix_chains(a))
    assert "sidechaincompress" not in g  # no bed to duck
    assert "apad=whole_dur=60[amaster]" in g
    assert "amix=inputs=2:normalize=0:duration=first[mx]" in g
