"""MISC-239: the final receipt says what audio the graph that ran actually mixed — the engine's
own final-dispatch audio QC compares the planned music/SFX against this row (plan_match.ReceiptAudioMix)."""
from __future__ import annotations

from pathlib import Path

import pytest

from podagent import render, render_onepass as op
from podagent.models import RenderSpec

SHA = "0" * 64
# engine contract, its plan-match ReceiptAudioMix class — the field names exactly
ENGINE_FIELDS = {"music_bed": bool, "bed_lufs": (float, type(None)), "sfx_mixed": int}


def _spec(*, music: bool, sfx: int) -> RenderSpec:
    inputs = [{"id": "base", "kind": "video", "sha256": SHA, "url": "u"}]
    overlays: dict = {}
    if music:
        inputs.append({"id": "music/bed.mp3", "kind": "audio", "sha256": SHA, "url": "u"})
        overlays["music"] = {"track": "music/bed.mp3", "start": 0.0, "gain": 0.35}
    if sfx:
        inputs += [{"id": f"sfx/{k}.wav", "kind": "audio", "sha256": SHA, "url": "u"} for k in range(sfx)]
        overlays["sfx"] = [{"sound": f"sfx/{k}.wav", "at": 1.0 + k, "gain": 0.5} for k in range(sfx)]
    return RenderSpec.model_validate({
        "spec_version": 6, "job_id": "j-am", "slug": "am", "mode": "final", "inputs": inputs,
        "timeline": {"fps": 30, "width": 320, "height": 240,
                     "segments": [{"src": "base", "in": 0.0, "out": 6.0, "speed": 1.0}]},
        "encode": {"video": "libx264", "preset": "veryfast", "cq": 23, "pix_fmt": "yuv420p",
                   "audio": "aac", "audio_bitrate": "192k"},
        "overlays": overlays,
        "outputs": [{"id": "receipt", "kind": "receipt", "put_url": "https://x/r.json?sig=PUT"},
                    {"id": "master", "kind": "master", "put_url": "https://x/m.mp4?sig=PUT"}]})


def _receipt(spec: RenderSpec, monkeypatch, tmp_path: Path) -> dict:
    """Through the real `prepare` → `assemble` → `build_receipt`; only the ffmpeg pre-passes are stubbed."""
    monkeypatch.setenv("POD_IMAGE_TAG", "b" * 40)
    monkeypatch.setattr(op, "_check_assets", lambda *a: None)
    monkeypatch.setattr(op, "_check_inputs", lambda *a: None)
    monkeypatch.setattr(op, "_voice_filters", lambda *a: ("highpass=f=80", "loudnorm=I=-20:TP=-1.5:LRA=11"))
    monkeypatch.setattr(render, "_prerender_bed", lambda *a: tmp_path / "music_bed.flac")
    paths = {i.id: tmp_path / i.id.replace("/", "__") for i in spec.inputs}
    p = op.prepare(spec, paths, tmp_path, gpu=False)
    graph, cmd = op.assemble(p)
    return op.build_receipt(p, graph, cmd, 0.0)


def _engine_valid(row: dict) -> None:
    assert set(row) == set(ENGINE_FIELDS)
    for k, t in ENGINE_FIELDS.items():
        assert isinstance(row[k], t), (k, row[k])
    assert not isinstance(row["sfx_mixed"], bool)


def test_the_receipt_says_what_audio_was_mixed(monkeypatch, tmp_path) -> None:
    row = _receipt(_spec(music=True, sfx=3), monkeypatch, tmp_path)["audio_mix"]
    _engine_valid(row)
    assert row == {"music_bed": True, "bed_lufs": render._MUSIC_LUFS, "sfx_mixed": 3}


@pytest.mark.parametrize("sfx", [0, 2])
def test_a_voice_only_graph_reports_no_music_bed(monkeypatch, tmp_path, sfx) -> None:
    row = _receipt(_spec(music=False, sfx=sfx), monkeypatch, tmp_path)["audio_mix"]
    _engine_valid(row)
    assert row == {"music_bed": False, "bed_lufs": None, "sfx_mixed": sfx}


def test_the_row_reads_the_graph_not_the_plan() -> None:
    """A bed index the graph never wires, or an SFX chain missing from it, is not reported as mixed."""
    a = render._AudioMix(voice_idx=0, bed_idx=1, clean="highpass=f=80", vln="loudnorm=I=-20", dur=6.0,
                         sfx=((2, 1.0, 0.5), (3, 2.0, 0.5)))
    graph = ";".join(render._audio_mix_chains(a))
    assert render.audio_mix_facts(a, graph) == {"music_bed": True, "bed_lufs": -33.0, "sfx_mixed": 2}
    unwired = ";".join(c for c in graph.split(";") if not c.startswith(("[1:a]", "[3:a]")))
    assert render.audio_mix_facts(a, unwired) == {"music_bed": False, "bed_lufs": None, "sfx_mixed": 1}
    assert render.audio_mix_facts(None, graph) == {"music_bed": False, "bed_lufs": None, "sfx_mixed": 0}
