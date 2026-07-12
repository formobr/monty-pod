"""Pure unit tests for the render translator — filtergraph and argv only, no ffmpeg execution."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from podagent import render
from podagent.models import MotionKeyframe, RenderSpec

_EXAMPLES = Path(__file__).resolve().parents[1] / "contracts" / "examples"


def _data(name: str) -> dict:
    return json.loads((_EXAMPLES / name).read_text())


def _spec(name: str) -> RenderSpec:
    return RenderSpec.model_validate(_data(name))


def test_preview_golden_graph() -> None:
    g = render.build_filtergraph(_spec("spec.preview.json"), gpu=True)
    assert "trim=start=12.333:end=18.9" in g
    assert "setpts=(PTS-STARTPTS)/1.5" in g
    assert "atempo=1.5" in g
    assert "concat=n=2:v=1:a=1" in g
    assert g.rstrip().endswith("[vout][aout]")


def test_anim_expr_smoothstep_and_constant() -> None:
    kfs = [
        MotionKeyframe(t=0.0, rect=[0.1, 0.05, 0.8, 0.8]),
        MotionKeyframe(t=5.566, rect=[0.14, 0.08, 0.72, 0.72]),
    ]
    e = render.anim_expr(kfs, 0, "ease_in_out", "iw")
    assert "if(" in e
    assert "3-2*" in e  # smoothstep p*p*(3-2p)
    assert "0.1" in e and "0.14" in e  # both rect x values present
    assert e.endswith("*iw")

    one = render.anim_expr(kfs[:1], 0, "ease_in_out", "iw")
    assert "if(" not in one
    assert "0.1" in one


def test_atempo_chain_product() -> None:
    data = _data("spec.preview.json")
    data["timeline"]["segments"][0]["speed"] = 2.5  # past the single-atempo ceiling
    g = render.build_filtergraph(RenderSpec.model_validate(data), gpu=False)
    a0 = next(c for c in g.split(";") if c.endswith("[a0]"))
    factors = [float(x) for x in re.findall(r"atempo=([0-9.]+)", a0)]
    assert len(factors) == 2
    prod = 1.0
    for f in factors:
        prod *= f
    assert abs(prod - 2.5) < 1e-6


def test_gpu_vs_cpu_motion() -> None:
    spec = _spec("spec.preview.json")
    g = render.build_filtergraph(spec, gpu=True)
    assert "libplacebo" in g
    assert "hwupload" in g and "hwdownload" in g

    c = render.build_filtergraph(spec, gpu=False)
    assert "libplacebo" not in c
    # static crop at seg0's first keyframe rect [x=0.1, y=0.05, w=0.8, h=0.8]
    assert "crop=w=iw*0.8:h=ih*0.8:x=iw*0.1:y=ih*0.05" in c


def test_build_command_encode_flags() -> None:
    spec = _spec("spec.preview.json")
    ipaths = {i.id: Path(f"/work/{i.id}") for i in spec.inputs}
    out = Path("/work/out.mp4")

    cpu = render.build_command(spec, ipaths, out, gpu=False)
    assert "libx264" in cpu and "-crf" in cpu
    assert "-movflags" in cpu and "+faststart" in cpu

    gpu = render.build_command(spec, ipaths, out, gpu=True)
    assert "h264_nvenc" in gpu and "-cq" in gpu
    assert "-movflags" in gpu and "+faststart" in gpu


def test_final_overlays_not_implemented() -> None:
    spec = _spec("spec.final.json")

    class _StubCP:
        def post_event(self, payload: dict) -> None:
            raise AssertionError("post_event must not be reached before the render runs")

    with pytest.raises(NotImplementedError):
        render.render_spec(spec, _StubCP())  # type: ignore[arg-type]


def test_input_ids_excludes_non_av_assets() -> None:
    # a caption/cover font is downloaded but must NOT be fed as ffmpeg -i (a TTF is not a decodable stream)
    spec = _spec("spec.final.json")
    ids = render.input_ids(spec)
    assert "caption_font" not in ids
    assert spec.timeline.segments[0].src in ids                 # the base video IS consumed
    assert spec.overlays.broll_final.broll[0].clip in ids       # broll clips ARE consumed
    ipaths = {i.id: Path(f"/work/{i.id}") for i in spec.inputs}
    cmd = render.build_command(spec, ipaths, Path("/work/out.mp4"), gpu=False)
    assert "/work/caption_font" not in [cmd[n + 1] for n, a in enumerate(cmd) if a == "-i"]


def test_multi_source_input_order() -> None:
    data = _data("spec.preview.json")
    src2 = dict(data["inputs"][0])
    src2["id"] = "src2"
    data["inputs"].append(src2)
    data["timeline"]["segments"][1]["src"] = "src2"  # seg1 now reads the second input
    spec = RenderSpec.model_validate(data)

    assert render.input_ids(spec) == ["src", "src2"]
    g = render.build_filtergraph(spec, gpu=False)
    assert "[0:v]" in g and "[1:v]" in g

    ipaths = {i.id: Path(f"/work/{i.id}") for i in spec.inputs}
    cmd = render.build_command(spec, ipaths, Path("/work/out.mp4"), gpu=False)
    i_paths = [cmd[n + 1] for n, a in enumerate(cmd) if a == "-i"]
    assert i_paths == ["/work/src", "/work/src2"]
