"""MISC-921: the partner's referral link is drawn under the watermark; nothing else about the graph moves."""
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from podagent import finalize, frames
from podagent.models import SpecWatermark
from podagent.render_onepass import _check_assets

SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "contracts" / "spec.schema.json").read_text())
WM_SCHEMA = SCHEMA["$defs"]["watermark"]

BASE = dict(sting="s", idle="i", width=300, margin=10, position="bottom-right")
REF = dict(ref_text="monty.fast/r/anna-k", ref_font="font", x=100, y=200)
GRAPH = dict(base_v="b", sting_v="s", width=300, overlay_xy="100:200", base_a=None,
             chime_a=None, chime_vol=1.0, grid="60", sample_rate=48000)


@pytest.mark.parametrize("delay", [0.0, 1.5])
def test_no_ref_text_graph_is_unchanged(delay):
    graph, out_v, out_a = finalize.watermark_filter(**GRAPH, delay=delay, idle_v="idle_in")
    normalize = f"pad={finalize.WM_CANVAS_W}:{finalize.WM_CANVAS_H}:0:(oh-ih)/2:color=black@0,fps=60"
    expect = (f"[s]{normalize},setpts=PTS-STARTPTS[i];"
              f"[idle_in]{normalize},setpts=PTS-STARTPTS[d];"
              "[i][d]concat=n=2:v=1:a=0[wm0];[wm0]scale=300:-1[wm]")
    if delay > 0:
        d = frames.frame_at(delay, "60")
        expect += (f";[wm]setpts=PTS+{frames.pts_at(d, '60')}[wmd]"
                   f";[b][wmd]overlay=100:200:enable='gte({frames.index('60')},{d})':shortest=1:format=auto[wmv]")
    else:
        expect += ";[b][wm]overlay=100:200:shortest=1:format=auto[wmv]"
    assert graph == expect and (out_v, out_a) == ("wmv", None)


def _with_ref(delay, **kw):
    return finalize.watermark_filter(**{**GRAPH, "idle_v": "idle_in"}, delay=delay, ref_text="monty.fast/r/anna-k",
                                     ref_font_path="/fonts/Brand.ttf", ref_xy=(100, 200), **kw)


@pytest.mark.parametrize("delay", [0.0, 1.5])
def test_ref_text_is_one_drawtext_under_the_mascot_on_the_overlay_window(delay):
    graph, out_v, _ = _with_ref(delay, ref_size=30, ref_color="#12ab34", ref_gap=-5)
    assert out_v == "wmv"
    assert graph.count("drawtext=") == 1
    dt = graph.split("drawtext=", 1)[1]
    assert dt.endswith("[wmv]")
    assert "fontfile='/fonts/Brand.ttf'" in dt
    assert "text='monty.fast/r/anna-k'" in dt
    assert "fontsize=30" in dt and "fontcolor=#12ab34" in dt
    assert "x=100+(300-text_w)/2" in dt
    assert "y=200+300*600/1200+(-5)" in dt
    overlay_enable = re.search(r"overlay=[^;]*?(enable='[^']*')", graph)
    if delay > 0:
        assert overlay_enable and f":{overlay_enable.group(1)}" in dt
    else:
        assert overlay_enable is None and "enable=" not in dt


def test_ref_text_defaults_and_escaping():
    graph, _, _ = finalize.watermark_filter(
        **{**GRAPH, "idle_v": "i2"}, delay=0.0, ref_text="monty.fast/r/abc",
        ref_font_path="C:\\f:o.ttf", ref_xy=(1, 2))
    assert "fontsize=28" in graph and "fontcolor=#FFFFFF" in graph and "+(0)" in graph
    assert "fontfile='C\\:\\\\f\\:o.ttf'" in graph


def test_ref_text_without_font_or_xy_is_refused_by_the_builder():
    with pytest.raises(ValueError):
        finalize.watermark_filter(**GRAPH, idle_v="i2", delay=0.0, ref_text="monty.fast/r/abc")


def test_model_accepts_minimal_and_defaults():
    wm = SpecWatermark(**BASE, **REF)
    assert (wm.ref_size, wm.ref_color, wm.ref_gap) == (28, "#FFFFFF", 0)
    assert SpecWatermark(**BASE).ref_text is None


@pytest.mark.parametrize("bad", [
    dict(REF, ref_font=None),
    {k: v for k, v in REF.items() if k not in ("x", "y")},
    dict(REF, ref_text="monty.fast/r/a'b"),
    dict(REF, ref_text="evil.com/r/abc"),
    dict(REF, ref_text="monty.fast/r/ab"),
    dict(REF, ref_color="red"),
    dict(REF, ref_size=7),
    dict(REF, ref_gap=401),
])
def test_model_refuses(bad):
    with pytest.raises(ValidationError):
        SpecWatermark(**BASE, **{k: v for k, v in bad.items() if v is not None})


def test_schema_and_model_agree():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(WM_SCHEMA)
    cases = [dict(REF), dict(REF, ref_font=None), {k: v for k, v in REF.items() if k not in ("x", "y")},
             dict(REF, ref_text="monty.fast/r/a'b"), dict(REF, ref_text="evil.com/r/abc"),
             dict(REF, ref_text="monty.fast/r/ab"), dict(REF, ref_color="#12345"),
             dict(REF, ref_size=7), dict(REF, ref_size=201), dict(REF, ref_gap=-401),
             dict(REF, ref_gap=400, ref_size=8, ref_color="#aBcDeF"), dict(REF, ref_font="")]
    for case in cases:
        data = {**BASE, **{k: v for k, v in case.items() if v is not None}}
        try:
            SpecWatermark(**data)
            model_ok = True
        except ValidationError:
            model_ok = False
        assert validator.is_valid(data) == model_ok, case


def _spec(font):
    wm = SpecWatermark(**BASE, **dict(REF, ref_font=font))
    ov = SimpleNamespace(motion_plan=None, finalize=SimpleNamespace(accents=[], logo=None, watermark=wm))
    return SimpleNamespace(mode="final", overlays=ov)


def test_render_refuses_ref_font_that_is_not_an_input():
    paths = {"s": Path("s.webm"), "i": Path("i.webm"), "font": Path("f.ttf")}
    _check_assets(_spec("font"), paths)
    with pytest.raises(RuntimeError, match="nofont"):
        _check_assets(_spec("nofont"), paths)


def _host_font():
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/TTF/DejaVuSans.ttf"):
        if Path(p).is_file():
            return p
    return None


def _first_frame(ffmpeg, **ref):
    graph, out_v, _ = finalize.watermark_filter(
        base_v="0:v", sting_v="1:v", idle_v="2:v", width=200, overlay_xy="100:50", base_a=None, chime_a=None,
        chime_vol=1.0, delay=0.0, grid="30", sample_rate=48000, **ref)
    clip = f"color=c=black@0:s={finalize.WM_CANVAS_W}x{finalize.WM_CANVAS_H}:d=1:r=30,format=rgba"
    cmd = [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=0x202020:s=640x360:d=1:r=30",
           "-f", "lavfi", "-i", clip, "-f", "lavfi", "-i", clip, "-filter_complex", graph,
           "-map", f"[{out_v}]", "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def test_rendered_pixels_below_the_watermark_box_change_with_ref_text():
    ffmpeg = shutil.which("ffmpeg")
    font = _host_font()
    if ffmpeg is None or font is None:
        pytest.skip("needs a host ffmpeg and DejaVuSans.ttf")
    filters = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True).stdout
    if b"drawtext" not in filters:
        pytest.skip("host ffmpeg has no drawtext filter")
    plain = _first_frame(ffmpeg)
    with_ref = _first_frame(ffmpeg, ref_text="monty.fast/r/anna-k", ref_font_path=font, ref_xy=(100, 50))
    row = 640 * 3
    below = slice(150 * row, 200 * row)  # the box is y 50..150 (200px wide, 2:1 canvas)
    assert plain[below] != with_ref[below]
    assert plain[:150 * row] == with_ref[:150 * row]
