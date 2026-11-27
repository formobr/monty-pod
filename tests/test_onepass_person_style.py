"""super-director v13 R6 — person-plane STYLES in the one-pass graph (spec overlays.person_matte.styles): a contour
(dilation × width + gblur glow), a drop shadow, black and white (U = V = 128) and a greyed frame (hue s=0), each on its
own windows; no styles = the R4 graph, byte for byte. The graph is also run through the real ffmpeg on synthetic
sources so every filter spelling is one ffmpeg accepts."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from podagent import mograph, render_onepass as op
from podagent.models import RenderSpec

from test_render_onepass import _BASE_SPEC, _prepared, assert_connected, _MASTER_MAPS, _REF_MAPS  # noqa: E402

SHA = "0" * 64
LAYERS = (
    {"mov": "/w/behind.mov", "start": 2.0, "dur": 3.0, "glass": False, "z": 1},
    {"mov": "/w/over.mov", "start": 3.0, "dur": 1.0, "glass": False, "z": 4},
)
NEON = {"windows": [[2.0, 3.0]], "ring": {"color": "#d6ff3a", "width": 3, "glow": 8.0}}
POSTER = {"windows": [[3.0, 4.0]], "gray": True, "frame_gray": True, "ring": {"color": "#ffffff", "width": 2},
          "shadow": {"dx": 0, "dy": 6, "blur": 4.0, "opacity": 0.4}}


def _spec(styles=None) -> RenderSpec:
    d = json.loads(json.dumps(_BASE_SPEC))
    d["inputs"].append({"id": "person_matte_alpha", "kind": "video", "sha256": SHA, "url": "https://x/a.mkv"})
    d["overlays"]["person_matte"] = {"alpha": "person_matte_alpha", "windows": [[2.0, 5.0]]}
    if styles is not None:
        d["overlays"]["person_matte"]["styles"] = styles
    return RenderSpec.model_validate(d)


def test_no_styles_is_the_r4_graph_byte_for_byte():
    assert mograph.person_style_overlays("b", "p", [(2.0, 5.0)], None, 30, "o") == \
        mograph.person_overlay("b", "p", [(2.0, 5.0)], 30, "o")
    plain, _ = op.assemble(_prepared(_spec(), layers=LAYERS))
    assert "dilation" not in plain and "vframegray" not in plain and "lutyuv" not in plain


def test_styles_draw_ring_glow_shadow_gray_on_their_own_windows():
    graph, _ = op.assemble(_prepared(_spec([NEON, POSTER]), layers=LAYERS))
    assert graph.count("dilation=enable=") == 3 + 2          # width 3 for neon, 2 for the poster
    assert "gblur=sigma=8:steps=3" in graph and "gblur=sigma=4:steps=3" in graph
    assert "lutyuv=u=128:v=128" in graph                      # the poster's b/w person
    assert "hue=s=0:enable=" in graph and "[vframegray]" in graph
    y, u, v = mograph.yuv709("#d6ff3a")
    assert f"lutyuv=y={y}:u={u}:v={v}" in graph
    assert "overlay=x=0:y=6:" in graph                        # the shadow's offset
    # the plain cut-out keeps [4, 5): frames 120..149 at 30 fps; neon [2, 3) = 60..89
    assert ",120,149)" in graph and ",60,89)" in graph
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


def test_a_later_style_wins_where_two_overlap_and_styles_stay_inside_the_plane():
    a = {"windows": [[1.0, 4.0]], "gray": True}
    b = {"windows": [[3.0, 6.0]], "ring": {"color": "#ffffff", "width": 1}}
    assert mograph.style_windows([a, b], [(2.0, 5.0)]) == [[(2.0, 3.0)], [(3.0, 5.0)]]
    assert mograph.frame_gray_chain("x", [a], [(2.0, 5.0)], 30, "y") is None


def test_yuv709_limited_range():
    assert mograph.yuv709("#ffffff") == (235, 128, 128)
    assert mograph.yuv709("#000000") == (16, 128, 128)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_the_style_graph_runs_in_ffmpeg():
    """2 s of a yuva plane (a white disc on alpha) over a grey base: every style filter parses and the ring colour
    lands next to the disc."""
    windows = [(0.0, 2.0)]
    st = [{"windows": [[0.0, 1.0]], **{k: v for k, v in NEON.items() if k != "windows"}},
          {"windows": [[1.0, 2.0]], **{k: v for k, v in POSTER.items() if k != "windows"}}]
    gray = mograph.frame_gray_chain("0:v", st, windows, 30, "vg")
    over = mograph.person_style_overlays("vg", "pl", windows, st, 30, "vout")
    fc = (f"{gray};[1:v]format=yuva420p,geq=lum='255':cb=128:cr=128:"
          f"a='if(lt(hypot(X-32,Y-32),12),255,0)'[pl];{over}")
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=30:d=2", "-f", "lavfi", "-i",
           "color=c=white:s=64x64:r=30:d=2", "-filter_complex", fc, "-map", "[vout]", "-frames:v", "2",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode()[-800:]
    px = r.stdout[:64 * 64 * 3]
    ring_px = px[(32 * 64 + 32 + 13) * 3:(32 * 64 + 32 + 13) * 3 + 3]    # 13 px right of the disc centre: the ring
    assert ring_px[1] > 200 and ring_px[2] < 120, list(ring_px)          # lime: high G, low B
