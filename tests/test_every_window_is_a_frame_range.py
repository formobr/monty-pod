"""MISC-246, pod side: every timed window the renderer emits is an integer output-FRAME range computed once
by podagent.frames (nearest frame, start inclusive, end exclusive), never a formatted float second — so two
co-terminal elements end on the same frame at any fps, whichever builder drew them. Pure graph assertions."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

import test_render_onepass as t
from podagent import frames, mograph, render, render_onepass as op
from podagent.models import RenderSpec

SHA = t.SHA
_GATE = re.compile(r"between\(round\(t\*(\d+(?:/\d+)?)\),(\d+),(\d+)\)")


def _window(clause: str) -> tuple[int, int]:
    """(first, last) output frame of a clause's `enable` gate (or of a bare gate expression)."""
    gate = re.search(r"enable='([^']*)'", clause)
    m = _GATE.search(gate.group(1) if gate else clause)
    assert m is not None, clause
    return int(m.group(2)), int(m.group(3))


def _clause(graph: str, pad: str) -> str:
    """The overlay clause that rides [pad] (rewire may namespace it)."""
    hits = [c for c in graph.split(";") if re.search(rf"\[{pad}(?:__\w+)?\]overlay=", c)]
    assert len(hits) == 1, (pad, hits)
    return hits[0]


def _merged_spec(fps: int, *, start: float, cut_dur: float, everything: bool = False) -> RenderSpec:
    """The onepass fixture with ONE still cutaway over [start, start+cut_dur) added; `everything` also
    turns on every accent kind and a film-burn set, so the whole builder surface lands in one graph."""
    d = json.loads(json.dumps(t._BASE_SPEC))
    d["timeline"]["fps"] = fps
    d["inputs"].append({"id": "broll/c.jpg", "kind": "image", "sha256": SHA, "url": "https://x/c.jpg"})
    d["overlays"]["broll_final"] = {"broll": [
        {"clip": "broll/c.jpg", "start": start, "preset": "in", "dur": cut_dur, "in": 0.0,
         "transition_in": {"kind": "slide_wipe", "edge": "entry", "direction": "left", "dur": 0.2},
         "transition_out": {"kind": "dissolve", "edge": "return", "dur": 0.3}}]}
    if everything:
        t._add_film_burn(d)
        d["overlays"]["finalize"]["accents"] += [
            {"kind": k, "at": 1.0 + 0.7 * i, "intensity": 0.5}
            for i, k in enumerate(("camera_shake", "grain", "zoom_punch", "glitch", "zoom_blur", "rgb_split"))]
    return RenderSpec.model_validate(d)


def _layers(start: float, dur: float) -> tuple[dict, ...]:
    backing = {"treatment": "glass", "from": 0.1,
               "motion": [{"t": 0.0, "rect": [-0.3, 0.1, 0.3, 0.1]}, {"t": 0.5, "rect": [0.05, 0.1, 0.3, 0.1]}]}
    return ({"mov": "/w/seq0.mov", "start": start, "dur": dur, "glass": True, "head_below": True,
             "backing": backing},)


# --- the one rounding helper ---------------------------------------------------

@pytest.mark.parametrize("fps", [30, 60, 29.97, 59.94])
def test_one_rounding_helper_puts_every_spelling_of_a_time_on_one_frame(fps) -> None:
    """The bug in one line: 2/60 s formatted to 3 decimals in one graph and to 6 in another must still
    be the same frame, at every grid, NTSC included."""
    for k in (1, 2, 4, 7, 61, 3599):
        exact = k / frames.rate(fps)
        assert {frames.frame_at(float(exact), fps), frames.frame_at(round(float(exact), 3), fps),
                frames.frame_at(round(float(exact), 6), fps)} == {k}


def test_a_frame_range_is_half_open_and_never_empty() -> None:
    assert frames.frame_range(1.0, 2.0, 30) == (30, 60)           # [30, 60): frame 60 is the next element's
    assert frames.between(30, 60, 30) == "between(round(t*30),30,59)"
    assert frames.frame_range(1.0, 1.001, 30) == (29, 30)         # a non-empty span keeps one frame…
    assert frames.frame_range(1.0, 1.01, 30) == (29, 30)          # …the one before its (unmoved) end frame
    assert frames.frame_range(0.0, 0.01, 30) == (0, 1)            # frame 0 has nothing before it
    assert frames.frame_range(1.0, 1.0, 30) == (30, 30)           # an empty span stays empty


@pytest.mark.parametrize("fps", [30, 60, 29.97, 59.94, 25, 24])
@pytest.mark.parametrize("end", [0.034, 1.0 + 0.004, 2.5101])
def test_a_sub_frame_span_stays_co_terminal_with_a_longer_one(fps, end) -> None:
    """A window shorter than one frame keeps its END frame: two elements ending at the same instant
    end on the same frame however short one of them is (reviewer case: [0.030,0.034) vs [0.0,0.034))."""
    long_ = frames.frame_range(max(0.0, end - 0.5), end, fps)
    short = frames.frame_range(end - 0.004, end, fps)
    assert short[1] == long_[1] and short[1] - short[0] == 1
    assert _window(_clause(mograph.overlay_filtergraph(
        [{"start": end - 0.004, "dur": 0.004, "glass": False}], fps=fps)[0], "o0"))[1] == long_[1] - 1
    assert frames.pts_at(30, 30) == "30/30/TB"
    assert frames.pts_at(12, 60000 / 1001) == "12012/60000/TB"    # exact rational, no float second


# --- co-terminal elements ------------------------------------------------------

@pytest.mark.parametrize("fps", [30, 60, 29.97, 59.94, 25, 24])
@pytest.mark.parametrize("start,cut_dur,mog_dur", [
    (2.0, 3.0, 3.0),
    (1.0, 4 / 60, 0.0667),      # the reported pair: one builder sees 4/60, the other its 4-decimal form
    (0.5, 0.1, 0.1),
    (0.030, 0.004, 0.004),      # sub-frame: still one frame, still ending on the rounded end frame
    (4.0 + 1 / 60, 1.5 - 1 / 60, 1.5 - 1 / 60),
])
def test_co_terminal_elements_end_on_the_same_frame(fps, start, cut_dur, mog_dur) -> None:
    """A cutaway and a mograph section over the same [start, end) open and close on the SAME output
    frames — in their own builders and inside the one merged graph the encoder runs."""
    want = frames.frame_range(start, start + cut_dur, fps)
    assert frames.frame_range(start, start + mog_dur, fps) == want

    spec = _merged_spec(fps, start=start, cut_dur=cut_dur)
    cut = _window(_clause(render.build_filtergraph(spec, gpu=False), "b0"))
    mog = _window(_clause(mograph.overlay_filtergraph(list(_layers(start, mog_dur)), fps=fps)[0], "o0"))
    assert cut == mog == (want[0], want[1] - 1)

    graph, _cmd = op.assemble(t._prepared(spec, layers=_layers(start, mog_dur)))
    assert _window(_clause(graph, "b0")) == _window(_clause(graph, "o0")) == cut
    assert _window(_clause(graph, "hw0")) == cut                 # head-below copy: same window
    assert re.search(rf"\[hb0__mog\]trim=start_frame={want[0]}:end_frame={want[1]},", graph)
    # the receipt row reads the very window the graph carries
    row = op._broll_rows(spec, graph)[0]
    assert "error" not in row and _window(row["enable"]) == cut


# --- no float-second window anywhere in the onepass graph -------------------------

@pytest.mark.parametrize("fps", [30, 60, 29.97, 59.94])
def test_the_onepass_graph_carries_no_float_second_window(fps) -> None:
    spec = _merged_spec(fps, start=2.0, cut_dur=3.0, everything=True)
    graph, _cmd = op.assemble(t._prepared(spec, layers=_layers(2.0, 3.0), flares=(0.3,)))
    assert "between(t," not in graph
    assert "between(n," not in graph                     # framesync's `n` is NOT the main frame index
    enables = re.findall(r"enable='([^']*)'", graph)
    assert len(enables) >= 15
    for e in enables:
        assert not re.search(r"\d\.\d", e), e            # every gate is integers on the grid
        assert _GATE.search(e) or re.fullmatch(rf"(lt|gte)\({re.escape(frames.index(fps))},\d+\)", e), e
    for seat in re.findall(r"setpts=PTS(?:-STARTPTS)?\+([^\[,;]*)", graph):
        assert re.fullmatch(r"\d+/\d+/TB", seat), seat   # every seat is an exact frame offset
    # a seconds-trim is only ever the SOURCE cut on an input pad; every output window trims in frames
    for m in re.finditer(r"(\[[^\]]+\])?trim=([^,;\[]*)", graph):
        if re.search(r"(^|:)(start|end|duration)=", m.group(2)):
            assert m.group(1) and re.fullmatch(r"\[\d+:v\]", m.group(1)), m.group(0)
    assert "fade=t=in:st=" not in graph and "fade=t=out:st=" not in graph


# --- the receipt's plan side ----------------------------------------------------

@pytest.mark.parametrize("fps", [30, 60])
@pytest.mark.parametrize("dur", [3.0, 2.4, 0.5, 2 / 60])
def test_receipt_frame_counts_are_unchanged_on_grid(fps, dur) -> None:
    """On the grid the frame range counts exactly what the old ceil(dur*fps) did — the receipt's plan
    side moves only where it used to disagree with the graph."""
    spec = _merged_spec(fps, start=12.0, cut_dur=dur)
    expected = op.tap_frames_expected(spec)
    assert expected["vtap0"] == max(1, math.ceil(round(dur * fps, 6)))
    a, b = render.broll_window(spec.overlays.broll_final.broll[0], fps)
    assert expected["vtap0"] == b - a
    assert f"trim=end_frame={b - a}," in render.build_filtergraph(spec, gpu=False)


# --- transitions take their frames from their two endpoints -----------------------

def _seam_spec(fps, *, start: float, cut_dur: float, tr_in: dict, tr_out: dict | None = None) -> RenderSpec:
    d = json.loads(json.dumps(t._BASE_SPEC))
    d["timeline"]["fps"] = fps
    d["inputs"].append({"id": "broll/c.jpg", "kind": "image", "sha256": SHA, "url": "https://x/c.jpg"})
    clip = {"clip": "broll/c.jpg", "start": start, "preset": "in", "dur": cut_dur, "in": 0.0,
            "transition_in": tr_in}
    if tr_out is not None:
        clip["transition_out"] = tr_out
    d["overlays"]["broll_final"] = {"broll": [clip]}
    return RenderSpec.model_validate(d)


def _mog_last(s: float, e: float, fps) -> int:
    return _window(_clause(mograph.overlay_filtergraph(
        [{"start": s, "dur": e - s, "glass": False}], fps=fps)[0], "o0"))[1]


@pytest.mark.parametrize("fps", [30, 60, 29.97, 59.94, 25, 24])
@pytest.mark.parametrize("start,cut_dur,tr_dur", [
    (0.02, 1.0, 0.02),          # the critic's case: seam [0.02,0.04) vs mograph [0,0.04) at 30 fps
    (1.0, 2.0, 0.2),
    (1.0 + 1 / 60, 1.5, 0.25),
    (2.5101, 1.3, 0.0667),
    (0.5, 1.0, 4 / 60),
])
def test_a_transition_co_terminal_with_a_mograph_window_ends_on_the_same_frame(fps, start, cut_dur,
                                                                               tr_dur) -> None:
    """A slide/push seam's gate is the frame_range of (start, start+dur) — its END frame is the one any
    other element ending at start+dur gets, never start's frame plus a separately rounded duration."""
    spec = _seam_spec(fps, start=start, cut_dur=cut_dur,
                      tr_in={"kind": "slide_wipe", "edge": "entry", "direction": "left", "dur": tr_dur},
                      tr_out={"kind": "push", "edge": "return", "direction": "up", "dur": tr_dur})
    clause = _clause(render.build_filtergraph(spec, gpu=False), "b0")
    # one nested if per seam, the return seam outermost: the gates read [return, entry]
    gates = [(int(m[1]), int(m[2])) for m in _GATE.findall(re.search(r"overlay=x='([^']*)'", clause).group(1))]
    assert len(gates) == 2
    y_gate, x_gate = gates
    end_in, end = start + tr_dur, start + cut_dur
    assert x_gate[1] == _mog_last(0.0, end_in, fps) == _mog_last(max(0.0, end_in - 0.5), end_in, fps)
    assert x_gate == (lambda r: (r[0], r[1] - 1))(frames.frame_range(start, end_in, fps))
    assert y_gate[1] == _mog_last(end - 0.5, end, fps) == _window(clause)[1]   # return seam ends with the cutaway
    assert y_gate[0] == frames.frame_range(end - tr_dur, end, fps)[0]


@pytest.mark.parametrize("fps", [30, 60, 59.94])
def test_a_dissolve_ramp_ends_on_its_endpoint_frame(fps) -> None:
    start, cut_dur, tr = 1.0 + 1 / 60, 2.0, 0.0667
    spec = _seam_spec(fps, start=start, cut_dur=cut_dur, tr_in={"kind": "dissolve", "edge": "entry", "dur": tr},
                      tr_out={"kind": "dissolve", "edge": "return", "dur": tr})
    a, b = render.broll_window(spec.overlays.broll_final.broll[0], fps)
    g = render.build_filtergraph(spec, gpu=False)
    n_in = int(re.search(r"fade=t=in:s=0:n=(\d+)", g).group(1))
    s_out, n_out = map(int, re.search(r"fade=t=out:s=(\d+):n=(\d+)", g).groups())
    assert a + n_in == frames.frame_range(start, start + tr, fps)[1]
    assert a + s_out == frames.frame_range(start + cut_dur - tr, start + cut_dur, fps)[0]
    assert a + s_out + n_out == b


# --- film_burn: the engine's precomputed slip -------------------------------------

_SLIP = [[-0.42, 0.0], [-0.3, 120.0], [-0.2, 120.0], [-0.1, -150.0], [-0.04, 0.0], [0.0, 0.0]]


def _burn_spec(slip_on: tuple[bool, bool]) -> RenderSpec:
    d = json.loads(json.dumps(t._BASE_SPEC))
    t._only_film_burn(d)
    for acc, on in zip(d["overlays"]["finalize"]["accents"], slip_on, strict=True):
        if on:
            acc["slip"] = _SLIP
    return RenderSpec.model_validate(d)


def _burn_graph(spec: RenderSpec) -> str:
    return op.assemble(t._prepared(spec, flares=(0.3,)))[0]


def test_a_spec_slip_is_rendered_verbatim_and_its_absence_falls_back() -> None:
    from podagent import accents
    grid = frames.rate(t._BASE_SPEC["timeline"]["fps"])
    grid = f"{grid.numerator}/{grid.denominator}" if grid.denominator != 1 else str(grid.numerator)
    base = _burn_graph(_burn_spec((False, False)))
    both = _burn_graph(_burn_spec((True, True)))
    first = _burn_graph(_burn_spec((True, False)))
    want = accents._pw_linear(3.5, [tuple(p) for p in _SLIP], grid)
    assert f"({want})" in both and f"({accents._pw_linear(7.0, [tuple(p) for p in _SLIP], grid)})" in both
    assert want not in base and both != base
    # the fallback is today's seeded draw, and a slip on one burn does not shift the other burn's draw
    jy = lambda g: re.search(r"crop=\d+:\d+:0:y='mod\(\((.*)\)\+\d+,\d+\)'", g).group(1)  # noqa: E731
    seeded = jy(_burn_graph(_burn_spec((False, False))))
    assert jy(first).split(")+(")[1] == seeded.split(")+(")[1]
    assert jy(first).split(")+(")[0].lstrip("(") == want


def test_slip_rides_film_burn_only_on_model_and_schema() -> None:
    from jsonschema import Draft202012Validator

    from podagent.models import SpecAccent
    with pytest.raises(ValueError, match="must not carry slip"):
        SpecAccent(kind="glitch", at=1, intensity=0.5, slip=[(-0.42, 0.0), (0.0, 0.0)])
    ok = SpecAccent(kind="film_burn", at=1, intensity=1, burn="b", clicks="c", slip=_SLIP)
    assert ok.slip == [tuple(p) for p in _SLIP]
    schema = json.loads((Path(__file__).parents[1] / "contracts" / "spec.schema.json").read_text())
    v = Draft202012Validator({"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": "#/$defs/accent"})
    good = {"kind": "film_burn", "at": 1, "intensity": 1, "burn": "b", "clicks": "c", "slip": _SLIP}
    assert not list(v.iter_errors(good))
    assert list(v.iter_errors({"kind": "glitch", "at": 1, "intensity": 0.5, "slip": _SLIP}))
    assert list(v.iter_errors({**good, "slip": [[0.0, 1.0, 2.0], [0.1, 0.0]]}))
