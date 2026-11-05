"""Every probed field of a STUB video op's dry placeholder equals what its `_VIDEO_SHAPE` row resolves to
(MISC-972): the engine plans on the rendered video's clock, so a placeholder that differs from the live
output in ANY of width/height/SAR/rotation/rate/duration makes a dry replay diverge from its live record."""
from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from podagent.ops import dry, registry

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not on this runner")

_SRC_DUR = 3.0

# Minimal valid params per op (the fixture a NEW stub video op must add; a missing one fails by name).
_PARAMS: dict[str, dict[str, Any]] = {
    "cut.apply": {"keep": [[0.0, 1.0], [1.5, 2.5]], "fps_grid": "30000/1001", "max_h": 180},
    "camera.apply": {"keyframes": [{"t": 0, "rect": [0, 0, 1, 1]}], "interp": "linear",
                     "out_w": 320, "out_h": 180, "encode_profile": "preview"},
    "edit.splice": {"spans": [[0.5, 1.0]]},
    "edit.weld": {"master_lufs": -14.0, "master_tp": -1.0},
    "media.cut_proxy": {"max_h": 180},
    "media.fetch": {"url": "https://example.com/a.mp4"},
    "media.normalize": {"width": 320, "height": 180, "fps": 30},
    "media.scale": {"height": 180, "encode_profile": "proxy", "fps": "25"},
    "mograph.render": {"comp": "Statement", "fps": "30000/1001"},
    "motion.kenburns": {"preset": "in", "scale": "cover", "encode_profile": "preview"},
    "opener.build": {"format": "question", "dur": 1.0, "w": 320, "h": 180, "fps": "30000/1001"},
}


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60).stdout


def _make_source(dst: Path, *, rate: str, sar: str, rotation: int, size: str = "640x360") -> Path:
    base = dst.with_name("base_" + dst.name)
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
          "-f", "lavfi", "-i", f"color=c=blue:s={size}:r={rate}:d={_SRC_DUR},setsar={sar}",
          "-f", "lavfi", "-i", f"sine=f=440:r=48000:d={_SRC_DUR}", "-c:a", "aac",
          "-c:v", "libx264", "-preset", "ultrafast", "-shortest", str(base if rotation else dst)])
    if rotation:
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-display_rotation", str(rotation),
              "-i", str(base), "-c", "copy", str(dst)])
    return dst


def _probe(path: Path) -> dict[str, Any]:
    """width/height AS DECODED (swapped for a 90/270 display matrix), sar, rotation, exact rate, duration."""
    doc = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height,sample_aspect_ratio,r_frame_rate:stream_tags=rotate"
                           ":stream_side_data=rotation:format=duration", "-of", "json", str(path)]))
    st = doc["streams"][0]
    rot = 0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    rot = rot or int(st.get("tags", {}).get("rotate", 0))
    w, h = (st["height"], st["width"]) if abs(rot) % 180 == 90 else (st["width"], st["height"])
    n, _, d = str(st.get("sample_aspect_ratio") or "1:1").partition(":")
    sar = Fraction(int(n), int(d)) if n.isdigit() and d.isdigit() and int(n) and int(d) else Fraction(1)
    return {"w": w, "h": h, "sar": sar, "rot": rot, "rate": Fraction(st["r_frame_rate"]),
            "dur": float(doc["format"]["duration"])}


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("srcs")
    plain = _make_source(d / "plain.mp4", rate="30000/1001", sar="1/1", rotation=0)
    sar = _make_source(d / "sar.mp4", rate="25", sar="4/3", rotation=0)   # non-square pixels, always exercised
    rot = _make_source(d / "rot.mp4", rate="25", sar="1/1", rotation=90)
    if _probe(rot)["rot"] == 0:
        rot = None   # this ffmpeg build cannot write a display matrix: only the rotation case is skipped
    return {"plain": plain, "sar": sar, "rot": rot}


def _stub_video_ops() -> list[str]:
    return sorted(name for name, (mode, _why) in dry._CLASSIFICATION.items()
                  if mode == dry.STUB and any(p.kind == "video" for p in registry.get(name).outputs))


def _expected(row: dry.Shape, op_name: str, params: dict[str, Any], src: dict[str, Any] | None) -> dict[str, Any]:
    """Each field from its row token, computed here from the source's probe / the params — not by dry.py."""
    def tok(field: str) -> tuple[str, str]:
        kind, _, arg = getattr(row, field).partition(":")
        return kind, arg

    out: dict[str, Any] = {}
    disp_h = src["h"] if src else None
    kind, arg = tok("height")
    if kind == "source":
        out["h"] = disp_h
    elif kind == "param":
        out["h"] = params[arg]
    elif kind == "cap":
        out["h"] = min(disp_h, params[arg]) if arg in params else disp_h
    elif kind == "scale":
        out["h"] = params[arg]
    else:
        out["h"] = 64
    kind, arg = tok("width")
    if kind == "source":
        out["w"] = src["w"]
    elif kind == "param":
        out["w"] = params[arg]
    elif kind == "follow_height":
        # ffmpeg's -2 rounds the width to the nearest even with halves UP
        out["w"] = src["w"] if out["h"] == disp_h else 2 * int(Fraction(src["w"] * out["h"], 2 * disp_h) + Fraction(1, 2))
    else:
        out["w"] = 64
    out["sar"] = Fraction(1)
    if row.sar == "source":
        # the live op autorotates (a transpose), which inverts the pixel aspect
        out["sar"] = 1 / src["sar"] if abs(src["rot"]) % 180 == 90 else src["sar"]
    assert row.rotation == "baked"
    out["rot"] = 0
    kind, arg = tok("rate")
    if kind == "source":
        out["rate"] = src["rate"]
    elif kind == "param":
        out["rate"] = Fraction(params[arg])
    elif kind == "param?":
        out["rate"] = Fraction(params[arg]) if arg in params else src["rate"]
    else:
        out["rate"] = None
    return out


def _bind(op: registry.Op, src: Path | None, tmp: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    inputs: dict[str, Path] = {}
    if src is not None:
        for port in op.inputs:
            if port.kind == "video":
                inputs[port.id] = src
    outputs = {p.id: tmp / f"{p.id}{'.json' if p.kind == 'json' else '.mp4'}" for p in op.outputs}
    return inputs, outputs


@pytest.mark.parametrize("kind", ["plain", "sar", "rot"])
def test_every_stub_video_op_placeholder_equals_its_shape_row(kind, sources, tmp_path):
    src_path = sources[kind]
    if src_path is None:
        pytest.skip("no display-matrix writer in this ffmpeg")
    ops = _stub_video_ops()
    assert "cut.apply" in ops and "camera.apply" in ops   # the iteration reads the classification
    for name in ops:
        assert name in _PARAMS, f"{name}: add minimal params for the mirror test"
        op, params = registry.get(name), _PARAMS[name]
        row = dry._VIDEO_SHAPE[name]
        has_video_in = any(p.kind == "video" for p in op.inputs)
        work = tmp_path / f"{kind}_{name}"
        inputs, outputs = _bind(op, src_path if has_video_in else None, work)
        if row.source_port == "main":   # a different, plain "cold" proves the row reads `main`
            inputs["cold"] = sources["plain"] if kind != "plain" else sources["sar"]
        dry.resolve(op)(params=params, inputs=inputs, outputs=outputs)
        video = next(p for p in op.outputs if p.kind == "video")
        got = _probe(outputs[video.id])
        main_src = inputs.get(row.source_port or "") or (src_path if has_video_in else None)
        src = _probe(main_src) if main_src is not None else None
        want = _expected(row, name, params, src)
        for field in ("w", "h", "sar", "rot"):
            assert got[field] == want[field], f"{name} [{kind}] {field}: {got[field]} != {want[field]}"
        if want["rate"] is not None:
            assert got["rate"] == want["rate"], f"{name} [{kind}] rate"
        pinned = dry._VIDEO_DUR.get(name)
        want_dur = pinned(params) if pinned else (_SRC_DUR if inputs else 1.0)
        tol = 1.0 / float(got["rate"]) + (0.0 if pinned else 0.1)
        assert abs(got["dur"] - want_dur) <= tol, f"{name} [{kind}] duration {got['dur']} vs {want_dur}"


@pytest.mark.parametrize("name", ["cut.apply", "media.cut_proxy"])
def test_a_9x16_source_capped_at_720_rounds_the_width_half_up(name, tmp_path):
    # 1080*720/1920 = 405 -> ffmpeg's scale=-2:720 gives 406 (halves up), Python's round() would give 404
    src = _make_source(tmp_path / "v.mp4", rate="30000/1001", sar="1/1", rotation=0, size="1080x1920")
    op = registry.get(name)
    inputs, outputs = _bind(op, src, tmp_path / name)
    params = dict(_PARAMS[name], max_h=720)
    dry.resolve(op)(params=params, inputs=inputs, outputs=outputs)
    got = _probe(outputs["dst"])
    assert (got["w"], got["h"]) == (406, 720)


def test_a_stub_video_op_without_a_shape_row_is_refused_naming_the_table(monkeypatch):
    monkeypatch.delitem(dry._VIDEO_SHAPE, "camera.apply")
    with pytest.raises(registry.OpError, match="_VIDEO_SHAPE"):
        dry.resolve(registry.get("camera.apply"))


def test_a_param_that_sets_a_field_but_is_absent_is_a_named_refusal(sources, tmp_path):
    op = registry.get("camera.apply")
    inputs, outputs = _bind(op, sources["plain"], tmp_path)
    with pytest.raises(dry.DryStubUnreadableShape, match="out_w"):
        dry.resolve(op)(params={"out_h": 180}, inputs=inputs, outputs=outputs)


def test_no_bound_video_input_keeps_the_64x64_fallback(tmp_path):
    op = registry.get("cut.apply")
    inputs, outputs = _bind(op, None, tmp_path)
    dry.resolve(op)(params=_PARAMS["cut.apply"], inputs=inputs, outputs=outputs)
    got = _probe(outputs["dst"])
    assert (got["w"], got["h"], got["sar"], got["rot"]) == (64, 64, Fraction(1), 0)
    assert got["rate"] == 30
