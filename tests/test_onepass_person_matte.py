"""super-director v13 R4 — the person plane in the one-pass graph (spec overlays.person_matte): sections with layer z < 3
→ the plate's own pixels under the RVM alpha, cut through the base's trims/speed → sections z ≥ 3 and legacy ones;
gated to the windows minus b-roll cutaways; refused by name when the alpha is missing or short."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from podagent import mograph, render_onepass as op
from podagent.models import RenderSpec

from test_render_onepass import _BASE_SPEC, _prepared, assert_connected, _MASTER_MAPS, _REF_MAPS  # noqa: E402

SHA = "0" * 64
LAYERS = (
    {"mov": "/w/behind.mov", "start": 2.0, "dur": 3.0, "glass": False, "z": 1},
    {"mov": "/w/legacy.mov", "start": 2.5, "dur": 1.0, "glass": False, "z": None},
    {"mov": "/w/over.mov", "start": 3.0, "dur": 1.0, "glass": False, "z": 4},
)


def _spec(windows=((2.0, 5.0),), broll=None) -> RenderSpec:
    d = json.loads(json.dumps(_BASE_SPEC))
    d["inputs"].append({"id": "person_matte_alpha", "kind": "video", "sha256": SHA, "url": "https://x/a.mkv"})
    d["overlays"]["person_matte"] = {"alpha": "person_matte_alpha", "windows": [list(w) for w in windows]}
    if broll:
        d["inputs"].append({"id": "broll/c.mp4", "kind": "video", "sha256": SHA, "url": "https://x/c.mp4"})
        d["overlays"]["broll_final"] = {"broll": broll}
    return RenderSpec.model_validate(d)


def test_person_plane_sits_between_the_behind_layers_and_the_rest():
    graph, _cmd = op.assemble(_prepared(_spec(), layers=LAYERS))
    chains = graph.split(";")
    i_behind = next(i for i, c in enumerate(chains) if "vmogbelow" in c and "overlay" in c)
    i_person = next(i for i, c in enumerate(chains) if "[vpersonplane]overlay" in c)
    i_over = next(i for i, c in enumerate(chains) if c.startswith("[vperson]") or "[vperson][" in c)
    assert i_behind < i_person < i_over
    assert any("alphamerge" in c for c in chains)
    # the plate is cut by the base's own segments, spelled the same way: 0-6 at 1, 10-15 at 1.25
    assert any("trim=start=0:end=6,setpts=(PTS-STARTPTS)/1," in c for c in chains)
    assert any("trim=start=10:end=15,setpts=(PTS-STARTPTS)/1.25," in c for c in chains)
    # [2, 5) at 30 fps → frames 60..149
    assert "between(" in chains[i_person] and ",60,149)" in chains[i_person]
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


def test_no_person_matte_keeps_the_old_order():
    from test_render_onepass import _spec as base_spec
    graph, _ = op.assemble(_prepared(base_spec(), layers=LAYERS))
    assert "alphamerge" not in graph and "vperson" not in graph


def test_a_cutaway_gates_the_person_plane_off():
    sp = _spec(windows=((2.0, 8.0),), broll=[{"clip": "broll/c.mp4", "start": 4.0, "dur": 2.0, "preset": "in"}])
    assert op.person_matte_windows(sp) == [(2.0, 4.0), (6.0, 8.0)]
    assert mograph.person_windows([(0.0, 1.0)], [(0.0, 1.0)]) == []


def test_split_by_person():
    below, above = mograph.split_by_person(list(LAYERS))
    assert [x["mov"] for x in below] == ["/w/behind.mov"]
    assert [x["mov"] for x in above] == ["/w/legacy.mov", "/w/over.mov"]
    assert mograph.sd_layer_z({"layer": {"z": 1}}) == 1 and mograph.sd_layer_z({"sdZ": 4}) == 4
    assert mograph.sd_layer_z({"text": "x"}) is None


def test_spec_refuses_an_alpha_that_is_not_an_input_and_a_two_plate_timeline():
    d = json.loads(json.dumps(_BASE_SPEC))
    d["overlays"]["person_matte"] = {"alpha": "nope", "windows": [[1, 2]]}
    with pytest.raises(ValueError, match="is not an inputs"):
        RenderSpec.model_validate(d)
    d["inputs"].append({"id": "nope", "kind": "video", "sha256": SHA, "url": "https://x/n.mkv"})
    d["timeline"]["segments"][1]["src"] = "nope"
    with pytest.raises(ValueError, match="ONE plate"):
        RenderSpec.model_validate(d)


def _clip(path: Path, frames: int, fmt: str) -> None:
    enc = ["-c:v", "ffv1", "-pix_fmt", "gray"] if fmt == "gray" else ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=30",
                    "-frames:v", str(frames), *enc, str(path)], check=True)


def test_missing_or_short_alpha_is_refused_by_name(tmp_path):
    sp = _spec()
    plate, alpha = tmp_path / "plate.mp4", tmp_path / "alpha.mkv"
    _clip(plate, 30, "yuv")
    paths = {"base": plate, "person_matte_alpha": tmp_path / "missing.mkv"}
    with pytest.raises(RuntimeError, match="alpha input 'person_matte_alpha' is missing"):
        op._check_person_matte(sp, paths)
    _clip(alpha, 29, "gray")
    paths["person_matte_alpha"] = alpha
    with pytest.raises(RuntimeError, match="has 29 frames, the plate 'base' has 30"):
        op._check_person_matte(sp, paths)
    _clip(alpha, 30, "gray")
    op._check_person_matte(sp, paths)
