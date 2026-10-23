"""A grid that is not the integer comp raster crosses the op seam EXACTLY (MISC-578).

A 29.97 source used to cross as a float (cut.apply/cut.audio `fps_grid`, camera.apply `geom.fps`) or not
at all (media.scale/mograph.render `fps` are integers, so it truncated to 29 or rounded to 30). The five
grid params now also take a "num/den" string. Params are the only schema-validated surface and
registry.validate_params is the same call on both sides, so these checks are the seam itself.

The ingest CFR directive (media.normalize.fps), the opener's comp raster (opener.build.comp_fps) and the
analysis sampling rate (measure.framediff.fps) are deliberately NOT widened.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from podagent.ops import registry

pytest.importorskip("jsonschema")

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"

_GEOM = {"crop_x": 0, "crop_y": 0, "sw": 2160, "sh": 3840, "cover_w": 2160, "cover_h": 3840,
         "ax": 0.5, "ay": 0.42, "fps": 30.0}
_ZOOM_KF = [{"t": 0.0, "z": 1.1, "head_x": 0.5, "eye_y": 0.42, "interp": "cos"}]


def _cut(fps):
    return {"keep": [[0.0, 2.0], [3.5, 5.75]], "fps_grid": fps}


def _scale(fps):
    return {"height": 960, "encode_profile": "proxy", "fps": fps}


def _mograph(fps):
    return {"comp": "Statement", "fps": fps}


def _camera(fps):
    return {"keyframes": _ZOOM_KF, "geom": {**_GEOM, "fps": fps},
            "out_w": 1080, "out_h": 1920, "encode_profile": "final"}


# op name -> builder that places a value at that op's grid param
WIDENED = {
    "cut.apply": _cut,
    "cut.audio": _cut,
    "media.scale": _scale,
    "mograph.render": _mograph,
    "camera.apply": _camera,
}

# today's valid numbers per op, at and inside each branch's unchanged bounds
TODAYS_NUMBERS = {
    "cut.apply": [30.0, 29.97, 23.976, 240, 0.5],
    "cut.audio": [30.0, 29.97, 25, 240],
    "media.scale": [1, 24, 30, 240],
    "mograph.render": [1, 30, 60, 120],
    "camera.apply": [30.0, 29.97, 1000.0, 0.01],
}

# numbers refused today, still refused: the number branch's bounds did not move
TODAYS_REFUSED_NUMBERS = {
    "cut.apply": [0, -1, 240.5],
    "cut.audio": [0, 241],
    "media.scale": [0, 241, 29.97],
    "mograph.render": [0, 121, 29.97],
    "camera.apply": [0, -30.0],
}

MALFORMED = ["29.97", "30", "30000/0", "0/1001", "030000/1001", "30000/01", "-30000/1001",
             "30000 / 1001", " 30000/1001", "30000/1001 ", "30000/1001/1", "/1001", "30000/",
             "1e3/1", ""]


@pytest.mark.parametrize("name", sorted(WIDENED))
@pytest.mark.parametrize("rational", ["30000/1001", "24000/1001", "60000/1001", "25/1", "30/1"])
def test_the_widened_params_accept_a_rational_string(name, rational):
    registry.validate_params(name, WIDENED[name](rational))


@pytest.mark.parametrize("name", sorted(WIDENED))
def test_the_widened_params_still_accept_todays_numbers(name):
    for value in TODAYS_NUMBERS[name]:
        registry.validate_params(name, WIDENED[name](value))
    for value in TODAYS_REFUSED_NUMBERS[name]:
        with pytest.raises(registry.OpError, match="invalid params"):
            registry.validate_params(name, WIDENED[name](value))


def test_the_rational_golden_validates_at_the_op_seam():
    """The golden that carries a rational grid must also pass the per-op params check, not only the
    envelope schema, or it is a golden for a request the pod would refuse."""
    doc = json.loads((CONTRACTS / "examples" / "pod_job.ops-rational.json").read_text())
    steps = doc["chain"]["steps"]
    assert any(isinstance(s["params"].get("fps"), str) for s in steps)
    for step in steps:
        registry.validate_params(step["op"], step["params"])


@pytest.mark.parametrize("name", sorted(WIDENED))
@pytest.mark.parametrize("bad", MALFORMED)
def test_malformed_rationals_are_refused(name, bad):
    with pytest.raises(registry.OpError, match="invalid params"):
        registry.validate_params(name, WIDENED[name](bad))


def test_the_unwidened_params_still_refuse_a_string():
    from jsonschema import Draft202012Validator
    for name, key in (("media.normalize", "fps"), ("opener.build", "comp_fps"), ("measure.framediff", "fps")):
        spec = registry.get(name).params_schema["properties"][key]
        assert "anyOf" not in spec and spec.get("type") in ("integer", "number"), \
            f"{name}.{key} was widened; it must stay a number"
        assert not Draft202012Validator(spec).is_valid("30000/1001"), f"{name}.{key} took a rational"


def test_the_widened_contract_versions_moved():
    want = {"cut.apply": 3, "cut.audio": 2, "media.scale": 5, "mograph.render": 2, "camera.apply": 2}
    got = {name: json.loads((CONTRACTS / "ops" / f"{name}.json").read_text())["version"] for name in want}
    assert got == want
    assert {name: registry.get(name).version for name in want} == want
