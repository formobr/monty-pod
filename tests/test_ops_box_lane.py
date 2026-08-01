"""The two box-lane ops, pod side: `measure.framediff` (the camera snap's decode) and `media.scale`'s
`height_mode` (the no-upscale clamp). The control-plane box may hold no media byte, so both must cross as
NUMBERS and GEOMETRY — never a threshold going out, never a picture coming home. Each test is a REFUSAL."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from podagent.models import OpChain
from podagent.ops import registry

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
_PACK = {"url": "https://x/p.tar", "sha256": "a" * 64, "size": 10}
_GOOD = {"boundaries": [120, 300], "span": 39, "fps": 30.0, "frames": 900}


def _decl(op: str) -> dict:
    return json.loads((CONTRACTS / "ops" / f"{op}.json").read_text())


# ── measure.framediff: numbers cross, the snap rule does not ─────────────────────────────────────

def test_the_declaration_is_honest():
    d = _decl("measure.framediff")
    assert d["op"] == "measure.framediff" and d["judgement"] is False
    assert d["needs"] == ["video_encode"]
    assert d["parity"]["mode"] == "numeric" and d["parity"]["tol"] == 0.5
    assert [p["kind"] for p in d["outputs"]] == ["json"], "a diff pass that returns pixels is the old failure"
    assert d["handler"] == "montyops.measure_framediff:run"


def test_no_snap_rule_is_in_the_params():
    """DISCLOSURE, the same line media.filmstrip draws: `boundaries`/`span` are geometry — WHERE to look.
    A floor, a K, a dominance ratio or a `snap` flag here would put a piece of the camera plan on a pod."""
    props = set(_decl("measure.framediff")["params"]["properties"])
    assert props == {"boundaries", "span", "fps", "frames"}
    forbidden = {"floor", "threshold", "k", "mad_k", "ratio", "snap", "window", "min_spike", "dominance"}
    assert not (forbidden & props)


def test_params_are_validated_against_the_declaration():
    pytest.importorskip("jsonschema")
    registry.validate_params("measure.framediff", _GOOD)
    with pytest.raises(registry.OpError):                      # closed schema — a threshold cannot sneak in
        registry.validate_params("measure.framediff", {**_GOOD, "floor": 7.0})
    with pytest.raises(registry.OpError):                      # the grid is not optional: it names the frames
        registry.validate_params("measure.framediff", {k: v for k, v in _GOOD.items() if k != "fps"})
    with pytest.raises(registry.OpError):                      # a zero span returns nothing to argmax over
        registry.validate_params("measure.framediff", {**_GOOD, "span": 0})
    with pytest.raises(registry.OpError):                      # a boundary is a frame INDEX, not a time
        registry.validate_params("measure.framediff", {**_GOOD, "boundaries": [1.5]})


def test_the_op_is_pod_safe_and_reaches_preflight():
    """The gate must not be a wall: a chain naming the new op passes on an image that declares it, which is
    the whole difference between shipping the op and bumping the pin."""
    from podagent.ops import runner

    registry.assert_pod_safe("measure.framediff")
    runner.preflight_chain(OpChain(job_id="j", pack=_PACK, steps=[{
        "id": "fd", "op": "measure.framediff", "needs": [], "params": _GOOD,
        "inputs": [{"port": "src", "url": "https://x/in.mp4"}],
        "outputs": [{"port": "measured", "url": "https://x/fd.json"}]}]))


# ── media.scale: the clamp is ADDITIVE, and that is asymmetric ───────────────────────────────────

_SCALE = {"height": 960, "encode_profile": "proxy"}


def test_todays_media_scale_payload_still_validates_untouched():
    """THE additive proof on this side: an engine that predates `height_mode` sends exactly what it always
    sent, and this image must take it. A `required` on the new key would break every old caller at once."""
    pytest.importorskip("jsonschema")
    registry.validate_params("media.scale", _SCALE)
    assert "height_mode" not in _decl("media.scale")["params"].get("required", [])
    assert _decl("media.scale")["params"]["properties"]["height_mode"]["default"] == "exact"


def test_the_clamp_is_a_closed_enum():
    pytest.importorskip("jsonschema")
    registry.validate_params("media.scale", {**_SCALE, "height_mode": "at_most"})
    registry.validate_params("media.scale", {**_SCALE, "height_mode": "exact"})
    with pytest.raises(registry.OpError):
        registry.validate_params("media.scale", {**_SCALE, "height_mode": "min(h,ih)"})


def test_an_unknown_param_is_refused_whole_which_is_why_the_engine_gates():
    """THE ASYMMETRY, stated where it is enforced: this is what an image PREDATING `height_mode` does with a
    payload carrying it. Additive is free only in the pod→engine direction, so the engine withholds the
    field below its minimum pin (op_chains.OPS_MIN_IMAGE)."""
    pytest.importorskip("jsonschema")
    with pytest.raises(registry.OpError, match="invalid params"):
        registry.validate_params("media.scale", {**_SCALE, "no_upscale": True})


def test_the_op_version_moved_but_the_envelope_did_not():
    """A param added to an existing op bumps THAT op; `contracts/VERSION` pins the envelope and stays put —
    the split that makes a new tool cost two files instead of a release."""
    assert _decl("media.scale")["version"] == 2
    assert (CONTRACTS / "VERSION").read_text().strip() == "5"
