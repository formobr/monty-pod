"""camera.apply's params schema is the RUNTIME VALIDATOR on both sides (MISC-141 fold round 2, H3):
podagent.ops.registry.validate_params runs the SAME contracts/ops/camera.apply.json fragment the control
plane checks before dispatch, so a payload the schema admits is one the pod admits too. Round 1 forgot to
widen this schema for the fold's param shape (top-level `geom`, keyframes carrying z/head_x/eye_y/interp or
t/bump, no top-level `interp`) — every post-fold camera.apply call died with `'interp' is a required
property` before this fix. These tests are pure contract-level checks (no engine import); the cross-boundary
check that `scripts.op_chains.camera_params`'s REAL output validates is
tests/test_camera_path_smoothness.py::test_camera_apply_validate_params_accepts_the_real_fold_trajectory
(engine side, since it needs head_trajectory to build a real trajectory.json).
"""
from __future__ import annotations

import pytest

from podagent.ops import registry

_GEOM = {"crop_x": 0, "crop_y": 0, "sw": 2160, "sh": 3840, "cover_w": 2160, "cover_h": 3840,
        "ax": 0.5, "ay": 0.42, "fps": 30.0}
_BASE = {"out_w": 1080, "out_h": 1920, "encode_profile": "final"}


def _legacy(**overrides):
    params = {"keyframes": [{"t": 0.0, "rect": [0.1, 0.2, 0.5, 0.6]},
                            {"t": 1.0, "rect": [0.11, 0.2, 0.49, 0.6]}],
              "interp": "linear", **_BASE}
    params.update(overrides)
    return params


def _zoom(keyframes=None, **overrides):
    params = {"keyframes": keyframes if keyframes is not None else [
        {"t": 0.0, "z": 1.1, "head_x": 0.5, "eye_y": 0.42, "interp": "cos"},
        {"t": 2.0, "z": 1.3, "head_x": 0.5, "eye_y": 0.42, "interp": "cos"},
    ], "geom": _GEOM, **_BASE}
    params.update(overrides)
    return params


def test_the_legacy_rect_shape_still_validates():
    """A trajectory.json persisted before MISC-141 (no `geom`, one top-level `interp`, keyframes carry
    `t`/`rect`) validates unchanged — the fold's oneOf must not narrow the pre-fold shape."""
    registry.validate_params("camera.apply", _legacy())


def test_the_zoom_shape_with_ramp_and_bump_keyframes_validates():
    """MISC-141 fold round 2's real shape: ramp keyframes (t/z/head_x/eye_y/interp) mixed with the punch's
    own bump keyframes (t/bump only) in the SAME "keyframes" array, `geom` carrying `fps`."""
    kfs = [
        {"t": 0.0, "z": 1.1, "head_x": 0.5, "eye_y": 0.42, "interp": "cos"},
        {"t": 2.0, "z": 1.3, "head_x": 0.5, "eye_y": 0.42, "interp": "cos"},
        {"t": 0.5, "bump": 0.0},
        {"t": 0.8, "bump": 0.35},
        {"t": 1.2, "bump": 0.0},
    ]
    registry.validate_params("camera.apply", _zoom(keyframes=kfs))


def test_a_ramp_keyframe_missing_interp_is_refused():
    kfs = [{"t": 0.0, "z": 1.1, "head_x": 0.5, "eye_y": 0.42}]     # no "interp"
    with pytest.raises(registry.OpError):
        registry.validate_params("camera.apply", _zoom(keyframes=kfs))


def test_a_ramp_keyframe_with_an_unknown_key_is_refused():
    kfs = [{"t": 0.0, "z": 1.1, "head_x": 0.5, "eye_y": 0.42, "interp": "cos", "bump": 0.1}]
    with pytest.raises(registry.OpError):
        registry.validate_params("camera.apply", _zoom(keyframes=kfs))


def test_a_bump_keyframe_with_an_unknown_key_is_refused():
    kfs = [{"t": 0.0, "z": 1.1, "head_x": 0.5, "eye_y": 0.42, "interp": "cos"},
          {"t": 0.5, "bump": 0.3, "interp": "linear"}]             # bump never carries interp
    with pytest.raises(registry.OpError):
        registry.validate_params("camera.apply", _zoom(keyframes=kfs))


def test_a_rect_keyframe_in_the_zoom_arm_is_refused():
    """A legacy rect item cannot sneak into a `geom`-bearing payload — every keyframe must be a ramp or a
    bump shape once `geom` is present, since a rect item has no `z` for camera.apply to compose with."""
    kfs = [{"t": 0.0, "rect": [0.1, 0.1, 0.5, 0.6]}]
    with pytest.raises(registry.OpError):
        registry.validate_params("camera.apply", _zoom(keyframes=kfs))


def test_geom_and_top_level_interp_together_is_refused():
    """The zoom shape carries `interp` PER KEYFRAME — a payload with BOTH `geom` and a top-level `interp`
    matches neither oneOf branch."""
    with pytest.raises(registry.OpError):
        registry.validate_params("camera.apply", _zoom(interp="linear"))


def test_geom_missing_fps_is_refused():
    """`geom.fps` (MISC-141 fold round 2) is what the chunk splitter snaps a trim boundary to a real frame
    with — an old `geom` without it must not silently validate."""
    geom = dict(_GEOM)
    del geom["fps"]
    with pytest.raises(registry.OpError):
        registry.validate_params("camera.apply", _zoom(geom=geom))


def test_a_single_ramp_keyframe_still_validates():
    """A degenerate one-keyframe zoom trajectory (a single-frame span) is still a valid ramp shape."""
    registry.validate_params("camera.apply", _zoom(keyframes=[
        {"t": 0.0, "z": 1.1, "head_x": 0.5, "eye_y": 0.42, "interp": "cos"},
    ]))
