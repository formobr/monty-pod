"""super-director v13 layout layers on the pod (podagent/mograph.py `sdFace`): the head plate is staged ONCE into the
bundle and handed to every `SdLayer` section that asks for it; nothing changes for a section that does not."""
from types import SimpleNamespace

import pytest

from podagent import mograph


def _capture(monkeypatch, tmp_path):
    got: dict = {}
    monkeypatch.setattr(mograph, "remotion_dir", lambda *_a: tmp_path / "rd")
    monkeypatch.setattr(mograph, "_run_batch", lambda _rd, items, *_a: got.setdefault("items", items))
    monkeypatch.setattr(mograph, "_pack", lambda metas, *_a: metas)
    (tmp_path / "rd").mkdir()
    return got


def test_sd_face_props_is_identity_without_the_request():
    props = {"layer": {"kind": "library"}, "durationInSeconds": 2.0}
    assert mograph.sd_face_props(props, 4.0, None) is props


def test_sd_face_props_swaps_the_request_for_the_staged_plate():
    out = mograph.sd_face_props({"sdFace": "base", "src": "_photo/a.mp4"}, 6.5, "_sd/base.mp4")
    assert out == {"src": "_photo/a.mp4", "faceSrc": "_sd/base.mp4", "faceStart": 6.5}


def test_a_face_request_with_no_plate_is_refused_by_name():
    with pytest.raises(RuntimeError, match="sdFace=base"):
        mograph.sd_face_props({"sdFace": "base"}, 1.0, None)


def test_render_layers_stages_the_plate_once_for_every_layout(tmp_path, monkeypatch):
    got = _capture(monkeypatch, tmp_path)
    base = tmp_path / "head_dyn.mp4"
    base.write_bytes(b"plate")
    secs = [SimpleNamespace(comp="SdLayer", start=s, glass=False,
                            props={"sdFace": "base", "src": "_photo/a.mp4", "durationInSeconds": 2.0})
            for s in (1.0, 5.0)]
    mograph._render_layers(secs, None, {}, tmp_path, base_path=base)
    staged = tmp_path / "rd" / "public" / "_sd" / "base.mp4"
    assert staged.read_bytes() == b"plate"
    assert [(i["props"]["faceSrc"], i["props"]["faceStart"]) for i in got["items"]] == [
        ("_sd/base.mp4", 1.0), ("_sd/base.mp4", 5.0)]
    assert all("sdFace" not in i["props"] for i in got["items"])


def test_a_long_layer_carries_its_own_ceiling(tmp_path, monkeypatch):
    got = _capture(monkeypatch, tmp_path)
    secs = [SimpleNamespace(comp="SdLayer", start=0.0, glass=False, props={"durationInSeconds": 300.0}),
            SimpleNamespace(comp="SdLayer", start=0.0, glass=False, props={"durationInSeconds": 3.0}),
            SimpleNamespace(comp="Stat", start=0.0, glass=False, props={"durationInSeconds": 300.0})]
    mograph._render_layers(secs, None, {}, tmp_path)
    assert got["items"][0]["maxSeconds"] == 302.0
    assert got["items"][1]["maxSeconds"] == 120.0           # never under render_batch's own default
    assert "maxSeconds" not in got["items"][2]               # a legacy comp is untouched


def test_no_face_request_stages_nothing(tmp_path, monkeypatch):
    _capture(monkeypatch, tmp_path)
    base = tmp_path / "head_dyn.mp4"
    base.write_bytes(b"plate")
    mograph._render_layers([SimpleNamespace(comp="Stat", start=0.0, glass=False, props={})], None, {}, tmp_path,
                           base_path=base)
    assert not (tmp_path / "rd" / "public" / "_sd").exists()
