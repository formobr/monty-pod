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


def test_a_layer_item_carries_its_tag_and_deadline(tmp_path, monkeypatch):
    got = _capture(monkeypatch, tmp_path)
    secs = [SimpleNamespace(comp="SdLayer", start=0.0, glass=False,
                            props={"durationInSeconds": 2.0, "layer": {"id": "sd13_m_m8"}}),
            SimpleNamespace(comp="SplitScreen", start=1.0, glass=False,
                            props={"durationInSeconds": 1.0, "sdLayer": "sd13_frame_c1"}),
            SimpleNamespace(comp="Stat", start=0.0, glass=False, props={"durationInSeconds": 1.0})]
    mograph._render_layers(secs, None, {}, tmp_path)
    a, b, c = got["items"]
    assert (a["tag"], a["wallS"]) == ("sd13_m_m8", 60.0 + 60)
    assert (b["tag"], b["wallS"]) == ("sd13_frame_c1", 60.0 + 30)
    assert "tag" not in c and "wallS" not in c                 # a legacy section is untouched


def test_the_wall_names_the_layer_in_flight(tmp_path, monkeypatch):
    """Live 056 v13: the batch hung on one layer and the pod said only «ran out its 900s wall». The wall now
    reads render_batch's on-disk item log and names what was in flight, what failed, and what never started."""
    import subprocess

    spec = tmp_path / "batch.json"
    items = [{"comp": "SdLayer", "tag": t, "seqdir": str(tmp_path / t)} for t in ("a", "b", "c")]

    def hang(*_a, **_k):
        log = mograph._items_log(spec)
        log.write_text('{"tag":"a","comp":"SdLayer","start":true}\n'
                       '{"tag":"a","comp":"SdLayer","ok":false,"error":"boom"}\n'
                       '{"tag":"b","comp":"SdLayer","start":true}\n')
        raise subprocess.TimeoutExpired("node", 900)
    monkeypatch.setattr(mograph.subprocess, "run", hang)
    monkeypatch.setattr(mograph, "_oom_count", lambda: None)
    with pytest.raises(RuntimeError) as e:
        mograph._run_batch(tmp_path, items, spec, None)
    msg = str(e.value)
    assert "in flight: b" in msg and "a (boom)" in msg and "not reached: 1" in msg


def test_the_cost_receipt_does_not_clobber_the_batch_result(tmp_path, monkeypatch, capsys):
    """Regression (stand bench r3b-056-bench): the receipt loop reused `r`, the batch's CompletedProcess."""
    import subprocess

    spec = tmp_path / "batch.json"
    seq = tmp_path / "s"
    seq.mkdir()
    (seq / "element-0.png").write_bytes(b"x")

    def ran(*_a, **_k):
        mograph._items_log(spec).write_text('{"tag":"a","comp":"SdLayer","ok":true,"frames":3,"s":1.5}\n')
        return subprocess.CompletedProcess([], 0, b"", b"")
    monkeypatch.setattr(mograph.subprocess, "run", ran)
    mograph._run_batch(tmp_path, [{"comp": "SdLayer", "tag": "a", "seqdir": str(seq)}], spec, None)
    assert "layer a (SdLayer) ok 3 frames in 1.5s" in capsys.readouterr().err


def test_a_failed_layer_is_refused_by_id_even_with_leftover_frames(tmp_path, monkeypatch):
    import subprocess

    spec = tmp_path / "batch.json"
    seq = tmp_path / "s"
    seq.mkdir()
    (seq / "element-0.png").write_bytes(b"x")          # a crash-retry's leftover frame

    def ran(*_a, **_k):
        mograph._items_log(spec).write_text(
            '{"tag":"sd13_frame_c2","comp":"SplitScreen","ok":false,"error":"undefined was passed to staticFile()"}\n')
        return subprocess.CompletedProcess([], 1, b"", b"")
    monkeypatch.setattr(mograph.subprocess, "run", ran)
    monkeypatch.setattr(mograph, "_oom_count", lambda: None)
    with pytest.raises(RuntimeError, match=r"sd13_frame_c2 \(undefined was passed to staticFile\(\)\)"):
        mograph._run_batch(tmp_path, [{"comp": "SplitScreen", "tag": "sd13_frame_c2", "seqdir": str(seq)}], spec, None)
