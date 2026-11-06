"""The cutout infer kind (docs/research/pod-cutout-birefnet.md): a request on the pod yields the benched RGBA cut or a NAMED
refusal — never a silent CPU run, never a pass-through reported as a success. The model is a tiny scripted stand-in that
returns a fixed map (no weights, no GPU); the pixel pipeline, the files PUT and the receipt are the real code."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from podagent import infer_cutout as ic
from podagent.models import CutoutParams

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ("2.11.0+cu128", "0.26.0+cu128")
SHA = "c" * 64
MAP = 8     # the stand-in's input / output size


def _manifest(**over) -> dict:
    m = {"hf_repo": "ZhengPeng7/BiRefNet_dynamic", "hf_revision": "280306042f57b7a33854319da62fd86aaa89ec4c",
         "input": [1, 3, MAP, MAP], "dtype": "float32", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
         "torch": RUNTIME[0], "torchvision": RUNTIME[1]}
    m.update(over)
    return m


def _write_weights(tmp_path: Path, fill, **manifest_over) -> Path:
    """A weights directory the way the build tool ships it: config.json manifest + ONE traced .pt (a scripted stand-in
    whose output is `fill`, an [8,8] map, whatever the picture is)."""
    torch = pytest.importorskip("torch")

    class Fixed(torch.nn.Module):
        def __init__(self, grid):
            super().__init__()
            self.register_buffer("grid", grid)

        def forward(self, x):
            return self.grid

    wdir = tmp_path / "weights"
    wdir.mkdir()
    (wdir / "config.json").write_text(json.dumps(_manifest(**manifest_over)))
    torch.jit.save(torch.jit.script(Fixed(torch.as_tensor(fill, dtype=torch.float32)[None, None].clone())),
                   str(wdir / "birefnet.pt"))
    return wdir


def _block() -> list[list[float]]:
    """The subject: a 4x4 block in the middle of the 8x8 map."""
    return [[1.0 if 2 <= y < 6 and 2 <= x < 6 else 0.0 for x in range(MAP)] for y in range(MAP)]


def _service(tmp_path: Path, fill, **manifest_over) -> ic.CutoutService:
    return ic.CutoutService("stand-in", _write_weights(tmp_path, fill, **manifest_over), SHA, device="cpu", runtime=RUNTIME)


def _params(tmp_path: Path, **srcs: Path) -> tuple[CutoutParams, dict[str, Path]]:
    outs = {k: tmp_path / "out" / f"{k}.png" for k in srcs}
    params = CutoutParams.model_validate({"items": [
        {"id": k, "src_url": p.as_uri(), "out_url": outs[k].as_uri()} for k, p in srcs.items()]})
    return params, outs


def _run(tmp_path: Path, svc: ic.CutoutService, **srcs: Path) -> tuple[dict, dict[str, Path]]:
    params, outs = _params(tmp_path, **srcs)
    receipt = tmp_path / "receipt.json"
    svc.run(params, receipt.as_uri())
    return json.loads(receipt.read_text()), outs


def _gradient(path: Path, w: int = 32, h: int = 16):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    arr = np.zeros((h, w, 3), np.uint8)
    arr[..., 0] = (np.arange(w)[None, :] * 7) % 256
    arr[..., 1] = (np.arange(h)[:, None] * 13) % 256
    arr[..., 2] = 100
    Image.fromarray(arr).save(path)
    return arr


def test_a_normal_item_is_the_source_rgb_under_the_model_alpha(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "a.png"
    arr = _gradient(src)
    receipt, outs = _run(tmp_path, _service(tmp_path, _block()), a=src)

    (item,) = receipt["items"]
    assert item["status"] == "cut" and (item["w"], item["h"]) == (32, 16)
    rgba = np.asarray(Image.open(outs["a"]).convert("RGBA"))
    assert rgba.shape == (16, 32, 4)
    assert (rgba[..., :3] == arr).all(), "the cut must carry the source's own pixels"
    assert rgba[8, 16, 3] == 255 and rgba[0, 0, 3] == 0
    # the receipt's bbox is the PNG's own alpha >= 24/255 extent, and the subject is a middle block, not the frame
    assert item["bbox"] == ic.bbox_fractions(rgba[..., 3] >= 24)
    x0, y0, x1, y1 = item["bbox"]
    assert 0 < x0 < 0.5 < x1 < 1 and 0 < y0 < 0.5 < y1 < 1
    assert item["bytes"] == outs["a"].stat().st_size


def test_an_exif_rotated_jpeg_is_cut_upright(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "r.jpg"
    wide = np.zeros((12, 24, 3), np.uint8)
    wide[:, :12] = (255, 0, 0)                 # stored left half red, right half blue
    wide[:, 12:] = (0, 0, 255)
    exif = Image.Exif()
    exif[0x0112] = 6                           # display = rotate 90 clockwise: the stored left becomes the TOP
    Image.fromarray(wide).save(src, exif=exif, quality=100, subsampling=0)
    receipt, outs = _run(tmp_path, _service(tmp_path, _block()), r=src)

    (item,) = receipt["items"]
    assert (item["w"], item["h"]) == (12, 24), "the cut must be the upright picture's size"
    rgba = np.asarray(Image.open(outs["r"]).convert("RGBA"))
    assert rgba.shape == (24, 12, 4)
    top, bottom = rgba[1, 6], rgba[22, 6]
    assert top[0] > 200 > top[2] and bottom[2] > 200 > bottom[0], "red must be on top after the rotation"


def test_a_source_over_the_cap_is_shrunk_to_2048_on_its_long_side(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "big.png"
    Image.new("RGB", (2100, 100), (10, 20, 30)).save(src)
    receipt, outs = _run(tmp_path, _service(tmp_path, _block()), big=src)
    assert (receipt["items"][0]["w"], receipt["items"][0]["h"]) == (2048, 98)       # 100 * 2048 / 2100 = 97.5
    assert Image.open(outs["big"]).size == (2048, 98)


def test_an_already_transparent_png_passes_through_by_name_with_no_model_call(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "t.png"
    rgba = np.zeros((16, 16, 4), np.uint8)
    rgba[:, 8:] = (200, 100, 50, 255)           # the right half is the picture, the left half is already clear
    Image.fromarray(rgba).save(src)
    svc = _service(tmp_path, _block())

    def no_model(_img):
        raise AssertionError("the model ran on a picture that is already cut")

    svc.alpha_map = no_model
    receipt, outs = _run(tmp_path, svc, t=src)
    (item,) = receipt["items"]
    assert item["status"] == "already_cut" and item["bbox"] == [0.5, 0.0, 1.0, 1.0]
    assert "bytes" not in item and not outs["t"].exists(), "an already-cut item PUTs nothing"


def test_an_opaque_rgba_png_is_not_mistaken_for_a_cut(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "o.png"
    Image.fromarray(np.full((16, 16, 4), 255, np.uint8)).save(src)     # has an alpha channel, all opaque
    receipt, _ = _run(tmp_path, _service(tmp_path, _block()), o=src)
    assert receipt["items"][0]["status"] == "cut"


def test_a_map_with_nothing_in_it_is_empty_cut_and_puts_nothing(tmp_path):
    src = tmp_path / "e.png"
    _gradient(src)
    receipt, outs = _run(tmp_path, _service(tmp_path, [[0.0] * MAP] * MAP), e=src)
    assert receipt["items"] == [{"id": "e", "status": "empty_cut"}]
    assert not outs["e"].exists()


def test_a_map_that_is_all_picture_is_no_separation_and_puts_nothing(tmp_path):
    src = tmp_path / "n.png"
    _gradient(src)
    receipt, outs = _run(tmp_path, _service(tmp_path, [[1.0] * MAP] * MAP), n=src)
    (item,) = receipt["items"]
    assert item["status"] == "no_separation" and item["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert not outs["n"].exists()


def test_bytes_that_are_not_a_picture_are_named_unreadable_and_the_rest_still_cut(tmp_path):
    bad, good = tmp_path / "bad.bin", tmp_path / "ok.png"
    bad.write_bytes(b"not a picture")
    _gradient(good)
    receipt, outs = _run(tmp_path, _service(tmp_path, _block()), bad=bad, ok=good)
    assert [(i["id"], i["status"]) for i in receipt["items"]] == [("bad", "unreadable"), ("ok", "cut")]
    assert outs["ok"].exists() and not outs["bad"].exists()


def test_the_receipt_validates_against_the_contract(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    srcs = {}
    for k in ("a", "e"):
        srcs[k] = tmp_path / f"{k}.png"
        _gradient(srcs[k])
    receipt, _ = _run(tmp_path, _service(tmp_path, _block()), **srcs)
    schema = json.loads((ROOT / "contracts" / "cutout.schema.json").read_text())
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors(receipt))
    assert receipt["torch"] == RUNTIME[0] and receipt["torchvision"] == RUNTIME[1] and receipt["model_sha256"] == SHA


def test_the_request_validates_against_the_contract():
    jsonschema = pytest.importorskip("jsonschema")
    from podagent.models import InferRequest

    req = {"infer_version": 7, "job_id": "j", "kind": "cutout", "model": "birefnet", "put_url": "https://x/o/r.json",
           "weights": {"url": "https://x/w.tar", "sha256": SHA},
           "cutout": {"items": [{"id": "s1", "src_url": "https://x/s.jpg", "out_url": "https://x/o.png"}]}}
    schema = json.loads((ROOT / "contracts" / "infer_request.schema.json").read_text())
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors(req))
    assert InferRequest.model_validate(req).cutout is not None


def test_weights_traced_under_another_runtime_are_refused_naming_both_pairs(tmp_path):
    wdir = _write_weights(tmp_path, _block(), torch="2.8.0+cu128", torchvision="0.23.0+cu128")
    with pytest.raises(ic.CutoutRefused) as exc:
        ic.CutoutService("stand-in", wdir, SHA, device="cpu", runtime=RUNTIME)
    msg = str(exc.value)
    assert "2.8.0+cu128" in msg and "0.23.0+cu128" in msg and RUNTIME[0] in msg and RUNTIME[1] in msg


def test_the_runtime_check_needs_both_versions_to_match():
    m = ic.Manifest("r", "v", 8, 8, "float32", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), "2.11.0+cu128", "0.26.0+cu128")
    ic.check_runtime(m, "2.11.0+cu128", "0.26.0+cu128")
    for torch_v, tv_v in (("2.11.1+cu128", "0.26.0+cu128"), ("2.11.0+cu128", "0.25.0+cu128")):
        with pytest.raises(ic.CutoutRefused):
            ic.check_runtime(m, torch_v, tv_v)


def test_a_pod_with_no_cuda_refuses_and_never_runs_on_the_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    wdir = _write_weights(tmp_path, _block())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ic.CutoutRefused, match="never CPU"):
        ic.CutoutService("stand-in", wdir, SHA)


def test_a_manifest_missing_a_field_or_holding_two_models_is_refused(tmp_path):
    wdir = tmp_path / "w"
    wdir.mkdir()
    m = _manifest()
    del m["torchvision"]
    (wdir / "config.json").write_text(json.dumps(m))
    (wdir / "a.pt").write_bytes(b"x")
    with pytest.raises(ic.CutoutRefused, match="torchvision"):
        ic.read_manifest(wdir)
    (wdir / "config.json").write_text(json.dumps(_manifest()))
    (wdir / "b.pt").write_bytes(b"x")
    with pytest.raises(ic.CutoutRefused, match="exactly one"):
        ic.read_manifest(wdir)


def test_the_image_installs_torchvision_on_the_torch_cu128_line_and_pins_it():
    text = "\n".join(ln for ln in (ROOT / "Dockerfile").read_text().splitlines() if not ln.strip().startswith("#"))
    cmds = re.findall(r"pip install\b[^;&]*", re.sub(r"\\\n", " ", text))
    torch_line = [c for c in cmds if "download.pytorch.org/whl/cu128" in c]
    assert len(torch_line) == 1 and re.search(r"\btorch\b.*\btorchaudio\b.*\btorchvision\b", torch_line[0])
    assert not [c for c in cmds if "torchvision" in c and c not in torch_line], "torchvision must not ride the PyPI line"
    pins = (ROOT / "constraints.txt").read_text().splitlines()
    assert "torchvision==0.26.0+cu128" in pins and "torch==2.11.0+cu128" in pins
