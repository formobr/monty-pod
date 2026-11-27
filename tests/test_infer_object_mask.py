"""The object_mask infer kind: the caller's noun phrase is grounded by the same Florence task and the object is cut out with SAM — an
RGBA cut PNG, a grey mask PNG and a receipt row per picture. Florence and SAM are the stubs of tests/test_infer_screen_quad.py at the
module's two model seams (no weights, no GPU); the intake, the box selection, the mask choice, the PNGs and the receipt are real."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from podagent.models import InferRequest, ObjectMaskOut, ObjectMaskParams, ObjectMaskPayload

from test_infer_screen_quad import BOX, H, W, StubGrounder, StubSegmenter, _picture, _service

ROOT = Path(__file__).resolve().parents[1]
REGION = (100, 60, 40, 30)        # x, y, w, h of the stub mask: 40 x 30 px at (100, 60) of the 400 x 300 picture


def _region_mask():
    np = pytest.importorskip("numpy")
    m = np.zeros((H, W), bool)
    x, y, w, h = REGION
    m[y:y + h, x:x + w] = True
    return m


def _run(tmp_path: Path, svc, phrase: str = "red tram", with_mask: bool = True) -> tuple[dict, Path, Path]:
    src = tmp_path / "src.png"
    _picture(src)
    cut, mask = tmp_path / "cut.png", tmp_path / "mask.png"
    item = {"id": "a", "src_url": src.as_uri(), "phrase": phrase, "cut_url": cut.as_uri()}
    if with_mask:
        item["mask_url"] = mask.as_uri()
    receipt = tmp_path / "receipt.json"
    svc.run_object_mask(ObjectMaskParams.model_validate({"items": [item]}), receipt.as_uri())
    return json.loads(receipt.read_text()), cut, mask


def test_the_phrase_is_grounded_verbatim_and_the_mask_becomes_an_rgba_cut_a_grey_mask_and_a_found_row(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    grounder, segmenter = StubGrounder([BOX]), StubSegmenter(_region_mask())
    receipt, cut, mask = _run(tmp_path, _service(tmp_path, grounder, segmenter))

    assert grounder.seen == [((W, H), "red tram")]
    assert segmenter.embeds == 1
    (row,) = receipt["items"]
    assert row["status"] == "found" and (row["w"], row["h"]) == (W, H)
    x, y, w, h = REGION
    for got, want in zip(row["bbox"], (x / W, y / H, w / W, h / H)):
        assert abs(got - want) < 0.01
    rgba = Image.open(cut)
    assert rgba.mode == "RGBA" and rgba.size == (W, H)
    arr = np.asarray(rgba)
    assert (arr[y:y + h, x:x + w, 3] == 255).all() and int(arr[..., 3].sum()) == 255 * w * h
    assert (arr[y:y + h, x:x + w, :3] == (30, 60, 90)).all(), "the picture's own pixels inside the mask"
    grey = Image.open(mask)
    assert grey.mode == "L" and grey.size == (W, H)
    g = np.asarray(grey)
    assert (g[y:y + h, x:x + w] == 255).all() and int((g == 255).sum()) == w * h and int((g == 0).sum()) == W * H - w * h
    ObjectMaskPayload.model_validate(receipt)
    Draft202012Validator(json.loads((ROOT / "contracts/object_mask.schema.json").read_text())).validate(row)


def test_without_a_mask_url_no_mask_png_is_put(tmp_path):
    receipt, cut, mask = _run(tmp_path, _service(tmp_path, StubGrounder([BOX]), StubSegmenter(_region_mask())), with_mask=False)
    assert receipt["items"][0]["status"] == "found" and cut.is_file() and not mask.exists()


def test_no_box_is_none_with_florence_returned_no_box_and_nothing_is_put(tmp_path):
    receipt, cut, mask = _run(tmp_path, _service(tmp_path, StubGrounder([]), StubSegmenter(_region_mask())))
    (row,) = receipt["items"]
    assert row["status"] == "none" and row["reasons"] == ["florence_returned_no_box"] and "bbox" not in row
    assert not cut.exists() and not mask.exists()


def test_a_mask_over_80_percent_of_the_picture_is_none(tmp_path):
    np = pytest.importorskip("numpy")
    receipt, cut, _ = _run(tmp_path, _service(tmp_path, StubGrounder([BOX]), StubSegmenter(np.ones((H, W), bool))))
    (row,) = receipt["items"]
    assert row["status"] == "none" and any("no_mask_under_80_percent" in r for r in row["reasons"]) and not cut.exists()


def test_the_request_and_the_rows_validate_against_the_schemas_and_models():
    schema = json.loads((ROOT / "contracts/infer_request.schema.json").read_text())
    req = json.loads((ROOT / "contracts/examples/infer_request.object_mask.json").read_text())
    Draft202012Validator(schema).validate(req)
    InferRequest.model_validate(req)
    bad = json.loads((ROOT / "contracts/examples/invalid/infer_request.object_mask_phrase_with_quote.json").read_text())
    assert next(Draft202012Validator(schema).iter_errors(bad), None) is not None
    with pytest.raises(ValueError):
        InferRequest.model_validate(bad)
    with pytest.raises(ValueError):
        ObjectMaskOut(id="a", status="found", w=4, h=4)
