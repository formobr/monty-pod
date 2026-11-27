"""The screen_quad infer kind (docs/research/pod-screen-quad.md): a request on the pod yields the device face's 4 corners, a NAMED
`none` with reasons, or `unreadable` — never a job error for a bad mask, never a silent CPU run. Florence and SAM are replaced by
stubs at the module's two model seams (no weights, no GPU); the intake, the box selection, the opencv quad fit, the choice, the
receipt and the files PUT are the real code."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from podagent import infer_screen_quad as sq
from podagent.models import InferRequest, ScreenQuadParams, ScreenQuadPayload

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ("2.11.0+cu128", "0.26.0+cu128", "4.57.6")
SHA = "d" * 64
W, H = 400, 300
TILTED = [(120, 60), (300, 90), (280, 230), (100, 200)]          # TL, TR, BR, BL in px of the 400x300 picture
BOX = [90.0, 50.0, 310.0, 240.0]


def _manifest(**over) -> dict:
    m = {"florence": {"hf_repo": "florence-community/Florence-2-base", "hf_revision": "00921df66db7"},
         "sam": {"hf_repo": "facebook/sam2.1-hiera-base-plus", "hf_revision": "b7320756a133"},
         "transformers": RUNTIME[2], "torch": RUNTIME[0], "torchvision": RUNTIME[1],
         "dtypes": {"florence": "float16", "sam": "float32"}}
    m.update(over)
    return m


def _write_weights(tmp_path: Path, **manifest_over) -> Path:
    wdir = tmp_path / "weights"
    (wdir / "florence").mkdir(parents=True, exist_ok=True)
    (wdir / "sam").mkdir(exist_ok=True)
    (wdir / "config.json").write_text(json.dumps(_manifest(**manifest_over)))
    return wdir


def _poly_mask(points, w: int = W, h: int = H):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array(points, np.int32)], 1)
    return m.astype(bool)


def _blank(fw: int, fh: int):
    np = pytest.importorskip("numpy")
    return np.zeros((fh, fw), bool)


class StubGrounder:
    generation = {"max_new_tokens": 256, "num_beams": 3, "do_sample": False, "no_repeat_ngram_size": 3}

    def __init__(self, boxes):
        self._boxes, self.seen = boxes, []

    def boxes(self, img, phrase):
        self.seen.append((img.size, phrase))
        return self._boxes


class StubSegmenter:
    """Returns the same 3 masks (and iou scores) for every prompt; counts the embeddings it was asked for."""

    def __init__(self, mask, ious=(0.95, 0.97, 0.93), glass=None):
        self.masks3, self.ious, self.embeds, self.prompts = [mask, mask, mask], list(ious), 0, []
        self.glass, self.face_embeds, self.face_prompts = glass, 0, []

    def embed(self, img):
        if img.size != (W, H):                        # the upright face the glass step warps out of the picture
            self.face_embeds += 1
            return ("face", img.size)
        self.embeds += 1
        return "embedding"

    def masks(self, embedding, box, point=None):
        if isinstance(embedding, tuple):              # a glass-step prompt: `glass(face_w, face_h, box)` -> (3 masks, 3 ious)
            self.face_prompts.append(list(box))
            if self.glass is None:
                empty = _blank(*embedding[1])
                return [empty] * 3, [0.9] * 3
            return self.glass(*embedding[1], box)
        assert embedding == "embedding"
        self.prompts.append((list(box), point))
        return self.masks3, self.ious


def _service(tmp_path: Path, grounder, segmenter) -> sq.ScreenQuadService:
    return sq.ScreenQuadService("stand-in", _write_weights(tmp_path), SHA, runtime=RUNTIME, grounder=grounder,
                                segmenter=segmenter)


def _picture(path: Path, w: int = W, h: int = H) -> None:
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (w, h), (30, 60, 90)).save(path)


def _run(tmp_path: Path, svc: sq.ScreenQuadService, **srcs: tuple[Path, str]) -> dict:
    params = ScreenQuadParams.model_validate({"items": [
        {"id": k, "src_url": p.as_uri(), "device": dev} for k, (p, dev) in srcs.items()]})
    receipt = tmp_path / "receipt.json"
    svc.run(params, receipt.as_uri())
    return json.loads(receipt.read_text())


def test_a_tilted_screen_is_found_with_its_4_corners_tl_tr_br_bl_as_fractions(tmp_path):
    src = tmp_path / "phone.png"
    _picture(src)
    grounder, segmenter = StubGrounder([BOX]), StubSegmenter(_poly_mask(TILTED))
    receipt = _run(tmp_path, _service(tmp_path, grounder, segmenter), phone=(src, "phone"))

    (item,) = receipt["items"]
    assert item["status"] == "found" and item["surface"] == "device_face" and item["device"] == "phone"
    assert (item["w"], item["h"]) == (W, H)
    for got, (x, y) in zip(item["quad"], TILTED):
        assert abs(got[0] - x / W) < 0.01 and abs(got[1] - y / H) < 0.01
    assert grounder.seen == [((W, H), "phone screen")]
    assert segmenter.embeds == 1, "ONE image embedding per picture, reused by the box pass and the refinement pass"
    assert len(segmenter.prompts) == 2 and segmenter.prompts[1][1] is not None, "second pass = bbox + one positive point"
    assert receipt["generation"]["num_beams"] == 3 and receipt["transformers"] == RUNTIME[2]


def test_a_box_over_80_percent_and_a_box_holding_two_others_are_dropped_and_the_rest_go_largest_first():
    whole = [0.0, 0.0, 390.0, 290.0]
    desk = [20.0, 20.0, 380.0, 280.0]
    a, b = [40.0, 40.0, 120.0, 120.0], [200.0, 60.0, 300.0, 200.0]
    small = [5.0, 5.0, 15.0, 15.0]
    kept, reasons = sq.select_boxes([small, whole, a, desk, b], W, H)
    # `desk` holds a and b entirely: a scene, not a screen; `whole` is over 80 % of the frame
    assert [i for i, _ in kept] == [4, 2, 0]
    assert any("over_80_percent" in r and r.startswith("box 1") for r in reasons)
    assert any("contains_2_other_boxes" in r and r.startswith("box 3") for r in reasons)


def test_at_most_four_boxes_are_tried():
    boxes = [[10.0 * i, 10.0, 10.0 * i + 8.0, 18.0 + i] for i in range(7)]
    kept, _ = sq.select_boxes(boxes, W, H)
    assert len(kept) == sq.MAX_BOXES == 4


def test_a_sam_iou_under_0_9_is_none_with_its_reason(tmp_path):
    src = tmp_path / "p.png"
    _picture(src)
    segmenter = StubSegmenter(_poly_mask(TILTED), ious=(0.5, 0.6, 0.89))
    receipt = _run(tmp_path, _service(tmp_path, StubGrounder([BOX]), segmenter), p=(src, "tablet"))
    (item,) = receipt["items"]
    assert item["status"] == "none" and "quad" not in item
    assert any("box 0" in r and "sam_iou_0.89" in r for r in item["reasons"])
    assert len(segmenter.prompts) == 1, "no refinement pass for a box SAM does not trust"


@pytest.mark.parametrize("name,points", [("line", [(50, 100), (250, 100)]), ("triangle", [(50, 200), (250, 200), (150, 60)])])
def test_a_one_pixel_line_and_a_triangle_are_a_none_never_an_exception(tmp_path, name, points):
    src = tmp_path / f"{name}.png"
    _picture(src)
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    mask = np.zeros((H, W), np.uint8)
    if name == "line":
        cv2.line(mask, points[0], points[1], 1, 1)
    else:
        cv2.fillPoly(mask, [np.array(points, np.int32)], 1)
    receipt = _run(tmp_path, _service(tmp_path, StubGrounder([[20.0, 40.0, 280.0, 220.0]]), StubSegmenter(mask.astype(bool))),
                   p=(src, "laptop"))
    (item,) = receipt["items"]
    assert item["status"] == "none" and item["reasons"] and "quad" not in item


def test_a_corner_beyond_the_frame_bound_is_none_screen_cut_by_frame(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    src = tmp_path / "wide.png"
    _picture(src)
    # approxPolyN may extrapolate a corner past the picture; the mask is only what is inside it, so the stand-in mask is the
    # quad clipped to the frame and the fit returns the unclipped corner
    quad = np.array([[-0.4 * W, 0.2 * H], [0.55 * W, 0.18 * H], [0.57 * W, 0.71 * H], [-0.4 * W, 0.74 * H]])
    mask = _poly_mask([(0, 60), (220, 54), (228, 213), (0, 222)])
    monkeypatch.setattr(sq, "fit_quad", lambda m: (quad, ""))
    monkeypatch.setattr(sq, "polygon_iou", lambda q, m: 1.0)
    receipt = _run(tmp_path, _service(tmp_path, StubGrounder([[0.0, 50.0, 230.0, 230.0]]), StubSegmenter(mask)), p=(src, "monitor"))
    (item,) = receipt["items"]
    assert item["status"] == "none" and any("screen_cut_by_frame" in r for r in item["reasons"])
    # a corner a little outside the frame is NOT clamped and is still found
    inside = np.array([[-0.1 * W, 0.2 * H], [0.55 * W, 0.18 * H], [0.57 * W, 0.71 * H], [-0.05 * W, 0.74 * H]])
    monkeypatch.setattr(sq, "fit_quad", lambda m: (inside, ""))
    receipt = _run(tmp_path, _service(tmp_path, StubGrounder([[0.0, 50.0, 230.0, 230.0]]), StubSegmenter(mask)), p=(src, "monitor"))
    (item,) = receipt["items"]
    assert item["status"] == "found" and item["quad"][0][0] == pytest.approx(-0.1)


def test_an_exif_rotated_jpeg_is_read_upright_and_an_rgba_picture_is_composited_on_white(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "r.jpg"
    wide = np.zeros((12, 24, 3), np.uint8)
    wide[:, :12] = (255, 0, 0)
    wide[:, 12:] = (0, 0, 255)
    exif = Image.Exif()
    exif[0x0112] = 6                           # display = rotate 90 clockwise: the stored left becomes the TOP
    Image.fromarray(wide).save(src, exif=exif, quality=100, subsampling=0)
    img = sq.load_picture(src)
    assert img.size == (12, 24) and img.mode == "RGB"
    a = np.asarray(img)
    assert a[1, 6][0] > 200 > a[1, 6][2] and a[22, 6][2] > 200 > a[22, 6][0], "red must be on top after the rotation"

    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[:5, :, :] = (0, 0, 0, 0)               # fully transparent half: it must come out white, not black
    rgba[5:, :, :] = (200, 0, 0, 255)
    Image.fromarray(rgba).save(tmp_path / "a.png")
    out = np.asarray(sq.load_picture(tmp_path / "a.png"))
    assert tuple(out[1, 1]) == (255, 255, 255) and tuple(out[8, 1]) == (200, 0, 0)


def test_a_source_over_the_cap_is_shrunk_to_2048_on_its_long_side(tmp_path):
    _picture(tmp_path / "big.png", 4096, 1024)
    assert sq.load_picture(tmp_path / "big.png").size == (2048, 512)


def test_an_unreadable_file_is_named_while_the_other_items_of_the_request_still_answer(tmp_path):
    bad, good = tmp_path / "bad.bin", tmp_path / "ok.png"
    bad.write_bytes(b"not a picture")
    _picture(good)
    svc = _service(tmp_path, StubGrounder([BOX]), StubSegmenter(_poly_mask(TILTED)))
    receipt = _run(tmp_path, svc, bad=(bad, "phone"), ok=(good, "phone"))
    assert [(i["id"], i["status"]) for i in receipt["items"]] == [("bad", "unreadable"), ("ok", "found")]


def test_a_picture_florence_finds_nothing_in_is_none_with_a_reason(tmp_path):
    src = tmp_path / "p.png"
    _picture(src)
    receipt = _run(tmp_path, _service(tmp_path, StubGrounder([]), StubSegmenter(_poly_mask(TILTED))), p=(src, "tv"))
    (item,) = receipt["items"]
    assert item["status"] == "none"
    assert item["reasons"] == ["tv screen: florence_returned_no_box", "the screen of the tv: florence_returned_no_box"]


# ── the second phrase (docs/research/pod-screen-quad.md, Amendment 2026-11-19) ──────────────────────────────────────

LAPTOP = json.loads((ROOT / "tests" / "fixtures" / "laptop_masks.json").read_text())


def _laptop_mask(which: str):
    import base64
    import io

    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(LAPTOP["masks_png_b64"][which]))).convert("L")) > 127


class PhraseGrounder:
    """phrase → box; a phrase it does not know finds nothing."""

    def __init__(self, by_phrase):
        self.by_phrase, self.asked = by_phrase, []

    def boxes(self, img, phrase):
        self.asked.append(phrase)
        return self.by_phrase.get(phrase, [])


class BoxSegmenter:
    """box → the mask of the phrase that returned it; counts the embeddings of the picture (the face of the glass step is apart)."""

    def __init__(self, by_box, size):
        self.by_box, self.size, self.embeds = by_box, size, 0

    def embed(self, img):
        if img.size != self.size:
            return ("face", img.size)
        self.embeds += 1
        return "embedding"

    def masks(self, embedding, box, point=None):
        if isinstance(embedding, tuple):
            return [_blank(*embedding[1])] * 3, [0.9] * 3
        mask = self.by_box.get(tuple(round(float(v)) for v in box))
        if mask is None:                              # the second pass: a box around the mask it was given
            mask = next(iter(self.by_box.values()))
        return [mask] * 3, [0.95, 0.97, 0.93]


def _laptop_stubs():
    Image = pytest.importorskip("PIL.Image")
    fw, fh = LAPTOP["fixture_size"]
    sx, sy = fw / LAPTOP["size"][0], fh / LAPTOP["size"][1]
    scale = lambda b: [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
    boxes = {LAPTOP[k]["phrase"]: [scale(LAPTOP[k]["box"])] for k in ("device", "of")}
    masks = {tuple(round(v) for v in boxes[LAPTOP[k]["phrase"]][0]): _laptop_mask(k) for k in ("device", "of")}
    return PhraseGrounder(boxes), BoxSegmenter(masks, (fw, fh)), Image.new("RGB", (fw, fh))


def test_a_laptop_screen_is_asked_as_the_screen_of_the_laptop():
    grounder, segmenter, img = _laptop_stubs()
    answer = sq.locate_screen(grounder, segmenter, img, "laptop")
    assert answer.quad is not None
    assert grounder.asked == ["laptop screen", "the screen of the laptop"]
    fw, fh = img.size
    quad_px = [(x * fw, y * fh) for x, y in answer.quad]
    assert sq.polygon_iou(quad_px, _laptop_mask("of")) >= 0.95
    assert segmenter.embeds == 1, "ONE image embedding per picture, reused by the second phrase"
    assert answer.reasons[0].startswith("laptop screen: box 0: no_quad_fits_mask")


def test_the_second_phrase_is_never_asked_when_the_first_already_fits():
    grounder = PhraseGrounder({"phone screen": [BOX]})
    segmenter = StubSegmenter(_poly_mask(TILTED))
    answer = sq.locate_screen(grounder, segmenter, _blank_image(), "phone")
    assert answer.quad is not None and grounder.asked == ["phone screen"] and answer.reasons == []


def _blank_image():
    Image = pytest.importorskip("PIL.Image")
    return Image.new("RGB", (W, H))


def test_when_both_phrases_fail_the_reasons_name_both():
    grounder = PhraseGrounder({"laptop screen": [BOX], "the screen of the laptop": [[20.0, 40.0, 280.0, 220.0]]})
    segmenter = StubSegmenter(_poly_mask([(50, 200), (250, 200), (150, 60)]))
    answer = sq.locate_screen(grounder, segmenter, _blank_image(), "laptop")
    assert answer.quad is None and grounder.asked == ["laptop screen", "the screen of the laptop"]
    assert any(r.startswith("laptop screen: box 0") for r in answer.reasons)
    assert any(r.startswith("the screen of the laptop: box 0") for r in answer.reasons)
    assert segmenter.embeds == 1


def test_weights_built_for_another_runtime_refuse_the_kind_naming_both(tmp_path):
    for over in ({"transformers": "4.56.0"}, {"torch": "2.8.0+cu128"}, {"torchvision": "0.23.0+cu128"}):
        wdir = tmp_path / next(iter(over))
        (wdir / "florence").mkdir(parents=True)
        (wdir / "sam").mkdir()
        (wdir / "config.json").write_text(json.dumps(_manifest(**over)))
        with pytest.raises(sq.ScreenQuadRefused) as exc:
            sq.ScreenQuadService("stand-in", wdir, SHA, runtime=RUNTIME, grounder=StubGrounder([]),
                                 segmenter=StubSegmenter(None))
        msg = str(exc.value)
        assert next(iter(over.values())) in msg and all(v in msg for v in RUNTIME)


def test_a_manifest_missing_a_field_or_a_snapshot_dir_is_refused(tmp_path):
    wdir = tmp_path / "w"
    wdir.mkdir()
    m = _manifest()
    del m["torchvision"]
    (wdir / "config.json").write_text(json.dumps(m))
    with pytest.raises(sq.ScreenQuadRefused, match="torchvision"):
        sq.read_manifest(wdir)
    (wdir / "config.json").write_text(json.dumps(_manifest()))
    with pytest.raises(sq.ScreenQuadRefused, match="florence/"):
        sq.read_manifest(wdir)
    (wdir / "florence").mkdir()
    (wdir / "sam").mkdir()
    assert sq.read_manifest(wdir).sam_dtype == "float32"


def test_a_snapshot_that_loads_with_any_missing_unexpected_or_mismatched_key_is_refused():
    sq.check_loading_info("sam", {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []})
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        with pytest.raises(sq.ScreenQuadRefused, match=key):
            sq.check_loading_info("florence", {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], key: ["w"]})


def test_a_pod_with_no_cuda_refuses_and_never_runs_on_the_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(sq.ScreenQuadRefused, match="never CPU"):
        sq.ScreenQuadService("stand-in", _write_weights(tmp_path), SHA, runtime=RUNTIME)


def test_the_effective_generation_config_is_the_snapshots_under_the_overrides():
    eff = sq.effective_generation({"num_beams": 1, "no_repeat_ngram_size": 3, "early_stopping": True, "bad": [1, 2]})
    assert eff == {"do_sample": False, "early_stopping": True, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "num_beams": 3}


def test_the_request_and_the_receipt_validate_against_the_contract_and_a_device_outside_the_enum_is_refused(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    req = {"infer_version": 7, "job_id": "j", "kind": "screen_quad", "model": "florence_sam", "put_url": "https://x/o/r.json",
           "weights": {"url": "https://x/w.tar", "sha256": SHA},
           "screen_quad": {"items": [{"id": "s1", "src_url": "https://x/s.jpg", "device": "monitor"}]}}
    req_schema = json.loads((ROOT / "contracts" / "infer_request.schema.json").read_text())
    assert not list(jsonschema.Draft202012Validator(req_schema).iter_errors(req))
    assert InferRequest.model_validate(req).screen_quad is not None
    bad = json.loads(json.dumps(req))
    bad["screen_quad"]["items"][0]["device"] = "toaster"
    assert list(jsonschema.Draft202012Validator(req_schema).iter_errors(bad))
    with pytest.raises(Exception):
        InferRequest.model_validate(bad)

    src = tmp_path / "p.png"
    _picture(src)
    receipt = _run(tmp_path, _service(tmp_path, StubGrounder([BOX]), StubSegmenter(_poly_mask(TILTED))), p=(src, "phone"))
    schema = json.loads((ROOT / "contracts" / "screen_quad.schema.json").read_text())
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors(receipt))
    assert ScreenQuadPayload.model_validate(receipt).items[0].status == "found"
    assert receipt["model_sha256"] == SHA and receipt["torch"] == RUNTIME[0]


# ── the glass inside the face (docs/research/pod-screen-quad.md, Amendment 2026-11-16) ──────────────────────────────

INNER = (0.1, 0.15, 0.9, 0.85)          # the glass in FACE fractions (u0, v0, u1, v1)


def _face_to_picture(u: float, v: float) -> tuple[float, float]:
    """TILTED is a parallelogram: face fraction (u, v) → picture px."""
    (tlx, tly), (trx, try_), _, (blx, bly) = TILTED
    return tlx + u * (trx - tlx) + v * (blx - tlx), tly + u * (try_ - tly) + v * (bly - tly)


def _glass_picture(path: Path) -> None:
    """A dark picture whose glass (INNER of the TILTED face) is bright: the glass boundary is the sharp edge."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    arr = np.full((H, W, 3), 20, np.uint8)
    u0, v0, u1, v1 = INNER
    corners = [_face_to_picture(u, v) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
    cv2.fillPoly(arr, [np.round(np.array(corners)).astype(np.int32)], (230, 230, 230))
    Image.fromarray(arr).save(path)


def _inner_glass(fw: int, fh: int, box, *, hole: bool = False, notch: bool = False):
    """The sweep's masks in face px: the whole face at s = 0, the INNER rectangle at s = 0.05, nothing else."""
    np = pytest.importorskip("numpy")
    s = box[0] / fw
    if abs(s) < 1e-9:
        return [np.ones((fh, fw), bool)] * 3, [0.95] * 3
    if abs(s - 0.05) > 1e-6:
        return [_blank(fw, fh)] * 3, [0.95] * 3
    u0, v0, u1, v1 = INNER
    x0, y0, x1, y1 = round(u0 * fw), round(v0 * fh), round(u1 * fw), round(v1 * fh)
    m = _blank(fw, fh)
    m[y0:y1, x0:x1] = True
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    if hole:
        m[cy - 2:cy + 2, cx - 2:cx + 2] = False                     # 16 px inside ~14k: far under 2 % of the glass
    if notch:
        m[y0:y0 + 8, cx + 20:cx + 26] = False                       # cut in from the top border: open to the outside
    return [m] * 3, [0.95] * 3


def test_the_glass_inside_the_face_is_found_by_the_shrink_sweep_and_the_face_quad_is_still_answered(tmp_path):
    src = tmp_path / "phone.png"
    _glass_picture(src)
    segmenter = StubSegmenter(_poly_mask(TILTED), glass=_inner_glass)
    receipt = _run(tmp_path, _service(tmp_path, StubGrounder([BOX]), segmenter), phone=(src, "phone"))

    (item,) = receipt["items"]
    assert item["status"] == "found" and "reasons" not in item
    for got, (x, y) in zip(item["quad"], TILTED):
        assert abs(got[0] - x / W) < 0.01 and abs(got[1] - y / H) < 0.01
    u0, v0, u1, v1 = INNER
    for got, (u, v) in zip(item["glass"], ((u0, v0), (u1, v0), (u1, v1), (u0, v1))):
        x, y = _face_to_picture(u, v)
        assert abs(got[0] - x / W) < 0.01 and abs(got[1] - y / H) < 0.01
    assert "mask" not in item, "no mask_url, no PNG, no `mask`"
    assert segmenter.face_embeds == 1, "ONE embedding of the face for the whole sweep"
    steps = [round(b[0] / (b[2] + b[0]) / 0.0125, 6) for b in segmenter.face_prompts]
    assert len(segmenter.face_prompts) == 13 and steps[0] == 0 and steps[-1] == 12, "0 .. 15 % in 1.25 % steps"


def test_a_mask_url_gets_a_picture_sized_grey_png_with_a_small_hole_filled_and_a_border_notch_kept(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "phone.png"
    _glass_picture(src)
    segmenter = StubSegmenter(_poly_mask(TILTED), glass=lambda fw, fh, box: _inner_glass(fw, fh, box, hole=True, notch=True))
    params = ScreenQuadParams.model_validate({"items": [
        {"id": "phone", "src_url": src.as_uri(), "device": "phone", "mask_url": (tmp_path / "glass.png").as_uri()}]})
    receipt_path = tmp_path / "receipt.json"
    _service(tmp_path, StubGrounder([BOX]), segmenter).run(params, receipt_path.as_uri())
    (item,) = json.loads(receipt_path.read_text())["items"]
    assert item["mask"] is True and len(item["glass"]) == 4

    png = Image.open(tmp_path / "glass.png")
    assert png.mode == "L" and png.size == (W, H)
    a = np.asarray(png)
    assert set(np.unique(a)) <= {0, 255}
    matrix, fw, fh = sq._warp_to_face(np.array(TILTED, np.float32))
    back = np.linalg.inv(matrix)

    def at(fx: float, fy: float) -> int:
        x, y = cv2.perspectiveTransform(np.array([[[fx, fy]]], np.float32), back)[0, 0]
        return int(a[round(y), round(x)])

    u0, v0, u1, v1 = INNER
    x0, y0, x1, y1 = u0 * fw, v0 * fh, u1 * fw, v1 * fh
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert at(cx + 10, cy + 10) == 255, "inside the glass"
    assert at(cx, cy) == 255, "the small interior hole is filled"
    assert at(cx + 23, y0 + 3) == 0, "the notch cut in from the border is NOT filled"
    assert at(x0 / 2, cy) == 0 and at(cx, (y1 + fh) / 2) == 0, "the bezel is outside the glass"


@pytest.mark.parametrize("what", ["no_prompt_passes", "iou_under_0_70", "area_under_30_percent"])
def test_no_glass_candidate_over_the_thresholds_is_no_glass_with_a_reason_and_the_face_quad_still_answered(tmp_path, what):
    src = tmp_path / "phone.png"
    _glass_picture(src)

    def glass(fw, fh, box):
        if what == "no_prompt_passes":
            return [_blank(fw, fh)] * 3, [0.95] * 3
        m = _blank(fw, fh)
        share = 0.29 if what == "area_under_30_percent" else 0.5
        m[: int(fh * share), :] = True
        return [m] * 3, [0.69 if what == "iou_under_0_70" else 0.95] * 3

    receipt = _run(tmp_path, _service(tmp_path, StubGrounder([BOX]), StubSegmenter(_poly_mask(TILTED), glass=glass)),
                   phone=(src, "phone"))
    (item,) = receipt["items"]
    assert item["status"] == "found" and len(item["quad"]) == 4
    assert "glass" not in item and "mask" not in item
    assert item["reasons"] == ["glass: no_candidate_over_thresholds"]


def test_a_request_with_mask_url_and_a_row_with_glass_and_mask_validate_against_the_contracts(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    req = {"infer_version": 7, "job_id": "j", "kind": "screen_quad", "model": "florence_sam", "put_url": "https://x/o/r.json",
           "weights": {"url": "https://x/w.tar", "sha256": SHA},
           "screen_quad": {"items": [{"id": "s1", "src_url": "https://x/s.jpg", "device": "phone",
                                      "mask_url": "https://x/o/m.png"}]}}
    req_schema = json.loads((ROOT / "contracts" / "infer_request.schema.json").read_text())
    assert not list(jsonschema.Draft202012Validator(req_schema).iter_errors(req))
    assert InferRequest.model_validate(req).screen_quad.items[0].mask_url == "https://x/o/m.png"
    empty = json.loads(json.dumps(req))
    empty["screen_quad"]["items"][0]["mask_url"] = ""
    assert list(jsonschema.Draft202012Validator(req_schema).iter_errors(empty))
    with pytest.raises(Exception):
        InferRequest.model_validate(empty)

    src = tmp_path / "phone.png"
    _glass_picture(src)
    params = ScreenQuadParams.model_validate({"items": [
        {"id": "p", "src_url": src.as_uri(), "device": "phone", "mask_url": (tmp_path / "m.png").as_uri()}]})
    receipt_path = tmp_path / "receipt.json"
    _service(tmp_path, StubGrounder([BOX]), StubSegmenter(_poly_mask(TILTED), glass=_inner_glass)).run(
        params, receipt_path.as_uri())
    receipt = json.loads(receipt_path.read_text())
    schema = json.loads((ROOT / "contracts" / "screen_quad.schema.json").read_text())
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors(receipt))
    row = ScreenQuadPayload.model_validate(receipt).items[0]
    assert row.glass is not None and row.mask is True
    bad = json.loads(json.dumps(receipt))
    bad["items"][0]["glass"][0][0] = 1.5
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(bad))
    with pytest.raises(Exception):
        ScreenQuadPayload.model_validate(bad)
