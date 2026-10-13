"""MISC-342 — a pod failure never looks like data.

Prod, 2026-10: every b-roll clip_rank score was -1.0 for 11 days. The fetch guard refused every tile
(UrlNotAllowed), _fetch_tile swallowed it, and the payload carried a well-formed -1.0 per tile that the planner
read as "irrelevant footage". What each test pins:
  • a tile that cannot be fetched carries `errors[j] = {kind, detail}` and `score: null` — never a number;
  • a group in which every tile failed is `status: "error"` carrying the first tile's reason;
  • a face_probe source that will not decode ends the job status=error with the reason, not ok + zero frames.
Hermetic: stub towers, no weights, no GPU, no network, no clock.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import podagent.infer_cliprank as m
from podagent.cp import UrlNotAllowed
from podagent.models import ClipRankGroup, ClipRankParams
from test_infer_cliprank import _svc  # noqa: E402 — same stub towers, one definition

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
_CLIP_RANK_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "clip_rank.schema.json").read_text()))
_RESULT_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "infer_result.schema.json").read_text()))


def _run_payload(monkeypatch, tmp_path: Path, groups: list[ClipRankGroup], download) -> dict[str, Any]:
    """Run the real ClipRankService.run over stub towers; return the JSON body it PUT, schema-checked."""
    monkeypatch.setattr(m, "download", download)
    monkeypatch.setattr("PIL.Image.open", lambda p: types.SimpleNamespace(convert=lambda mode: "img"))
    put = tmp_path / "put" / "clip_rank.json"

    def fake_upload(src: Path, _url: str, _ct: str | None = None) -> None:
        put.parent.mkdir(parents=True, exist_ok=True)
        put.write_bytes(src.read_bytes())

    monkeypatch.setattr(m, "upload", fake_upload)
    _svc().run(ClipRankParams(groups=groups), "https://storage.example/o/1.json")
    body = json.loads(put.read_text())
    errors = sorted(_CLIP_RANK_SCHEMA.iter_errors(body), key=str)
    assert not errors, f"the payload the pod PUT is off-contract: {errors[0].message}"
    return body


def _write(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"img")
    return dest


def test_a_cliprank_tile_that_cannot_be_fetched_carries_its_error_and_no_score(monkeypatch, tmp_path) -> None:
    def download(url: str, dest: Path) -> Path:
        if url == "https://stock.example/refused.jpg":
            raise UrlNotAllowed("REFUSED: https://stock.example/refused.jpg?X-Amz-Signature=secret")
        return _write(dest)

    body = _run_payload(monkeypatch, tmp_path, [ClipRankGroup(
        intent="a falling crypto chart",
        image_urls=["https://r2.example/ok.jpg", "https://stock.example/refused.jpg"])], download)

    (group,) = body["groups"]
    assert group["status"] == "ok", "one live tile keeps the group a result"
    assert group["scores"] == [0.6, None], "a refused tile has NO score — never -1.0"
    assert -1.0 not in group["scores"]
    assert group["embeds"][1] is None
    assert group["errors"][0] is None
    err = group["errors"][1]
    assert err["kind"] == "url_not_allowed"
    assert "UrlNotAllowed" in err["detail"] and "stock.example/refused.jpg" in err["detail"]
    assert "secret" not in err["detail"] and "X-Amz-Signature" not in err["detail"]


@pytest.mark.parametrize(("exc", "kind"), [
    (UrlNotAllowed("REFUSED stock host"), "url_not_allowed"),
    (m.TransferTimeout("GET outlived its deadline"), "fetch_timeout"),
    (OSError("connection reset"), "fetch_failed"),
])
def test_each_fetch_failure_is_named_by_its_kind(monkeypatch, tmp_path, exc, kind) -> None:
    monkeypatch.setattr(m, "download", lambda url, dest: (_ for _ in ()).throw(exc))
    img, err, _ = m._fetch_tile("https://r2.example/t.jpg?sig=abc", tmp_path / "t")
    assert img is None and err.kind == kind and type(exc).__name__ in err.detail


def test_bytes_that_will_not_decode_are_a_decode_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(m, "download", lambda url, dest: _write(dest))
    monkeypatch.setattr("PIL.Image.open", lambda p: (_ for _ in ()).throw(OSError("cannot identify image")))
    img, err, _ = m._fetch_tile("https://r2.example/t.jpg", tmp_path / "t")
    assert img is None and err.kind == "decode_error" and "cannot identify image" in err.detail


def test_a_cliprank_group_where_every_tile_failed_is_an_error(monkeypatch, tmp_path) -> None:
    def download(url: str, dest: Path) -> Path:
        if "stock" in url:
            raise UrlNotAllowed(f"REFUSED {url}")
        return _write(dest)

    body = _run_payload(monkeypatch, tmp_path, [
        ClipRankGroup(intent="chart", image_urls=["https://stock.example/a.jpg", "https://stock.example/b.jpg"]),
        ClipRankGroup(intent="chart", image_urls=["https://r2.example/c.jpg"]),
    ], download)

    dead, live = body["groups"]
    assert dead["status"] == "error"
    assert dead["error"]["kind"] == "url_not_allowed" and "stock.example/a.jpg" in dead["error"]["detail"], (
        "the group error is its FIRST tile's reason")
    assert dead["scores"] == [None, None] and dead["embeds"] == [None, None]
    assert [e["kind"] for e in dead["errors"]] == ["url_not_allowed", "url_not_allowed"]
    assert live["status"] == "ok" and "error" not in live, "a dead group never poisons its neighbour"


def test_embed_only_group_names_why_it_has_no_score_and_stays_ok(monkeypatch, tmp_path) -> None:
    body = _run_payload(monkeypatch, tmp_path, [ClipRankGroup(intent="", image_urls=["a", "b"])],
                        lambda url, dest: _write(dest))
    (group,) = body["groups"]
    assert group["status"] == "ok", "its embeddings ARE the answer an embed-only caller asked for"
    assert group["scores"] == [None, None]
    assert [e["kind"] for e in group["errors"]] == ["empty_intent", "empty_intent"]
    assert all(e is not None for e in group["embeds"])


class _Capture:
    """cv2.VideoCapture as OpenCV behaves on bytes it cannot decode: either the container never opens, or it
    opens with a header but no frame ever reads. (opencv rides the `ml` extra, which CI does not install.)"""

    def __init__(self, opened: bool) -> None:
        self._opened = opened

    def isOpened(self) -> bool:   # noqa: N802 — OpenCV's spelling
        return self._opened

    def get(self, prop: int) -> float:
        return {0: 25.0, 1: 640.0, 2: 360.0}[prop] if self._opened else 0.0

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def read(self):
        return False, None

    def grab(self) -> bool:
        return False

    def release(self) -> None:
        pass


@pytest.mark.parametrize("opened", [False, True], ids=["container-never-opens", "opens-but-no-frame-reads"])
def test_face_probe_on_an_undecodable_video_is_an_error(monkeypatch, tmp_path, opened) -> None:
    import importlib
    import sys

    from podagent import main as agent_main

    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_POS_FRAMES = 0, 1, 2, 3
    cv2.VideoCapture = lambda _path: _Capture(opened)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.delitem(sys.modules, "podagent.infer_probe", raising=False)
    probe = importlib.import_module("podagent.infer_probe")
    monkeypatch.setitem(sys.modules, "podagent.infer_probe", probe)

    def download(_url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00not a video container\xff" * 64)   # non-empty, undecodable
        return dest

    uploads: list[Path] = []
    monkeypatch.setattr(probe, "download", download)
    monkeypatch.setattr(probe, "upload", lambda src, url, ct=None: uploads.append(src))

    svc = probe.ProbeService.__new__(probe.ProbeService)   # no YuNet file: the decoder refuses first
    svc.model_name = "yunet"
    svc.detector = types.SimpleNamespace(setInputSize=lambda _wh: None)

    class _CP:
        def __init__(self) -> None:
            self.results: list[dict[str, Any]] = []

        def note(self, _ev: dict[str, Any]) -> None:
            pass

        def report_infer_result(self, payload: dict[str, Any]) -> bool:
            self.results.append(payload)
            return True

    req = json.loads((CONTRACTS / "examples" / "infer_request.face_probe.json").read_text())
    yunet = tmp_path / "yunet.onnx"
    cp = _CP()
    agent_main._run_infer(req, cp, {}, {(yunet, req["model"]): svc}, {}, yunet, False,
                          corr_id="work/s/r", session_id="s")

    assert uploads == [], "no payload of zero-frame shots may be PUT for a video that never decoded"
    (result,) = cp.results
    assert result["status"] == "error" and result["kind"] == "face_probe"
    assert "UndecodableVideo" in result["error"] and "bytes" in result["error"], result["error"]
    wire = {k: v for k, v in result.items() if k != "session_id"}
    assert not sorted(_RESULT_SCHEMA.iter_errors(wire), key=str)


class _Rows:
    """The minimum tensor surface _score_prepared reads: `.rows`, as the stub towers' _Vec has."""

    def __init__(self, rows):
        self.rows = rows


@pytest.mark.parametrize(("score", "embed"), [
    (float("nan"), [0.6, 0.8]),
    (float("inf"), [0.6, 0.8]),
    (0.5, [float("nan"), 0.8]),
], ids=["nan-score", "inf-score", "nan-embed"])
def test_a_non_finite_forward_is_a_named_error_not_an_unnamed_null(score, embed) -> None:
    """Review blocker: json serializes NaN/inf as null, so a degenerate fp16 row used to leave the pod as
    `score: null, errors: null` — a failure with no name. It must be named, and the group still ok."""
    svc = _svc()
    svc._text_scores = lambda intent, ie: [0.6, score]
    res = svc._score_prepared(ClipRankGroup(intent="chart", image_urls=["a", "b"]), _Rows([[1.0], [1.0]]),
                              [[0.6, 0.8], embed], [0, 1], [None, None])
    assert res.status == "ok"
    assert res.scores == [0.6, None] and res.embeds == [[0.6, 0.8], None]
    assert res.errors[0] is None and res.errors[1].kind == "non_finite"
    body = json.loads(m.ClipRankPayload(model="x", groups=[res]).model_dump_json(exclude_none=True))
    assert not sorted(_CLIP_RANK_SCHEMA.iter_errors(body), key=str)


def test_a_group_whose_every_tile_is_non_finite_is_an_error() -> None:
    svc = _svc()
    svc._text_scores = lambda intent, ie: [float("nan"), float("nan")]
    res = svc._score_prepared(ClipRankGroup(intent="chart", image_urls=["a", "b"]), _Rows([[1.0], [1.0]]),
                              [[0.6, 0.8], [0.8, 0.6]], [0, 1], [None, None])
    assert res.status == "error" and res.error.kind == "non_finite"
    assert res.scores == [None, None] and res.embeds == [None, None]


@pytest.mark.parametrize(("scores", "embeds"), [
    ([float("nan")], [[0.1]]),
    ([0.3], [[float("inf")]]),
    ([0.3], [None]),
], ids=["nan-score", "inf-embed", "scored-tile-null-embed"])
def test_the_model_refuses_an_unnamed_failure(scores, embeds) -> None:
    from pydantic import ValidationError

    from podagent.models import ClipRankGroupResult

    with pytest.raises(ValidationError):
        ClipRankGroupResult(status="ok", scores=scores, embeds=embeds, errors=[None])
