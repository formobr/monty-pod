"""A clip_rank request may carry a judge-tiles request — (origin url, presigned PUT url) pairs — and its result
may carry one judge-tile result per pair plus the pod's own request timings, without a contracts VERSION bump
and without changing today's requests.

  • a request/payload carries EITHER >=1 group and no judge_tiles, OR groups == [] and >=1 judge tile;
  • an ok judge tile carries bytes, w, h and cache; an error one names its error and carries no size;
  • the rank payload carries the timings known before its own PUT (finite, never negative);
  • until the pod serves judge tiles, such a request is a NAMED error before any weights fetch or load.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from podagent import main as agent_main
from podagent.models import ClipRankJudgeTileResult, ClipRankParams, ClipRankPayload, InferRequest

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
EXAMPLES = CONTRACTS / "examples"
REQUEST_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "infer_request.schema.json").read_text()))
RESULT_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "clip_rank.schema.json").read_text()))

TILES = [{"origin_url": "https://images.stock.example/photo/1/thumb.jpg",
          "put_url": "https://storage.example/o/judge/1.jpg?sig=PUT"}]
GROUP = {"intent": "chart", "image_urls": ["https://storage.example/o/1.png?sig=GET"]}


def _load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text())


def _request(clip_rank: dict) -> dict:
    data = _load("infer_request.clip_rank.json")
    data["clip_rank"] = clip_rank
    return data


def _reject_non_finite(token: str) -> None:
    raise ValueError(f"{token} is not valid JSON (no NaN/Infinity literal)")


def _schema_ok(validator: Draft202012Validator, data: dict) -> bool:
    # Round-trip through strict JSON text first: standard JSON has no NaN/Infinity literal at all, and
    # Draft202012Validator's own numeric keywords don't catch a bare Python float('nan') either — `nan < 0`
    # and `nan >= 0` are both False, so `minimum: 0` lets it through silently if handed the raw object.
    try:
        as_json = json.loads(json.dumps(data, allow_nan=True), parse_constant=_reject_non_finite)
    except ValueError:
        return False
    return next(validator.iter_errors(as_json), None) is None


def _model_ok(model, data: dict) -> bool:
    try:
        model.model_validate(data)
    except ValidationError:
        return False
    return True


def test_a_judge_tiles_request_validates_with_empty_groups() -> None:
    data = _request({"groups": [], "judge_tiles": TILES})
    assert _schema_ok(REQUEST_SCHEMA, data)
    req = InferRequest.model_validate(data)
    assert req.clip_rank is not None and req.clip_rank.groups == []
    assert [t.origin_url for t in req.clip_rank.judge_tiles or []] == [TILES[0]["origin_url"]]
    # today's rank request is untouched: groups and no judge_tiles
    today = _load("infer_request.clip_rank.json")
    assert _schema_ok(REQUEST_SCHEMA, today) and _model_ok(InferRequest, today)
    assert InferRequest.model_validate(today).clip_rank.judge_tiles is None  # type: ignore[union-attr]
    # empty groups without judge tiles stays invalid, and so does an empty judge_tiles list
    for bad in ({"groups": []}, {"groups": [], "judge_tiles": []}):
        assert not _schema_ok(REQUEST_SCHEMA, _request(bad))
        assert not _model_ok(InferRequest, _request(bad))
    assert not _model_ok(InferRequest, _load("invalid/infer_request.clip_rank-no-groups.json"))
    # a judge tile needs both urls
    for half in ({"origin_url": TILES[0]["origin_url"]}, {"put_url": TILES[0]["put_url"]},
                 {"origin_url": "", "put_url": TILES[0]["put_url"]}):
        assert not _schema_ok(REQUEST_SCHEMA, _request({"groups": [], "judge_tiles": [half]}))
        assert not _model_ok(InferRequest, _request({"groups": [], "judge_tiles": [half]}))
    # an explicit null disagrees with the schema (no type for null here) exactly like the schema does —
    # it must fail, never be read as "absent"
    null_request = _request({"groups": [GROUP], "judge_tiles": None})
    assert not _schema_ok(REQUEST_SCHEMA, null_request)
    assert not _model_ok(InferRequest, null_request)
    null_payload = {**_load("clip_rank.payload.json"), "judge_tiles": None}
    assert not _schema_ok(RESULT_SCHEMA, null_payload)
    assert not _model_ok(ClipRankPayload, null_payload)


def test_groups_and_judge_tiles_together_are_refused() -> None:
    data = _request({"groups": [GROUP], "judge_tiles": TILES})
    assert not _schema_ok(REQUEST_SCHEMA, data)
    with pytest.raises(ValidationError, match="never groups and judge_tiles together"):
        ClipRankParams.model_validate(data["clip_rank"])
    rank = _load("clip_rank.payload.json")
    tile = {"status": "ok", "bytes": 1, "w": 2, "h": 3, "cache": "miss"}
    both = {**rank, "judge_tiles": [tile]}
    assert not _schema_ok(RESULT_SCHEMA, both)
    assert not _model_ok(ClipRankPayload, both)
    neither = {"model": rank["model"], "groups": []}
    assert not _schema_ok(RESULT_SCHEMA, neither)
    assert not _model_ok(ClipRankPayload, neither)


def test_an_ok_judge_tile_needs_bytes_and_size() -> None:
    ok = {"status": "ok", "bytes": 1200, "w": 384, "h": 216, "cache": "hit"}
    err = {"status": "error", "error": {"kind": "put_failed", "detail": "HTTPError: 403"}}
    for tile in (ok, err):
        payload = {"model": "m", "groups": [], "judge_tiles": [tile]}
        assert _schema_ok(RESULT_SCHEMA, payload) and _model_ok(ClipRankPayload, payload)
    bad_tiles = [
        *({k: v for k, v in ok.items() if k != drop} for drop in ("bytes", "w", "h", "cache")),
        {**ok, "w": 0},
        {**ok, "error": err["error"]},
        {"status": "error"},
        {**err, "bytes": 10},
        {**ok, "cache": "warm"},
    ]
    for tile in bad_tiles:
        payload = {"model": "m", "groups": [], "judge_tiles": [tile]}
        assert not _schema_ok(RESULT_SCHEMA, payload), tile
        assert not _model_ok(ClipRankPayload, payload), tile
    with pytest.raises(ValidationError, match="bytes, w, h and cache"):
        ClipRankJudgeTileResult.model_validate({k: v for k, v in ok.items() if k != "bytes"})


def test_the_rank_payload_carries_its_timings(monkeypatch, tmp_path) -> None:
    import types

    from podagent import infer_cliprank as m
    from podagent.models import ClipRankGroup

    timings = {"infer_s": 1.5, "tile_gather_work_s": 0.5, "unique_image_sets_n": 1.0}
    payload = {**_load("clip_rank.payload.json"), "timings": timings}
    assert _schema_ok(RESULT_SCHEMA, payload)
    assert ClipRankPayload.model_validate(payload).timings == timings
    for bad in (-0.1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError, match="finite and >= 0"):
            ClipRankPayload.model_validate({**payload, "timings": {"infer_s": bad}})
        assert not _schema_ok(RESULT_SCHEMA, {**payload, "timings": {"infer_s": bad}})

    # the service puts the timings known before its own PUT into the result the box reads
    svc = m.ClipRankService.__new__(m.ClipRankService)
    svc.model_id, svc.parallel = "siglip", 1
    svc._slots = types.SimpleNamespace(acquire=lambda timeout: True, release=lambda: None)
    svc._prepare = lambda images, ok: (["emb"], ok)
    svc._score_prepared = lambda g, embeds, ok, errors, measures=None: m.ClipRankGroupResult(
        status="ok", scores=[0.5], embeds=[[1.0]], errors=[None])
    monkeypatch.setattr(m, "_submit_tiles", lambda urls, workdir: urls)
    monkeypatch.setattr(m, "_gather", lambda pending, cells: (["img"], [True], [None]))
    put = tmp_path / "clip_rank.json"
    monkeypatch.setattr(m, "upload", lambda src, url, ct=None: put.write_bytes(src.read_bytes()))
    run = svc.run(ClipRankParams(groups=[ClipRankGroup(**GROUP)]), "https://storage.example/o/r.json?sig=PUT")
    body = json.loads(put.read_text())
    assert _schema_ok(RESULT_SCHEMA, body)
    assert set(body["timings"]) == {"infer_s", "tile_gather_work_s", "forward_work_s",
                                    "unique_image_sets_n", "reused_groups_n", "parallel_width_n"}
    for key in ("unique_image_sets_n", "reused_groups_n", "parallel_width_n"):
        assert body["timings"][key] == run.timings[key]
    assert all(v >= 0 for v in body["timings"].values())


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.results: list[dict] = []

    def send_event(self, ev: dict, *, wait: bool = False) -> bool:
        self.events.append(ev)
        return True

    def note(self, ev: dict) -> None:
        self.events.append(ev)

    def report_infer_result(self, payload: dict, wake=None) -> bool:
        self.results.append(payload)
        return True


@pytest.mark.parametrize("dry", [False, True], ids=["full", "dry"])
def test_a_judge_tiles_request_is_a_named_error_without_a_weights_load(monkeypatch, tmp_path, dry) -> None:
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")
    touched: list[str] = []
    monkeypatch.setattr("podagent.ops.dry.armed", lambda: dry)
    monkeypatch.setattr("podagent.ops.dry.run_clip_rank", lambda *a, **k: touched.append("dry"))
    monkeypatch.setattr("podagent.weights.ensure", lambda *a, **k: touched.append("fetch") or tmp_path)
    monkeypatch.setattr("podagent.infer_cliprank.ClipRankService", lambda *a, **k: touched.append("load"))
    align_cache = {"a" * 64: object()}
    cp = _CP()
    agent_main._run_infer(_load("infer_request.clip_rank_judge_tiles.json"), cp, align_cache, {}, {},
                          Path("/opt/models/yunet.onnx"), True, corr_id="c", session_id="s",
                          one_kind_resident=True)
    assert touched == []
    assert list(align_cache) == ["a" * 64], "no resident weights are dropped for a request the pod refuses"
    (res,) = cp.results
    assert res["status"] == "error" and res["kind"] == "clip_rank" and res["corr_id"] == "c"
    assert agent_main.JUDGE_TILES_NOT_SERVED in res["error"]
    assert agent_main.JUDGE_TILES_NOT_SERVED == "judge_tiles are not served by this image"


@pytest.mark.parametrize("name, model", [
    ("infer_request.clip_rank_judge_tiles.json", InferRequest),
    ("clip_rank.payload_judge_tiles.json", ClipRankPayload),
    ("clip_rank.payload_timings.json", ClipRankPayload),
])
def test_schema_and_model_agree_on_the_new_examples(name, model) -> None:
    validator = REQUEST_SCHEMA if name.startswith("infer_request") else RESULT_SCHEMA
    data = _load(name)
    assert _schema_ok(validator, data)
    dumped = model.model_validate(data).model_dump(exclude_none=True, mode="json")
    assert _schema_ok(validator, dumped)
    assert dumped == data
    for bad_name, bad_model, bad_validator in (
            ("invalid/infer_request.clip_rank-groups-with-judge-tiles.json", InferRequest, REQUEST_SCHEMA),
            ("invalid/clip_rank.judge-tile-ok-without-bytes.json", ClipRankPayload, RESULT_SCHEMA),
            ("invalid/infer_request.clip_rank-no-groups.json", InferRequest, REQUEST_SCHEMA)):
        assert not _schema_ok(bad_validator, _load(bad_name))
        assert not _model_ok(bad_model, _load(bad_name))
    assert (CONTRACTS / "VERSION").read_text().strip() == "7"


def _contracts_validate():
    import importlib.util

    spec = importlib.util.spec_from_file_location("contracts_validate", CONTRACTS / "validate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OK_TILE = {"status": "ok", "bytes": 1200, "w": 384, "h": 216, "cache": "hit"}
_ERR_TILE = {"status": "error", "error": {"kind": "put_failed", "detail": "HTTPError: 403"}}
_RANK_GROUP = {"status": "ok", "scores": [0.5], "embeds": [[1.0]], "errors": [None]}


def _tiles(*tiles: dict) -> dict:
    return {"model": "m", "groups": [], "judge_tiles": list(tiles)}


def _ranked(**extra) -> dict:
    return {"model": "m", "groups": [_RANK_GROUP], **extra}


# (name, schema prefix, document) — every shape gets ONE verdict, from the model and from the schema alike.
SHAPE_MATRIX = [
    ("ok tile", "clip_rank", _tiles(_OK_TILE)),
    ("error tile", "clip_rank", _tiles(_ERR_TILE)),
    ("ok tile with error null", "clip_rank", _tiles({**_OK_TILE, "error": None})),
    ("error tile with bytes null", "clip_rank", _tiles({**_ERR_TILE, "bytes": None})),
    ("ok tile without bytes", "clip_rank", _tiles({k: v for k, v in _OK_TILE.items() if k != "bytes"})),
    ("ok tile with integral float size", "clip_rank", _tiles({**_OK_TILE, "w": 384.0})),
    ("ok tile with fractional size", "clip_rank", _tiles({**_OK_TILE, "w": 384.5})),
    *((f"ok tile with bool {k}", "clip_rank", _tiles({**_OK_TILE, k: True})) for k in ("bytes", "w", "h")),
    *((f"ok tile with string {k}", "clip_rank", _tiles({**_OK_TILE, k: "5"})) for k in ("bytes", "w", "h")),
    ("timings numbers", "clip_rank", _ranked(timings={"infer_s": 1.5, "groups_n": 3})),
    ("timings bool", "clip_rank", _ranked(timings={"infer_s": True})),
    ("timings false", "clip_rank", _ranked(timings={"infer_s": False})),
    ("timings string", "clip_rank", _ranked(timings={"infer_s": "1.5"})),
    ("timings null value", "clip_rank", _ranked(timings={"infer_s": None})),
    ("timings null", "clip_rank", _ranked(timings=None)),
    ("timings negative", "clip_rank", _ranked(timings={"infer_s": -1})),
    ("timings nan", "clip_rank", _ranked(timings={"infer_s": float("nan")})),
    ("timings infinity", "clip_rank", _ranked(timings={"infer_s": float("inf")})),
    ("timings -infinity", "clip_rank", _ranked(timings={"infer_s": float("-inf")})),
    ("ok tile with nan bytes", "clip_rank", _tiles({**_OK_TILE, "bytes": float("nan")})),
    ("payload groups with judge_tiles", "clip_rank", _ranked(judge_tiles=[_OK_TILE])),
    ("payload empty groups without judge_tiles", "clip_rank", {"model": "m", "groups": []}),
    ("request rank", "infer_request", _request({"groups": [GROUP]})),
    ("request judge tiles", "infer_request", _request({"groups": [], "judge_tiles": TILES})),
    ("request groups with judge_tiles", "infer_request", _request({"groups": [GROUP], "judge_tiles": TILES})),
    ("request empty groups without judge_tiles", "infer_request", _request({"groups": []})),
]
_SHAPE_MODELS = {"clip_rank": ClipRankPayload, "infer_request": InferRequest}


def test_model_and_schema_agree_on_a_shape_matrix() -> None:
    validate = _contracts_validate()
    verdicts = {name: (validate.accepts(prefix, data), _model_ok(_SHAPE_MODELS[prefix], data))
                for name, prefix, data in SHAPE_MATRIX}
    disagree = {name: v for name, v in verdicts.items() if v[0] != v[1]}
    assert disagree == {}, f"(schema, model) verdicts differ: {disagree}"
    # the matrix is not vacuous: the shapes the goal names are refused by both, the plain ones accepted
    refused = {"ok tile with error null", "ok tile without bytes", "timings bool", "timings string",
               "ok tile with bool bytes", "ok tile with string w", "payload groups with judge_tiles",
               "payload empty groups without judge_tiles", "request groups with judge_tiles",
               "request empty groups without judge_tiles", "timings nan", "timings infinity",
               "timings -infinity", "ok tile with nan bytes"}
    accepted = {"ok tile", "error tile", "timings numbers", "request rank", "request judge tiles"}
    assert all(verdicts[n] == (False, False) for n in refused)
    assert all(verdicts[n] == (True, True) for n in accepted)
