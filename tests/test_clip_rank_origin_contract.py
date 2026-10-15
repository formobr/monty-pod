"""A clip_rank group may carry stock-provider ORIGIN urls instead of presigned urls, and a group result may
carry each tile's measured size — without a contracts VERSION bump and without changing today's requests.

  • a group carries EXACTLY ONE of image_urls / origin_urls; image_cells only ride with image_urls;
  • `measures`, when present, is parallel to the tiles; absent unless asked (today's results are unchanged);
  • an origin group is never fetched under the presigned-only rule (cp.download): it goes through the
    origin guard (tests/test_clip_rank_origin_fetch.py pins that path), and only it carries `measures`;
  • the dry stub sizes an origin group by its origin_urls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from podagent.models import (
    ClipRankGroup,
    ClipRankGroupResult,
    ClipRankParams,
    ClipRankPayload,
    InferRequest,
)

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
EXAMPLES = CONTRACTS / "examples"
REQUEST_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "infer_request.schema.json").read_text()))
RESULT_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "clip_rank.schema.json").read_text()))

PRESIGNED = ["https://storage.example/o/1.png?sig=GET", "https://storage.example/o/2.png?sig=GET"]
ORIGINS = ["https://images.stock.example/photo/1/thumb.jpg", "https://images.stock.example/photo/2/thumb.jpg"]


def _request(groups: list[dict]) -> dict:
    data = json.loads((EXAMPLES / "infer_request.clip_rank.json").read_text())
    data["clip_rank"]["groups"] = groups
    return data


def _schema_ok(validator: Draft202012Validator, data: dict) -> bool:
    return next(validator.iter_errors(data), None) is None


def _model_ok(model, data: dict) -> bool:
    try:
        model.model_validate(data)
    except ValidationError:
        return False
    return True


def test_exactly_one_url_kind_per_group() -> None:
    assert ClipRankGroup(intent="x", image_urls=PRESIGNED).tile_urls == PRESIGNED
    assert ClipRankGroup(intent="x", origin_urls=ORIGINS).tile_urls == ORIGINS
    with pytest.raises(ValidationError, match="exactly one"):
        ClipRankGroup(intent="x", image_urls=PRESIGNED, origin_urls=ORIGINS)
    with pytest.raises(ValidationError, match="exactly one"):
        ClipRankGroup(intent="x")
    with pytest.raises(ValidationError):
        ClipRankGroup(intent="x", origin_urls=[])
    with pytest.raises(ValidationError):
        ClipRankGroup(intent="x", image_urls=[])


def test_cells_are_refused_on_origin_groups() -> None:
    cell = (0, 0, 10, 10, 20, 20)
    assert ClipRankGroup(intent="x", image_urls=PRESIGNED[:1], image_cells=[cell]).image_cells == [cell]
    with pytest.raises(ValidationError, match="only with image_urls"):
        ClipRankGroup(intent="x", origin_urls=ORIGINS[:1], image_cells=[cell])
    data = _request([{"intent": "x", "origin_urls": ORIGINS[:1], "image_cells": [list(cell)]}])
    assert not _schema_ok(REQUEST_SCHEMA, data)
    assert not _model_ok(InferRequest, data)


def test_todays_requests_and_results_still_validate() -> None:
    for name, model, validator in (("infer_request.clip_rank.json", InferRequest, REQUEST_SCHEMA),
                                   ("clip_rank.payload.json", ClipRankPayload, RESULT_SCHEMA)):
        raw = (EXAMPLES / name).read_text()
        data = json.loads(raw)
        assert _schema_ok(validator, data), name
        instance = model.model_validate(data)
        # Byte-for-byte: re-serializing today's shape adds no new key (no origin_urls, no measures).
        assert json.loads(instance.model_dump_json(exclude_none=True)) == data, name
    payload = ClipRankPayload.model_validate_json((EXAMPLES / "clip_rank.payload.json").read_text())
    assert all(g.measures is None for g in payload.groups)


def test_measures_must_be_parallel_to_tiles() -> None:
    base = dict(status="ok", scores=[0.1, 0.2], embeds=[[0.1], [0.2]], errors=[None, None])
    ok = ClipRankGroupResult(**base, measures=[{"bytes": 0, "w": 4, "h": 5}, None])
    assert ok.measures[0].w == 4 and ok.measures[1] is None
    with pytest.raises(ValidationError, match="parallel"):
        ClipRankGroupResult(**base, measures=[{"bytes": 10, "w": 4, "h": 5}])
    for bad in ({"bytes": -1, "w": 4, "h": 5}, {"bytes": 1, "w": 0, "h": 5}, {"bytes": 1, "w": 4, "h": 0},
                {"bytes": 1, "w": 4}, {"bytes": 1, "w": 4, "h": 5, "extra": 1}):
        with pytest.raises(ValidationError):
            ClipRankGroupResult(**base, measures=[bad, None])
        assert not _schema_ok(RESULT_SCHEMA, {"model": "m", "groups": [{**base, "measures": [bad, None]}]})


@pytest.mark.parametrize("group,valid", [
    ({"intent": "x", "image_urls": PRESIGNED}, True),
    ({"intent": "x", "origin_urls": ORIGINS}, True),
    ({"intent": "", "origin_urls": ORIGINS}, True),
    ({"intent": "x", "image_urls": PRESIGNED[:1], "image_cells": [[0, 0, 1, 1, 2, 2]]}, True),
    ({"intent": "x", "image_urls": PRESIGNED, "origin_urls": ORIGINS}, False),
    ({"intent": "x"}, False),
    ({"intent": "x", "origin_urls": []}, False),
    ({"intent": "x", "origin_urls": ORIGINS[:1], "image_cells": [[0, 0, 1, 1, 2, 2]]}, False),
    ({"intent": "x", "origin_urls": ORIGINS[:1], "image_cells": [None]}, False),
])
def test_schema_and_models_agree_on_both_shapes(group: dict, valid: bool) -> None:
    data = _request([group])
    assert _schema_ok(REQUEST_SCHEMA, data) is valid
    assert _model_ok(InferRequest, data) is valid


def test_an_origin_group_never_goes_through_the_presigned_download(monkeypatch, tmp_path) -> None:
    import podagent.infer_cliprank as m
    from podagent import originrule
    from test_infer_cliprank import _svc

    fetched: list[str] = []

    def fake_download(url, dest):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"img")
        return dest

    def refuse(url, *, max_bytes, timeout_s):
        raise originrule.OriginRefused("REFUSED: test")

    monkeypatch.setattr(m, "download", fake_download)
    monkeypatch.setattr(m.originrule, "fetch", refuse)
    monkeypatch.setattr("PIL.Image.open", lambda p: type("I", (), {"convert": lambda self, mode: "img"})())
    put = tmp_path / "out.json"
    monkeypatch.setattr(m, "upload", lambda src, url, ct=None: put.write_bytes(src.read_bytes()))

    params = ClipRankParams(groups=[ClipRankGroup(intent="chart", origin_urls=ORIGINS),
                                    ClipRankGroup(intent="chart", image_urls=["a", "b"])])
    _svc().run(params, "put")

    assert fetched == ["a", "b"], "an origin url is never fetched as presigned"
    body = json.loads(put.read_text())
    assert _schema_ok(RESULT_SCHEMA, body)
    origin, presigned = body["groups"]
    assert origin["status"] == "error" and origin["error"]["kind"] == "origin_refused"
    assert origin["scores"] == [None, None] and origin["embeds"] == [None, None]
    assert [e["kind"] for e in origin["errors"]] == ["origin_refused"] * 2
    assert origin["measures"] == [None, None]
    assert presigned == {"status": "ok", "scores": [0.6, 0.8], "embeds": [[0.6, 0.8], [0.8, 0.6]],
                         "errors": [None, None]}, "an image_urls group carries no new field"

    single = _svc()._run_group(ClipRankGroup(intent="", origin_urls=ORIGINS[:1]), tmp_path / "g")
    assert single.status == "error" and [e.kind for e in single.errors] == ["origin_refused"]


def test_the_dry_stub_sizes_origin_groups(monkeypatch) -> None:
    from podagent import cp as _cp
    from podagent.ops import dry

    uploaded = {}
    monkeypatch.setattr(_cp, "upload", lambda src, _url, _ct=None: uploaded.update(body=json.loads(src.read_text())))
    params = ClipRankParams(groups=[ClipRankGroup(intent="crypto", origin_urls=ORIGINS + ORIGINS[:1]),
                                    ClipRankGroup(intent="finance", image_urls=PRESIGNED[:1])])
    dry.run_clip_rank(params, "https://put.example/clip_rank.json")
    groups = uploaded["body"]["groups"]
    assert [len(g["scores"]) for g in groups] == [3, 1]
    assert all(len(g["embeds"]) == len(g["errors"]) == len(g["scores"]) for g in groups)
