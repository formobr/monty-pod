"""The contour-dry stub answers a judge-tiles request and an origin group the way the real pod does (MISC-593).

  • a judge tile is a deterministic 512x512 JPEG (solid colour from its origin url, quality 82) PUT to its own url
    as image/jpeg, answered `ok` with that JPEG's bytes/w/h and `cache: hit`, then ONE result JSON PUT;
  • an origin group carries one deterministic measure per tile and the payload carries its timings; an image_urls
    group keeps today's deterministic scores and no measures;
  • main's dry branch sends a judge-tiles request to the stub — no weights fetch, no load, no origin fetch;
  • every stub progress line starts «contour-dry clip_rank stub», the mark the engine recognises a stub run by.

Hermetic: every PUT is captured by a fake upload, and no verdict depends on a clock or a thread's timing.
"""
from __future__ import annotations

import builtins
import io
import json
import threading
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from podagent import cp
from podagent import main as agent_main
from podagent.models import ClipRankGroup, ClipRankJudgeTile, ClipRankParams
from podagent.ops import dry

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
RESULT_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "clip_rank.schema.json").read_text()))
RESULT_URL = "https://storage.example/o/r.json?sig=PUT"
# Vendored from the engine's infer_cassette.DRY_STUB_MARKS (no engine import — this repo ships without it).
ENGINE_STUB_MARK = "contour-dry clip_rank stub"

A = "https://images.stock.example/photo/1/thumb.jpg"
B = "https://images.stock.example/photo/2/thumb.jpg"
C = "https://images.stock.example/photo/3/thumb.jpg"


def _put_url(i: int) -> str:
    return f"https://storage.example/o/judge/{i}.jpg?sig=PUT"


@pytest.fixture
def puts(monkeypatch) -> dict[str, tuple[str | None, bytes]]:
    """url -> (content type, body) for every PUT the stub makes; the tile PUTs come from pool threads."""
    seen: dict[str, tuple[str | None, bytes]] = {}
    lock = threading.Lock()

    def upload(src, url, content_type=None, **_kw):
        with lock:
            assert url not in seen, f"PUT twice to {url}"
            seen[url] = (content_type, Path(src).read_bytes())

    monkeypatch.setattr(cp, "upload", upload)
    return seen


def _judge_params() -> ClipRankParams:
    return ClipRankParams(groups=[], judge_tiles=[ClipRankJudgeTile(origin_url=u, put_url=_put_url(i))
                                                  for i, u in enumerate((A, B, C))])


def test_dry_judge_tiles_put_one_jpeg_per_tile(puts) -> None:
    dry.run_judge_tiles(_judge_params(), RESULT_URL)
    assert set(puts) == {_put_url(0), _put_url(1), _put_url(2), RESULT_URL}
    ct, body = puts[RESULT_URL]
    assert ct == "application/json"
    payload = json.loads(body)
    RESULT_SCHEMA.validate(payload)
    assert payload["groups"] == [] and len(payload["judge_tiles"]) == 3
    colours = []
    for i, res in enumerate(payload["judge_tiles"]):
        ct, jpeg = puts[_put_url(i)]
        assert ct == "image/jpeg"
        img = Image.open(io.BytesIO(jpeg))
        assert img.format == "JPEG" and img.size == (512, 512)
        assert res == {"status": "ok", "bytes": len(jpeg), "w": 512, "h": 512, "cache": "hit"}
        colours.append(img.convert("RGB").getpixel((256, 256)))
    assert len(set(colours)) == 3, "two candidates sharing one tile would hide a mis-addressed tile"
    assert payload["timings"]["judge_tiles_n"] == 3.0
    first = {u: b for u, (_, b) in puts.items() if u != RESULT_URL}
    puts.clear()
    dry.run_judge_tiles(_judge_params(), RESULT_URL)
    assert {u: b for u, (_, b) in puts.items() if u != RESULT_URL} == first, "a dry tile is deterministic"


def test_a_failed_put_is_a_named_tile_result_not_a_raise(monkeypatch) -> None:
    """One tile's PUT failing must not cost the judge the others, nor the result JSON — same contract as
    the real pod (infer_cliprank._judge_tile)."""
    seen: dict[str, tuple[str | None, bytes]] = {}

    def upload(src, url, content_type=None, **_kw):
        if url == _put_url(1):
            raise RuntimeError("PUT rejected: HTTP 403")
        seen[url] = (content_type, Path(src).read_bytes())

    monkeypatch.setattr(cp, "upload", upload)
    dry.run_judge_tiles(_judge_params(), RESULT_URL)
    assert set(seen) == {_put_url(0), _put_url(2), RESULT_URL}, "the other tiles and the result JSON still PUT"
    payload = json.loads(seen[RESULT_URL][1])
    RESULT_SCHEMA.validate(payload)
    ok0, bad, ok2 = payload["judge_tiles"]
    assert ok0["status"] == "ok" and ok2["status"] == "ok"
    assert bad["status"] == "error" and bad["error"]["kind"] == "put_failed"
    assert set(bad) == {"status", "error"}, "a failure is never a size"
    assert "sig=" not in bad["error"]["detail"], "the presigned PUT url's capability must not leak"


def test_dry_origin_groups_carry_measures(puts) -> None:
    params = ClipRankParams(groups=[
        ClipRankGroup(intent="crypto", origin_urls=[A, B]),
        ClipRankGroup(intent="finance", image_urls=["https://storage.example/o/t/1.jpg?sig=GET"]),
    ])
    dry.run_clip_rank(params, RESULT_URL)
    ct, body = puts[RESULT_URL]
    assert ct == "application/json" and list(puts) == [RESULT_URL]
    payload = json.loads(body)
    RESULT_SCHEMA.validate(payload)
    origin, image = payload["groups"]
    assert origin["measures"] == [{"bytes": 40000, "w": 512, "h": 512}] * 2
    assert "measures" not in image
    scores, embeds = dry._clip_rank_group_verdict(2)
    assert origin["scores"] == scores and origin["embeds"] == embeds
    assert image["scores"] == dry._clip_rank_group_verdict(1)[0]
    assert "infer_s" in payload["timings"]


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


def test_dry_judge_tiles_need_no_weights(monkeypatch, tmp_path, puts) -> None:
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")
    monkeypatch.setattr(dry, "armed", lambda: True)
    touched: list[str] = []
    monkeypatch.setattr("podagent.weights.ensure", lambda *a, **k: touched.append("fetch") or tmp_path)
    monkeypatch.setattr("podagent.infer_cliprank.ClipRankService", lambda *a, **k: touched.append("load"))
    monkeypatch.setattr("podagent.infer_cliprank.run_judge_tiles", lambda *a, **k: touched.append("real"))
    monkeypatch.setattr("podagent.infer_lanes.release_other_kinds", lambda *a, **k: touched.append("release"))
    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch" or name.startswith("transformers"):
            raise AssertionError(f"the dry judge-tiles stub imported {name!r} — it must never touch the card")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    raw = json.loads((CONTRACTS / "examples" / "infer_request.clip_rank_judge_tiles.json").read_text())
    cp_ = _CP()
    agent_main._run_infer(raw, cp_, {}, {}, {}, Path("/opt/models/yunet.onnx"), True,
                          corr_id="c", session_id="s", one_kind_resident=True)
    assert touched == []
    tiles = raw["clip_rank"]["judge_tiles"]
    assert set(puts) == {t["put_url"] for t in tiles} | {raw["put_url"]}
    assert all(puts[t["put_url"]][0] == "image/jpeg" for t in tiles)
    (res,) = cp_.results
    assert res["status"] == "ok" and res["kind"] == "clip_rank"


def test_the_stub_line_still_names_contour_dry(puts) -> None:
    lines: list[str] = []
    dry.run_judge_tiles(_judge_params(), RESULT_URL, lines.append)
    dry.run_clip_rank(ClipRankParams(groups=[ClipRankGroup(intent="x", origin_urls=[A])]),
                      "https://storage.example/o/r2.json?sig=PUT", lines.append)
    assert len(lines) == 2 and all(line.startswith(ENGINE_STUB_MARK) for line in lines), lines
