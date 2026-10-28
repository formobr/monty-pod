"""clip_rank fetches stock-provider ORIGIN urls itself (MISC-333): through the one origin guard, once per url per
request, measuring each tile and naming every failure where the box can see it.

  • an origin tile goes originrule.fetch → workdir file → decode → forward; cp.download never sees it;
  • a guard refusal is `origin_refused`; other failures keep fetch_failed / fetch_timeout / decode_error;
  • `measures` is parallel to the tiles: {bytes, w, h} for a decoded tile, null otherwise;
  • a url in several groups of one request is fetched once;
  • each origin group puts one progress() line with its failure counts and first redacted url;
  • neither CLIP_RANK_FETCH_WORKERS nor CLIP_RANK_LANES is read.

Hermetic: the pool runs inline on the caller's thread, the network is a fake (or a refusal that happens
before any socket), and the towers are stubs.
"""
from __future__ import annotations

import concurrent.futures as cf
import http.client
import io
import json
import urllib.error
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

import podagent.infer_cliprank as m
from podagent import cp, originrule
from podagent.models import ClipRankGroup, ClipRankParams
from test_infer_cliprank import _svc

RESULT_SCHEMA = Draft202012Validator(json.loads(
    (Path(__file__).resolve().parents[1] / "contracts" / "clip_rank.schema.json").read_text()))

A = "https://images.stock.example/photo/1/thumb.jpg"
B = "https://images.stock.example/photo/2/thumb.jpg"
C = "https://images.stock.example/photo/3/thumb.jpg"


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


class _InlinePool:
    """submit() runs on the caller's thread — no thread timing in any verdict."""

    def submit(self, fn, *args, **kwargs):
        fut = cf.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 — mirror a real Future
            fut.set_exception(exc)
        return fut


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """Inline pool, a presigned download that must never run for an origin, a captured PUT, stub towers that
    score every decoded image 0.5."""
    monkeypatch.setattr(m, "_FETCH_POOL", _InlinePool())
    monkeypatch.setattr(m, "download", lambda url, dest: pytest.fail(f"cp.download fetched {url}"))
    put = tmp_path / "out.json"
    monkeypatch.setattr(m, "upload", lambda src, url, ct=None: put.write_bytes(src.read_bytes()))
    svc = _svc()
    svc._prepare = lambda images, ok: ((None, [], ok) if not images
                                       else (list(images), [[0.6, 0.8] for _ in images], ok))
    svc._text_scores = lambda intent, ie: [0.5] * len(ie)
    progress: list[str] = []

    def run(groups: list[ClipRankGroup]) -> dict:
        svc.run(ClipRankParams(groups=groups), "https://storage.example/o/r.json?sig=PUT", progress.append)
        body = json.loads(put.read_text())
        assert next(RESULT_SCHEMA.iter_errors(body), None) is None
        return body

    run.progress = progress
    return run


def _fake_fetch(monkeypatch, bodies: dict):
    """originrule.fetch stand-in: a body, or an exception to raise, per url; records every call."""
    calls: list[tuple[str, int, float]] = []

    def fetch(url, *, max_bytes, timeout_s):
        calls.append((url, max_bytes, timeout_s))
        out = bodies[url]
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr(m.originrule, "fetch", fetch)
    return calls


def test_origin_tiles_are_fetched_through_the_guard_and_scored(monkeypatch, rig) -> None:
    calls = _fake_fetch(monkeypatch, {A: _png(4, 3), B: _png(6, 5)})
    body = rig([ClipRankGroup(intent="chart", origin_urls=[A, B])])
    (g,) = body["groups"]
    assert g["status"] == "ok" and g["scores"] == [0.5, 0.5] and g["errors"] == [None, None]
    assert g["embeds"] == [[0.6, 0.8], [0.6, 0.8]]
    # the engine's THUMB_MAX_BYTES, and the same per-read timeout an image_urls tile's download runs under
    assert calls == [(A, 8_000_000, cp._TIMEOUT), (B, 8_000_000, cp._TIMEOUT)]


def test_a_refused_origin_is_named_origin_refused(monkeypatch, rig) -> None:
    # The REAL guard: both refusals happen before any lookup or socket (scheme, IP literal).
    bad_scheme = "http://images.stock.example/photo/1/thumb.jpg"
    metadata = "https://169.254.169.254/latest/meta-data/"
    real_fetch = originrule.fetch
    ok_png = _png(4, 3)
    monkeypatch.setattr(m.originrule, "fetch", lambda url, **kw: ok_png if url == A else real_fetch(url, **kw))
    body = rig([ClipRankGroup(intent="chart", origin_urls=[bad_scheme, A, metadata])])
    (g,) = body["groups"]
    assert g["status"] == "ok"
    assert [e and e["kind"] for e in g["errors"]] == ["origin_refused", None, "origin_refused"]
    assert g["scores"] == [None, 0.5, None]


def test_other_origin_failures_keep_their_existing_kinds(monkeypatch, rig) -> None:
    _fake_fetch(monkeypatch, {A: TimeoutError("read timed out"), B: OSError("connection reset"),
                              C: b"not an image"})
    (g,) = rig([ClipRankGroup(intent="chart", origin_urls=[A, B, C])])["groups"]
    assert g["status"] == "error"
    assert [e["kind"] for e in g["errors"]] == ["fetch_timeout", "fetch_failed", "decode_error"]


def test_measures_are_parallel_and_null_for_failed_tiles(monkeypatch, rig) -> None:
    good = _png(7, 5)
    _fake_fetch(monkeypatch, {A: good, B: originrule.OriginRefused("REFUSED: test"), C: b"junk"})
    (g,) = rig([ClipRankGroup(intent="chart", origin_urls=[A, B, C])])["groups"]
    assert g["measures"] == [{"bytes": len(good), "w": 7, "h": 5}, None, None]
    assert len(g["measures"]) == len(g["scores"]) == len(g["errors"])


def test_a_url_in_two_groups_is_fetched_once(monkeypatch, rig) -> None:
    calls = _fake_fetch(monkeypatch, {A: _png(4, 3), B: _png(4, 3), C: _png(4, 3)})
    body = rig([ClipRankGroup(intent="chart", origin_urls=[A, B]),
                ClipRankGroup(intent="crash", origin_urls=[B, C]),
                ClipRankGroup(intent="chart", origin_urls=[A, B])])
    assert [u for u, *_ in calls] == [A, B, C], "one fetch per url per REQUEST, across every group"
    assert [g["status"] for g in body["groups"]] == ["ok", "ok", "ok"]
    assert all(g["measures"] == [{"bytes": len(_png(4, 3)), "w": 4, "h": 3}] * 2 for g in body["groups"])


def test_failure_counts_reach_progress_without_query_strings(monkeypatch, rig) -> None:
    a_q = A + "?w=640&cache=XYZSECRET"
    b_q = B + "?w=640&cache=XYZSECRET"
    _fake_fetch(monkeypatch, {a_q: originrule.OriginRefused("REFUSED: test"), b_q: TimeoutError("slow"),
                              C: _png(4, 3)})
    rig([ClipRankGroup(intent="chart", origin_urls=[C]),
         ClipRankGroup(intent="chart", origin_urls=[a_q, b_q, C])])
    lines = [p for p in rig.progress if p.startswith("clip_rank origin group")]
    assert lines == [
        "clip_rank origin group 1/2: 0/1 tile(s) failed",
        "clip_rank origin group 2/2: 2/3 tile(s) failed (fetch_timeout=1, origin_refused=1), "
        "first https://images.stock.example/photo/1/thumb.jpg"
        " — causes: OriginRefused×1, TimeoutError×1 — hosts: images.stock.example 2/3",
    ]
    assert not any("XYZSECRET" in p or "?" in p for p in rig.progress)


def test_the_progress_line_names_causes_and_hosts_without_a_query_or_token(monkeypatch, rig) -> None:
    """PROD 2026-10-28 job 20168ee1: 538 tiles failed and the box could not say why."""
    def blocked(host: str, code: int, ray: str) -> urllib.error.HTTPError:
        err = urllib.error.HTTPError(f"https://{host}/x", 403, "Forbidden", http.client.HTTPMessage(), None)
        err.origin_failure = originrule.OriginFailure(
            phase="http", host=host, addr="104.18.1.1", family="v4", status=403, cf_ray=ray,
            cf_code=str(code), exc="HTTPError")
        return err

    pexels = [f"https://images.pexels.com/p/{i}.jpg?token=SIGNEDTOKEN{i}" for i in range(3)]
    pixabay = "https://cdn.pixabay.com/q.jpg?Signature=SIGNEDTOKEN9"
    good = "https://cdn.pixabay.com/ok.jpg"
    bodies: dict = {u: blocked("images.pexels.com", 1020, "r") for u in pexels}
    bodies[pixabay] = blocked("cdn.pixabay.com", 1010, "r")
    bodies[good] = _png(4, 3)
    _fake_fetch(monkeypatch, bodies)
    rig([ClipRankGroup(intent="chart", origin_urls=[*pexels, pixabay, good])])
    (line,) = [p for p in rig.progress if p.startswith("clip_rank origin group")]
    assert "causes: http HTTPError 403 cf-1020×3, http HTTPError 403 cf-1010×1" in line
    assert "hosts: cdn.pixabay.com 1/2, images.pexels.com 3/3" in line
    assert "SIGNEDTOKEN" not in line and "?" not in line


def test_an_image_urls_group_carries_no_measures_and_no_origin_line(monkeypatch, rig, tmp_path) -> None:
    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_png(4, 3))
        return dest

    monkeypatch.setattr(m, "download", fake_download)
    monkeypatch.setattr(m.originrule, "fetch", lambda *a, **k: pytest.fail("an image_urls tile hit the guard"))
    (g,) = rig([ClipRankGroup(intent="chart", image_urls=["a", "b"])])["groups"]
    assert "measures" not in g and g["status"] == "ok"
    assert not any(p.startswith("clip_rank origin group") for p in rig.progress)


def test_no_env_lever_is_read_for_width_or_lanes(monkeypatch) -> None:
    monkeypatch.setenv("CLIP_RANK_LANES", "64")
    monkeypatch.setenv("CLIP_RANK_FETCH_WORKERS", "64")
    monkeypatch.setattr(m, "_free_vram_mb", lambda: 3254.0)   # the dev 2060: one lane by VRAM
    monkeypatch.setattr(m, "_host_threads", lambda: 12)
    assert m.lane_width() == 1
    assert m.fetch_width() == 32
    monkeypatch.setattr(m, "_FETCH_POOL", None)
    pool = m._fetch_pool()
    try:
        assert pool._max_workers == 32
    finally:
        pool.shutdown(wait=False)
