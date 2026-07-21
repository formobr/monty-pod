"""ClipRankService — the SigLIP forward, hermetic (stub towers, no weights, no GPU, no network).

What the pod side must guarantee, and what each test pins:
  • scores AND embeds come out of ONE forward, 4dp, in the request's image order;
  • intent "" is embed-only: the TEXT tower is never touched and every score is -1.0;
  • an image the pod cannot fetch scores -1.0/None instead of failing a whole batch of beats;
  • nothing in the result carries a ranking or a threshold — the payload is numbers only.
"""
from __future__ import annotations

import contextlib
import types
from pathlib import Path

import pytest

from podagent.infer_cliprank import ClipRankService
from podagent.models import ClipRankGroup


class _Vec:
    def __init__(self, rows):
        self.rows = rows

    def __matmul__(self, other):
        return _Vec([[sum(a * b for a, b in zip(r, c)) for c in other.rows] for r in self.rows])

    @property
    def T(self):
        return _Vec([list(c) for c in zip(*self.rows)])

    def squeeze(self, _d):
        return _Vec([r[0] for r in self.rows]) if self.rows and isinstance(self.rows[0], list) else self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class _In(dict):
    def to(self, _dev):
        return self


class _Torch:
    float32 = "f32"
    float16 = "f16"

    class nn:
        class functional:
            @staticmethod
            def normalize(x, dim=-1):
                return x

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()


def _svc(text_calls: list | None = None) -> ClipRankService:
    """A service with stub towers — bypasses __init__ so no weights load."""
    svc = ClipRankService.__new__(ClipRankService)
    svc.torch = _Torch()
    svc.device = "cpu"
    svc.dtype = _Torch.float32
    svc.model_id = "google/siglip2-so400m-patch14-384"
    svc.proc = lambda **kw: _In()

    class _Model:
        def get_image_features(self, **kw):
            return _Vec([[0.6, 0.8], [0.8, 0.6]])

        def get_text_features(self, **kw):
            if text_calls is not None:
                text_calls.append(kw)
            return _Vec([[1.0, 0.0]])

    svc.model = _Model()
    return svc


def test_one_forward_yields_scores_and_embeds() -> None:
    scores, embeds = _svc()._forward("a falling crypto chart", ["img", "img"])
    assert scores == [0.6, 0.8]
    assert embeds == [[0.6, 0.8], [0.8, 0.6]]


def test_empty_intent_is_embed_only_and_never_touches_the_text_tower() -> None:
    calls: list = []
    scores, embeds = _svc(calls)._forward("", ["img", "img"])
    assert scores == [-1.0, -1.0], "an embed-only group's scores have nothing to mean"
    assert embeds == [[0.6, 0.8], [0.8, 0.6]], "the image tower is text-independent — embeds still come back"
    assert calls == [], "no intent must mean no text forward"


def test_feat_unwraps_a_pooled_model_output_so_the_forward_never_crashes() -> None:
    """Regression: some transformers versions return a BaseModelOutputWithPooling (pooled embed in
    .pooler_output) from get_image/text_features instead of a bare tensor; F.normalize then calls .norm() on
    the OBJECT — the '...has no attribute norm' crash that lost every b-roll. _feat must unwrap .pooler_output
    so the forward runs. WITHOUT the guard this test raises AttributeError (the object has no .float())."""
    class _Pooled:
        def __init__(self, t):
            self.pooler_output = t

    svc = _svc()
    svc.model.get_image_features = lambda **kw: _Pooled(_Vec([[0.6, 0.8], [0.8, 0.6]]))
    svc.model.get_text_features = lambda **kw: _Pooled(_Vec([[1.0, 0.0]]))
    scores, embeds = svc._forward("some intent", ["img", "img"])
    assert scores == [0.6, 0.8]
    assert embeds == [[0.6, 0.8], [0.8, 0.6]]


def test_unfetchable_image_scores_minus_one_and_keeps_request_order(monkeypatch, tmp_path) -> None:
    import podagent.infer_cliprank as m

    def fake_download(url: str, dest: Path) -> Path:
        if url == "dead":
            raise OSError("404 on presigned GET")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"img")
        return dest

    monkeypatch.setattr(m, "download", fake_download)
    monkeypatch.setattr("PIL.Image.open", lambda p: types.SimpleNamespace(convert=lambda mode: "img"))

    group = ClipRankGroup(intent="chart", image_urls=["ok1", "dead", "ok2"])
    res = _svc()._run_group(group, tmp_path / "g0")

    assert res.scores == [0.6, -1.0, 0.8], "a dead tile sorts last; the live ones keep their slots"
    assert res.embeds == [[0.6, 0.8], None, [0.8, 0.6]]


def test_a_wholly_dead_group_returns_misses_without_a_forward(monkeypatch, tmp_path) -> None:
    import podagent.infer_cliprank as m

    monkeypatch.setattr(m, "download", lambda url, dest: (_ for _ in ()).throw(OSError("gone")))
    svc = _svc()
    svc.model = None  # any forward attempt would explode

    res = svc._run_group(ClipRankGroup(intent="chart", image_urls=["a", "b"]), tmp_path / "g0")
    assert res.scores == [-1.0, -1.0] and res.embeds == [None, None]


def test_payload_carries_numbers_only(monkeypatch, tmp_path) -> None:
    """The whole run: what lands at put_url is model + per-group scores/embeds, no ranking, no threshold."""
    import json

    import podagent.infer_cliprank as m
    from podagent.models import ClipRankParams

    monkeypatch.setattr(m, "download", lambda url, dest: (dest.parent.mkdir(parents=True, exist_ok=True),
                                                          dest.write_bytes(b"img"), dest)[-1])
    monkeypatch.setattr("PIL.Image.open", lambda p: types.SimpleNamespace(convert=lambda mode: "img"))
    put = tmp_path / "out" / "clip_rank.json"
    monkeypatch.setattr(m, "upload", lambda src, url, ct=None: (put.parent.mkdir(parents=True, exist_ok=True),
                                                               put.write_bytes(src.read_bytes())))

    params = ClipRankParams(groups=[ClipRankGroup(intent="chart", image_urls=["a", "b"]),
                                    ClipRankGroup(intent="", image_urls=["c", "d"])])
    infer_s = _svc().run(params, "https://storage.example/o/1.json?sig=PUT")

    assert infer_s >= 0
    body = json.loads(put.read_text())
    assert set(body) == {"model", "groups"}, "no ranking, no threshold, no rationale crosses back"
    assert [g["scores"] for g in body["groups"]] == [[0.6, 0.8], [-1.0, -1.0]]
    assert all(set(g) == {"scores", "embeds"} for g in body["groups"])


def test_service_is_registered_for_the_clip_rank_kind() -> None:
    from podagent import main

    assert "clip_rank" in main.INFER_KINDS


@pytest.mark.parametrize("intent", ["", "some intent"])
def test_forward_rounds_to_four_places(intent: str) -> None:
    """4dp is the wire convention (cosine error ~1e-3) — it must hold on both branches."""
    _, embeds = _svc()._forward(intent, ["img", "img"])
    assert all(len(str(x).split(".")[-1]) <= 4 for row in embeds for x in row)
