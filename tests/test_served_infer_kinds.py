"""MONTY_INFER_KINDS bounds what a pod serves and what VRAM it demands at boot (infer_lanes.SERVED_INFER_KINDS_WHY):
a cassette-replayed release smoke sends no align / clip_rank, so the pod must not need the card for them."""
from __future__ import annotations

from pathlib import Path

import pytest

from podagent import infer_cliprank
from podagent import infer_lanes as lanes
from podagent import main as agent_main

_W = {"url": "https://x/w.tar", "sha256": "b" * 64}
_RANK_REQ = {"infer_version": 6, "job_id": "j", "kind": "clip_rank", "model": "siglip",
             "put_url": "https://x/o/r.json", "weights": _W,
             "clip_rank": {"groups": [{"intent": "chart", "image_urls": ["u1"]}]}}


@pytest.fixture(autouse=True)
def _isolated_live_mark(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")


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


def _boot(monkeypatch, *, env: str | None, free: float) -> tuple[_CP, list[str]]:
    timeline: list[str] = []
    for name in ("_report_boot", "_nvenc_or_refuse", "_nvdec_or_refuse", "_vulkan_preflight"):
        monkeypatch.setattr(agent_main, name, lambda _cp, _n=name: timeline.append(_n))
    monkeypatch.setattr(agent_main, "_report_ready", lambda _cp, **_k: timeline.append("ready"))
    monkeypatch.setattr(infer_cliprank, "_free_vram_mb", lambda: free)
    monkeypatch.setattr(infer_cliprank, "vram_total_mb", lambda: 6144.0)
    if env is None:
        monkeypatch.delenv(lanes.SERVED_INFER_KINDS_ENV, raising=False)
    else:
        monkeypatch.setenv(lanes.SERVED_INFER_KINDS_ENV, env)
    cp = _CP()
    agent_main._capability_preflight(cp)
    return cp, timeline


def test_a_pod_serves_and_sizes_only_its_given_infer_kinds(monkeypatch, tmp_path):
    # the floor follows the served set
    assert agent_main.boot_vram_floor_mib({"clip_rank", "align"}) == 3248.0
    assert agent_main.boot_vram_floor_mib({"align"}) == 2012.0
    assert agent_main.boot_vram_floor_mib({"face_probe"}) is None
    assert agent_main.boot_vram_floor_mib(set()) is None

    # unset env = every kind, today's floor
    assert lanes.served_kinds(None) == frozenset({"align", "clip_rank", "face_probe"})
    assert agent_main.boot_vram_floor_mib() == 3248.0
    with pytest.raises(SystemExit) as exc:
        _boot(monkeypatch, env=None, free=3236.0)          # the 2026-10-02 laptop, all kinds
    assert exc.value.code == agent_main.BOOT_VRAM_REFUSAL_EXIT

    # the same laptop boots when it is given only align
    _, timeline = _boot(monkeypatch, env="align", free=3236.0)
    assert timeline[-1] == "ready"

    # no GPU kind served: a card with 100 MiB free boots
    for env in ("face_probe", ""):
        cp, timeline = _boot(monkeypatch, env=env, free=100.0)
        assert timeline[-1] == "ready" and cp.events == []

    # an unknown kind refuses boot by name, before ready
    with pytest.raises(SystemExit) as exc:
        _boot(monkeypatch, env="align,clip_rnak", free=15000.0)
    assert exc.value.code == agent_main.BOOT_VRAM_REFUSAL_EXIT
    # a clip_rank request on an align-only pod is refused by name, nothing fetched or loaded
    loads: list[str] = []
    monkeypatch.setattr("podagent.weights.ensure", lambda *a, **k: loads.append("fetch") or tmp_path)
    monkeypatch.setattr("podagent.infer_cliprank.ClipRankService",
                        lambda *a, **k: loads.append("load"))
    cp = _CP()
    agent_main._run_infer(dict(_RANK_REQ), cp, {}, {}, {}, Path("/opt/models/yunet.onnx"), True,
                          corr_id="c", session_id="s", served=frozenset({"align"}))
    assert loads == []
    assert len(cp.results) == 1
    res = cp.results[0]
    assert res["status"] == "error" and res["kind"] == "clip_rank"
    assert "clip_rank" in res["error"] and "MONTY_INFER_KINDS=align" in res["error"]


def test_an_unknown_kind_names_itself_in_the_boot_refusal(monkeypatch):
    monkeypatch.setenv(lanes.SERVED_INFER_KINDS_ENV, "align,clip_rnak")
    cp = _CP()
    with pytest.raises(SystemExit):
        agent_main._served_kinds_or_refuse(cp)
    assert len(cp.events) == 1
    step = cp.events[0]["step"]
    assert "clip_rnak" in step and "MONTY_INFER_KINDS" in step
    assert not agent_main._LIVE_MARK.exists()
