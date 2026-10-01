"""A rented card whose VRAM is already held by processes that are not ours is refused at boot, before ready
(main.BOOT_FREE_VRAM_WHY) — server 96828 OOM'd «15.48 GiB total, 19 MiB free» only after reporting ready."""
from __future__ import annotations

import pytest

from podagent import infer_cliprank
from podagent import main as agent_main
from podagent.infer_lanes import KIND_VRAM_MIB, RESERVE_MIB


@pytest.fixture(autouse=True)
def _isolated_live_mark(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def send_event(self, ev: dict, *, wait: bool = False) -> bool:
        self.events.append(ev)
        return True


def _stub_card(monkeypatch, *, free: float | None, total: float | None) -> None:
    monkeypatch.setattr(infer_cliprank, "_free_vram_mb", lambda: free)
    monkeypatch.setattr(infer_cliprank, "vram_total_mb", lambda: total)


def _stub_other_probes(monkeypatch, timeline: list[str]) -> None:
    monkeypatch.setattr(agent_main, "_report_boot", lambda _cp: timeline.append("boot"))
    monkeypatch.setattr(agent_main, "_nvenc_or_refuse", lambda _cp: timeline.append("nvenc"))
    monkeypatch.setattr(agent_main, "_nvdec_or_refuse", lambda _cp: timeline.append("nvdec"))
    monkeypatch.setattr(agent_main, "_vulkan_preflight", lambda _cp: timeline.append("vulkan"))
    monkeypatch.setattr(agent_main, "_report_ready", lambda _cp, **_k: timeline.append("ready"))


def test_the_floor_is_the_heaviest_kind_plus_reserve():
    assert agent_main.boot_vram_floor_mib() == max(KIND_VRAM_MIB.values()) + RESERVE_MIB == 3248.0


def test_boot_refuses_when_foreign_processes_hold_the_vram(monkeypatch):
    timeline: list[str] = []
    cp = _CP()
    _stub_other_probes(monkeypatch, timeline)
    _stub_card(monkeypatch, free=19.0, total=15.48 * 1024)
    agent_main._mark_alive()
    with pytest.raises(SystemExit) as exc:
        agent_main._capability_preflight(cp)
    assert exc.value.code == agent_main.BOOT_VRAM_REFUSAL_EXIT
    assert exc.value.code not in (2, 3, 4, 5)
    assert "ready" not in timeline and "nvdec" not in timeline
    assert len(cp.events) == 1
    ev = cp.events[0]
    assert (ev["stage"], ev["status"], ev["phase"]) == ("boot", "error", "work_finished")
    assert ev["step"].startswith("gpu_vram_occupied: free=19 total=15852 floor=3248")
    assert not agent_main._LIVE_MARK.exists(), "a deliberate refusal must read as a STOP, not a death"


def test_a_free_card_proceeds_to_ready(monkeypatch):
    timeline: list[str] = []
    cp = _CP()
    _stub_other_probes(monkeypatch, timeline)
    _stub_card(monkeypatch, free=15000.0, total=16311.0)
    agent_main._capability_preflight(cp)
    assert timeline == ["boot", "nvenc", "nvdec", "vulkan", "ready"]
    assert cp.events == []


def test_an_unreadable_card_is_not_refused(monkeypatch):
    timeline: list[str] = []
    cp = _CP()
    _stub_other_probes(monkeypatch, timeline)
    _stub_card(monkeypatch, free=None, total=None)
    agent_main._capability_preflight(cp)
    assert timeline[-1] == "ready"
    assert cp.events == []
