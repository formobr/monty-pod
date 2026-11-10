"""Every boot refusal frame names the worker the control plane knows this pod by (timings.worker), so the
box pool can find the billing pod and evict it instead of logging «refusal from worker (unnamed)»."""
from __future__ import annotations

import base64
import json
import subprocess

import pytest

from podagent import identity
from podagent import infer_cliprank
from podagent import main as agent_main

WORKER = "job-7f3a"


@pytest.fixture(autouse=True)
def _registered_worker(monkeypatch, tmp_path):
    body = base64.urlsafe_b64encode(json.dumps({"jid": WORKER}).encode()).decode().rstrip("=")
    monkeypatch.setenv("JOB_TOKEN", f"{body}.sig")
    monkeypatch.delenv("MONTY_JOB_TOKEN", raising=False)
    identity._reset()
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")
    monkeypatch.delenv("MONTY_INFER_KINDS", raising=False)
    yield
    identity._reset()


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def send_event(self, ev: dict, *, wait: bool = False) -> bool:
        self.events.append(ev)
        return True


def _assert_named(cp: _CP) -> None:
    assert len(cp.events) == 1
    ev = cp.events[0]
    assert (ev["stage"], ev["status"], ev["phase"]) == ("boot", "error", "work_finished")
    assert ev["timings"] == {"worker": WORKER}
    assert identity.worker_identity() == WORKER


def test_the_frame_helper_carries_the_registered_worker():
    ev = agent_main._boot_refusal_frame("REFUSING work: x")
    assert ev["step"] == "REFUSING work: x"
    assert ev["timings"]["worker"] == WORKER


def test_the_vram_refusal_names_the_worker(monkeypatch):
    monkeypatch.setattr(infer_cliprank, "_free_vram_mb", lambda: 19.0)
    monkeypatch.setattr(infer_cliprank, "vram_total_mb", lambda: 15852.0)
    monkeypatch.delenv("MONTY_OPS_CONTOUR_DRY", raising=False)
    cp = _CP()
    with pytest.raises(SystemExit):
        agent_main._free_vram_or_refuse(cp)
    _assert_named(cp)


def _failing_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout=b"", stderr=b"probe failed"))


def test_the_nvenc_refusal_names_the_worker(monkeypatch):
    _failing_ffmpeg(monkeypatch)
    cp = _CP()
    with pytest.raises(SystemExit):
        agent_main._nvenc_or_refuse(cp)
    _assert_named(cp)


def test_the_nvdec_refusal_names_the_worker(monkeypatch):
    _failing_ffmpeg(monkeypatch)
    cp = _CP()
    with pytest.raises(SystemExit):
        agent_main._nvdec_or_refuse(cp)
    _assert_named(cp)


def test_the_vulkan_refusal_names_the_worker(monkeypatch):
    monkeypatch.setattr(agent_main, "_run_vulkan_probe", lambda: (False, "no device", b""))
    monkeypatch.setattr(agent_main, "_log_vulkan_rank", lambda *a, **k: None)
    monkeypatch.setattr(agent_main, "_ldconfig_p", lambda: "")
    monkeypatch.setattr(agent_main, "_egl_icd_lib", lambda _out: None)
    monkeypatch.setattr(agent_main, "_boot_driver_version", lambda **_k: None)
    monkeypatch.setattr(agent_main, "_vulkan_host_evidence", lambda *_a: "evidence")
    monkeypatch.setattr(agent_main, "_vulkaninfo_summary", lambda: "")
    cp = _CP()
    with pytest.raises(SystemExit):
        agent_main._vulkan_preflight(cp)
    _assert_named(cp)


def test_the_served_kinds_refusal_names_the_worker(monkeypatch):
    monkeypatch.setenv("MONTY_INFER_KINDS", "no-such-kind")
    cp = _CP()
    with pytest.raises(SystemExit):
        agent_main._served_kinds_or_refuse(cp)
    _assert_named(cp)
