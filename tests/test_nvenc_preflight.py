"""A pod whose ffmpeg cannot open h264_nvenc is dead, not slow — and it used to prove that 157 s into
ingest, on a run that had already paid for b-roll. The encoder is established at boot, where the pod is."""
from __future__ import annotations

import subprocess

import pytest

from podagent import main as agent_main


@pytest.fixture(autouse=True)
def _isolated_live_mark(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")


class _CP:
    def __init__(self, *, accept: bool = True) -> None:
        self.events: list[dict] = []
        self.waits: list[bool] = []
        self.accept = accept

    def note(self, ev: dict) -> None:
        self.events.append(ev)

    def send_event(self, ev: dict, *, wait: bool = False) -> bool:
        self.events.append(ev)
        self.waits.append(wait)
        return self.accept


def _run(returncode: int = 0, stderr: bytes = b"") -> object:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=b"", stderr=stderr)


def test_a_vulkan_probe_warns_and_returns_false_when_the_camera_path_is_unavailable(monkeypatch):
    cp = _CP()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(1, b"first detail\nVK_ERROR_INCOMPATIBLE_DRIVER\n"))
    assert agent_main._vulkan_preflight(cp) is False
    warnings = [e for e in cp.events if e.get("status") == "step"]
    assert warnings and warnings[0]["step"].startswith("WARNING VULKAN UNAVAILABLE: exit 1:")
    assert "VK_ERROR_INCOMPATIBLE_DRIVER" in warnings[0]["step"]
    assert cp.waits == [True]


def test_vulkan_probe_warning_keeps_head_and_tail_and_vulkaninfo_summary(monkeypatch):
    """Every rank (default -> ldconfig lookup -> nvidia-smi) fails the same fake way, so the ladder falls all
    the way through without finding a libEGL_nvidia.so.0 to synthesize a manifest from — the SAME forensics
    exhausted-ladder path the deny-list eviction reason downstream depends on."""
    cp = _CP()
    root = b"VK_ERROR_INCOMPATIBLE_DRIVER\n"
    stderr = root + (b"middle-noise\n" * 200) + b"generic ffmpeg tail\n"
    seen: list[tuple[list[str], int]] = []

    def fake_run(cmd, *args, **kwargs):
        seen.append((list(cmd), kwargs["timeout"]))
        if str(cmd[0]).endswith("vulkaninfo"):
            return subprocess.CompletedProcess(cmd, 0, stdout=b"GPU0: NVIDIA summary\n", stderr=b"")
        return _run(1, stderr)

    monkeypatch.setattr(agent_main.shutil, "which", lambda name: "/usr/bin/vulkaninfo" if name == "vulkaninfo" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert agent_main._vulkan_preflight(cp) is False
    step = cp.events[0]["step"]
    assert "VK_ERROR_INCOMPATIBLE_DRIVER" in step
    assert "generic ffmpeg tail" in step
    assert "vulkaninfo: GPU0: NVIDIA summary" in step
    assert "ranks_tried=default" in step
    assert "driver_version=unknown" in step
    assert len(step) < 1800
    assert seen[0] == (list(agent_main.VULKAN_PROBE), 120), "rank 1 is the plain VULKAN_PROBE, timeout 120"
    assert seen[1][0] == ["ldconfig", "-p"], "a failed default rank must look for libEGL_nvidia before giving up"
    assert seen[-1][0] == ["/usr/bin/vulkaninfo", "--summary"], "the diagnostic summary is still appended last"
    assert cp.waits == [True]


def test_capacity_publishes_the_vulkan_capability_fact():
    payload = agent_main.capacity_payload(rank_lanes=1, fetch_workers=1, vulkan=False)
    assert payload["vulkan"] is False


def test_a_working_encoder_lets_the_pod_go_on(monkeypatch):
    cp = _CP()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(0))
    agent_main._nvenc_or_refuse(cp)          # must NOT raise
    assert not [e for e in cp.events if e.get("status") == "error"]


def test_a_dead_encoder_refuses_the_pod_before_it_claims_anything(monkeypatch):
    """NEGATIVE: this is the exact 2026-08-02 image — ffmpeg built against nvenc SDK 13.1 on a driver that
    offers 13.0. Let it through and the pod claims work it cannot finish."""
    err = (b"[h264_nvenc @ 0x1] Driver does not support the required nvenc API version. "
           b"Required: 13.1 Found: 13.0\n")
    cp = _CP()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(218, err))
    with pytest.raises(SystemExit) as e:
        agent_main._nvenc_or_refuse(cp)
    assert e.value.code != 0, "a pod that cannot encode must exit non-zero, not fall through to the claim loop"


def test_the_refusal_reaches_the_box_and_names_the_driver(monkeypatch):
    """A keyless pod has no other voice: refusing mutely is indistinguishable from a dead host, which is the
    silence this whole file exists to prevent."""
    err = b"[h264_nvenc @ 0x1] Driver does not support the required nvenc API version. Required: 13.1 Found: 13.0"
    cp = _CP()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(218, err))
    with pytest.raises(SystemExit):
        agent_main._nvenc_or_refuse(cp)
    errs = [e for e in cp.events if e.get("status") == "error"]
    assert errs, "the pod died without telling the control plane why"
    assert "nvenc API version" in errs[0]["step"], f"the refusal does not carry ffmpeg's reason: {errs[0]}"
    assert cp.waits == [True], "a refusal the process exits behind must wait for its typed ACK"


def test_refusal_preserves_root_stderr_and_tail_without_unbounded_dump(monkeypatch):
    root = b"Driver does not support the required nvenc API version\n"
    err = root + (b"middle-noise\n" * 200) + b"final driver detail"
    cp = _CP()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(218, err))
    with pytest.raises(SystemExit):
        agent_main._nvenc_or_refuse(cp)
    step = cp.events[0]["step"]
    assert root.decode().strip() in step
    assert "final driver detail" in step
    assert len(step) < 1200, "the bounded refusal accidentally forwarded the whole ffmpeg dump"


def test_nvidia_runtime_diagnostic_names_the_injected_driver(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(agent_main, "_log", lines.append)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        args=[], returncode=0, stdout=b"NVIDIA GeForce RTX 4090, 580.76.05\n", stderr=b"",
    ))
    agent_main._log_nvidia_runtime()
    assert lines == ["nvidia-smi OK · NVIDIA GeForce RTX 4090, 580.76.05"]


def test_nvidia_runtime_diagnostic_is_loud_but_not_a_gate(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(agent_main, "_log", lines.append)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        FileNotFoundError("nvidia-smi missing")))
    assert agent_main._log_nvidia_runtime() is None
    assert len(lines) == 1 and lines[0].startswith("WARNING nvidia-smi unavailable:")


def test_capability_refusal_clears_the_liveness_mark(monkeypatch, tmp_path):
    mark = tmp_path / "podagent.alive"
    mark.write_text("this incarnation")
    monkeypatch.setattr(agent_main, "_LIVE_MARK", mark)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(218, b"driver mismatch"))
    with pytest.raises(SystemExit):
        agent_main._nvenc_or_refuse(_CP())
    assert not mark.exists(), "a deliberate capability refusal would be reported as a false UNCLEAN death"


def test_an_ffmpeg_that_will_not_even_launch_is_a_refusal_too(monkeypatch):
    """OSError, not a non-zero exit: an image missing the binary outright must not read as a pass."""
    cp = _CP()

    def _boom(*a, **k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(SystemExit):
        agent_main._nvenc_or_refuse(cp)


def test_the_probe_actually_asks_for_the_encoder_we_ship_with(monkeypatch):
    """Without this, the probe could drift to libx264 and pass on a pod with no working NVENC at all."""
    seen: list[list[str]] = []

    def _capture(cmd, *a, **k):
        seen.append(list(cmd))
        return _run(0)

    monkeypatch.setattr(subprocess, "run", _capture)
    agent_main._nvenc_or_refuse(_CP())
    assert seen and "h264_nvenc" in seen[0], f"the boot probe does not exercise h264_nvenc: {seen}"


def test_ready_is_sent_synchronously_only_after_a_successful_probe(monkeypatch):
    timeline: list[str] = []

    class _OrderedCP(_CP):
        def send_event(self, ev: dict, *, wait: bool = False) -> bool:
            timeline.append(str(ev.get("phase")))
            return super().send_event(ev, wait=wait)

    cp = _OrderedCP()
    monkeypatch.setattr(agent_main, "_report_boot", lambda _cp: timeline.append("boot"))
    monkeypatch.setattr(agent_main, "_nvenc_or_refuse", lambda _cp: timeline.append("probe"))
    monkeypatch.setattr(agent_main, "_vulkan_preflight", lambda _cp: timeline.append("vulkan"))
    agent_main._capability_preflight(cp)
    assert timeline == ["boot", "probe", "vulkan", "ready"]
    assert cp.events == [{
        "stage": "boot", "status": "step", "phase": "ready",
        "step": "capability preflight passed",
    }]
    assert cp.waits == [True]


def test_a_failed_probe_never_emits_ready(monkeypatch):
    cp = _CP()
    monkeypatch.setattr(agent_main, "_report_boot", lambda _cp: None)
    monkeypatch.setattr(agent_main, "_vulkan_preflight", lambda _cp: pytest.fail("vulkan probe reached"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(218, b"driver mismatch"))
    with pytest.raises(SystemExit):
        agent_main._capability_preflight(cp)
    assert not [e for e in cp.events if e.get("phase") == "ready"]


def test_an_ambiguous_ready_ack_stops_before_capacity_or_dispatch():
    cp = _CP(accept=False)
    with pytest.raises(agent_main.DeliveryPending, match="readiness ACK remains ambiguous"):
        agent_main._report_ready(cp)
    assert cp.events[-1]["phase"] == "ready"


def test_main_never_reaches_dispatch_when_ready_ack_is_ambiguous(monkeypatch):
    cp = _CP(accept=False)
    dispatched = False

    def _dispatch(*_a, **_k):
        nonlocal dispatched
        dispatched = True

    monkeypatch.setenv("CP_URL", "https://control-plane.example")
    monkeypatch.setenv("JOB_TOKEN", "opaque-test-token")
    monkeypatch.setattr(agent_main, "ControlPlane", lambda *_a, **_k: cp)
    monkeypatch.setattr(agent_main.signal, "signal", lambda *_a, **_k: None)
    monkeypatch.setattr(agent_main, "_log_gpu_status", lambda: None)
    monkeypatch.setattr(agent_main, "_report_boot", lambda _cp: None)
    monkeypatch.setattr(agent_main, "_nvenc_or_refuse", lambda _cp: None)
    monkeypatch.setattr(agent_main, "_vulkan_preflight", lambda _cp, **_k: None)
    monkeypatch.setattr(agent_main, "_dispatch_loop", _dispatch)

    with pytest.raises(agent_main.DeliveryPending, match="readiness ACK remains ambiguous"):
        agent_main.main()
    assert not dispatched, "the pod claimed work despite an unacknowledged readiness verdict"
    assert not [e for e in cp.events if e.get("phase") == "capacity"]
