"""A pod whose ffmpeg cannot open h264_nvenc is dead, not slow — and it used to prove that 157 s into
ingest, on a run that had already paid for b-roll. The encoder is established at boot, where the pod is."""
from __future__ import annotations

import subprocess

import pytest

from podagent import main as agent_main


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def note(self, ev: dict) -> None:
        self.events.append(ev)


def _run(returncode: int = 0, stderr: bytes = b"") -> object:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=b"", stderr=stderr)


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
