"""A pod whose ffmpeg cannot decode via cuda hwaccel silently falls back to software (0.73x-realtime
incident, docs/research/nvdec-normalize-config.md) even though its NVENC probe passed — NVENC and NVDEC are
separate silicon. This probe forces the failure loud at boot, before the pod claims anything."""
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


def _fake_run(*, encode_rc: int = 0, encode_err: bytes = b"", decode_rc: int = 0,
             decode_err: bytes = b"") -> tuple[list[list[str]], object]:
    """Route by argv content: the encode leg asks for `hevc_nvenc`, the decode leg for `-hwaccel`."""
    seen: list[list[str]] = []

    def fake(cmd, *a, **k):
        seen.append(list(cmd))
        if "hevc_nvenc" in cmd:
            return _run(encode_rc, encode_err)
        if "-hwaccel" in cmd:
            return _run(decode_rc, decode_err)
        return _run(0)

    return seen, fake


def test_a_working_decoder_lets_the_pod_go_on(monkeypatch):
    cp = _CP()
    _seen, fake = _fake_run()
    monkeypatch.setattr(subprocess, "run", fake)
    agent_main._nvdec_or_refuse(cp)          # must NOT raise
    assert not [e for e in cp.events if e.get("status") == "error"]


def test_a_dead_decoder_refuses_the_pod_before_it_claims_anything(monkeypatch):
    """NEGATIVE: NVENC alone passing must not be read as NVDEC proof."""
    err = b"[hevc @ 0x1] Impossible to convert between the formats supported by the filter 'auto_scale_0'\n"
    cp = _CP()
    _seen, fake = _fake_run(decode_rc=187, decode_err=err)
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(SystemExit) as e:
        agent_main._nvdec_or_refuse(cp)
    assert e.value.code != 0, "a pod that cannot decode via cuda hwaccel must exit non-zero"


def test_the_refusal_reaches_the_box_and_names_the_reason(monkeypatch):
    err = b"Impossible to convert between the formats supported by the filter 'auto_scale_0'"
    cp = _CP()
    _seen, fake = _fake_run(decode_rc=187, decode_err=err)
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(SystemExit):
        agent_main._nvdec_or_refuse(cp)
    errs = [e for e in cp.events if e.get("status") == "error"]
    assert errs, "the pod died without telling the control plane why"
    assert "NVDEC" in errs[0]["step"] and "Impossible to convert" in errs[0]["step"]
    assert cp.waits == [True], "a refusal the process exits behind must wait for its typed ACK"


def test_an_ffmpeg_that_will_not_even_decode_is_a_refusal_too(monkeypatch):
    """OSError on the decode leg, not a non-zero exit: a missing cuda hwaccel device must not read as a pass."""
    cp = _CP()

    def _boom(cmd, *a, **k):
        if "hevc_nvenc" in cmd:
            return _run(0)
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(SystemExit):
        agent_main._nvdec_or_refuse(cp)


def test_a_dead_encoder_leg_also_refuses_before_ever_trying_to_decode(monkeypatch):
    """The probe encode must fail loudly too — a probe clip that never exists cannot prove NVDEC either way."""
    cp = _CP()
    seen, fake = _fake_run(encode_rc=218, encode_err=b"Driver does not support the required nvenc API version")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(SystemExit):
        agent_main._nvdec_or_refuse(cp)
    assert len(seen) == 1, "a failed probe-clip encode must never reach the decode leg"


def test_the_probe_exercises_the_hwdownload_filter_we_rely_on_for_a_loud_failure(monkeypatch):
    """Without `hwdownload` after `-hwaccel_output_format cuda`, a missing NVDEC decodes in software with
    rc 0 (docs/research/nvdec-normalize-config.md) and this whole probe would silently pass everywhere."""
    seen, fake = _fake_run()
    monkeypatch.setattr(subprocess, "run", fake)
    agent_main._nvdec_or_refuse(_CP())
    decode_cmd = next(c for c in seen if "-hwaccel" in c)
    assert "-hwaccel_output_format" in decode_cmd and "cuda" in decode_cmd
    assert "hwdownload" in " ".join(decode_cmd)


def test_capability_refusal_clears_the_liveness_mark(monkeypatch, tmp_path):
    mark = tmp_path / "podagent.alive"
    mark.write_text("this incarnation")
    monkeypatch.setattr(agent_main, "_LIVE_MARK", mark)
    _seen, fake = _fake_run(decode_rc=187, decode_err=b"driver mismatch")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(SystemExit):
        agent_main._nvdec_or_refuse(_CP())
    assert not mark.exists(), "a deliberate capability refusal would be reported as a false UNCLEAN death"


def test_nvdec_probe_runs_after_nvenc_in_capability_preflight(monkeypatch):
    timeline: list[str] = []
    cp = _CP()
    monkeypatch.setattr(agent_main, "_report_boot", lambda _cp: None)
    monkeypatch.setattr(agent_main, "_nvenc_or_refuse", lambda _cp: timeline.append("nvenc"))
    monkeypatch.setattr(agent_main, "_nvdec_or_refuse", lambda _cp: timeline.append("nvdec"))
    monkeypatch.setattr(agent_main, "_vulkan_preflight", lambda _cp: timeline.append("vulkan"))
    agent_main._capability_preflight(cp)
    assert timeline == ["nvenc", "nvdec", "vulkan"]
