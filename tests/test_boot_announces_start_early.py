"""A booting pod's first frame must precede every unbounded GPU step, so a hang there is visible on the
control plane (two silent rentals were cancelled with no way to tell whether the process ever started)."""
import pytest

from podagent import main as agent_main

EARLY = "process started · pre-gpu"


class _Stop(Exception):
    pass


class _CP:
    def __init__(self, log):
        self.log = log
        self.frames = []

    def note(self, frame):
        self.frames.append(frame)
        self.log.append(("note", frame["step"]))


def _arrange(monkeypatch, gpu):
    log: list = []
    cp = _CP(log)

    def _served(_cp):
        log.append(("served", None))
        return frozenset()

    monkeypatch.setenv("CP_URL", "https://control-plane.example")
    monkeypatch.setenv("JOB_TOKEN", "opaque-test-token")
    monkeypatch.delenv("MONTY_OPS_CONTOUR_DRY", raising=False)
    monkeypatch.setattr(agent_main, "ControlPlane", lambda *_a, **_k: cp)
    monkeypatch.setattr(agent_main.signal, "signal", lambda *_a, **_k: None)
    monkeypatch.setattr(agent_main, "_served_kinds_or_refuse", _served)

    def _gpu():
        log.append(("gpu", None))
        gpu()

    monkeypatch.setattr(agent_main, "_log_gpu_status", _gpu)
    return cp, log


def _boom():
    raise _Stop


def test_the_early_frame_precedes_served_kinds_and_every_gpu_step(monkeypatch):
    cp, log = _arrange(monkeypatch, _boom)
    with pytest.raises(_Stop):
        agent_main.main()
    kinds = [k for k, _ in log]
    first = log[0]
    assert first[0] == "note" and first[1].startswith(EARLY)
    assert kinds.index("note") < kinds.index("served") < kinds.index("gpu")
    # the second step is queued right before the GPU work starts
    assert log[kinds.index("gpu") - 1] == ("note", "gpu init started")


def test_the_early_frame_survives_a_gpu_step_that_fails(monkeypatch):
    def _raise():
        raise RuntimeError("driver wedged")

    cp, _log = _arrange(monkeypatch, _raise)
    with pytest.raises(RuntimeError):
        agent_main.main()
    assert any(f["step"].startswith(EARLY) for f in cp.frames)


def test_the_early_frame_has_boot_shape_and_no_gpu_fields(monkeypatch):
    cp, _log = _arrange(monkeypatch, _boom)
    with pytest.raises(_Stop):
        agent_main.main()
    frame = cp.frames[0]
    assert {k: frame[k] for k in ("stage", "status", "phase")} == {"stage": "boot", "status": "step",
                                                                  "phase": "started"}
    assert "py=" in frame["step"] and "pid=" in frame["step"]
    assert "gpu=" not in frame["step"]
    assert set(frame) == {"stage", "status", "phase", "step"}
