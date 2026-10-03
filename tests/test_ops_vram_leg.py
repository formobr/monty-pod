"""A HEAVY OP BOOKS WHAT IT COST ON THE CARD; A LIGHT OP AND AN UNREADABLE CARD BOOK NOTHING (VRAM_LEG_WHY).

Hermetic: the real `run_chain`, with the pack, the registry, the transport and the nvidia-smi reader stubbed.
Each test is NEGATIVE in the docs/TESTING.md sense — revert the watch and the first one fails; drop the
reader's guard and the last one fails the op it was only meant to measure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from podagent.ops import gpu_admission, registry, runner  # noqa: E402

_VRAM_KEYS = {"vram_free_before_mib", "vram_free_after_mib", "vram_peak_used_mib"}


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def send_event(self, payload: dict, *, wait: bool = False) -> bool:
        self.events.append(payload)
        return True

    def send_result(self, payload: dict, *, wait: bool = True) -> bool:
        return True

    def timeline_context(self, corr_id: str) -> dict:
        return {"complete": True, "incomplete_reasons": [], "pod_clock_id": "pod-clock-1",
                "attempt_id": "attempt-1", "clock_sync": [],
                "delivery": {"delivery_id": corr_id or "local", "attempt_id": "attempt-1"}}

    @property
    def terminal(self) -> dict:
        return self.events[-1]


class _Step:
    def __init__(self, sid: str, op, src: Path, out_port: str) -> None:
        self.id, self.op, self.params, self.needs, self.optional = sid, op.op, {}, [], False
        self.inputs = [type("B", (), {"port": p.id, "url": None, "from_step": None, "path": str(src)})()
                       for p in op.inputs]
        self.outputs = [type("B", (), {"port": out_port, "url": None, "urls": None})()]


class _Chain:
    def __init__(self, steps: list) -> None:
        self.steps, self.job_id, self.pack = steps, "j-1", None


@pytest.fixture
def op():
    """A real shipped op with one required output and a non-transport budget, so it CAN be made heavy."""
    o = next((x for x in registry.all_ops().values()
              if x.op not in gpu_admission.HEAVY_GPU_OPS and x.budget != "transport"
              and sum(1 for p in x.outputs if not p.optional) == 1
              and any(not p.many for p in x.outputs)), None)
    if o is None:
        pytest.fail("registry holds no light non-transport op with one required output — widen this fixture")
    return o


@pytest.fixture(autouse=True)
def _admission_clean():
    gpu_admission._reset_for_tests()
    yield
    gpu_admission._reset_for_tests()


def _run(monkeypatch, tmp_path, op, *, heavy: bool) -> dict:
    src = tmp_path / "in.bin"
    src.write_bytes(b"x" * 16)
    required = next(p.id for p in op.outputs if not p.optional and not p.many)

    def _fn(*, params, inputs, outputs):
        outputs[required].write_bytes(b"y" * 64)

    if heavy:
        monkeypatch.setattr(runner.gpu_admission, "HEAVY_GPU_OPS",
                            gpu_admission.HEAVY_GPU_OPS | {op.op})
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.registry, "assert_pod_safe", lambda *a, **k: None)
    monkeypatch.setattr(runner, "preflight_chain", lambda chain: None)
    monkeypatch.setattr(runner.pack, "activate_or_mismatch", lambda ref: tmp_path)
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _fn)
    monkeypatch.setattr(runner, "log", lambda *a, **k: None)
    cp = _CP()
    runner.run_chain(_Chain([_Step("s1", op, src, required)]), cp)
    assert cp.terminal["status"] == "ok", cp.terminal
    return cp.terminal["timings"]["steps"][0]


def _reader(values: list[float]):
    """free MiB in order: before, any samples, after (the last value repeats)."""
    it = iter(values)
    last = [values[-1]]

    def _read():
        last[0] = next(it, last[0])
        return last[0]
    return _read


def test_a_heavy_op_reports_its_vram_legs(monkeypatch, tmp_path, op):
    """NEGATIVE: drop the watch from the heavy branch and the M4 budget has nothing to read."""
    monkeypatch.setattr(runner, "_read_vram_free_mib", _reader([20000.0, 14000.0]))
    step = _run(monkeypatch, tmp_path, op, heavy=True)
    legs = step["legs"]
    assert _VRAM_KEYS <= set(legs), legs
    assert legs["vram_free_before_mib"] == 20000.0
    assert legs["vram_free_after_mib"] == 14000.0
    assert legs["vram_peak_used_mib"] == 6000.0
    assert {"slot_wait", "bind", "run", "put"} <= set(legs), "the runner's own legs are untouched"


def test_a_light_op_reports_no_vram_legs(monkeypatch, tmp_path, op):
    """A light op never takes GPU admission, so it never reads the card."""
    calls: list[int] = []
    monkeypatch.setattr(runner, "_read_vram_free_mib", lambda: calls.append(1) or 20000.0)
    step = _run(monkeypatch, tmp_path, op, heavy=False)
    assert not (_VRAM_KEYS & set(step["legs"])), step["legs"]
    assert not calls, "a light op read the card"


def test_an_unreadable_card_books_no_vram_legs_and_the_op_succeeds(monkeypatch, tmp_path, op):
    """NEGATIVE: let the reader's raise escape the sampler and the measurement fails the op it measures."""
    def _boom():
        raise RuntimeError("nvidia-smi: no devices")
    monkeypatch.setattr(runner, "_read_vram_free_mib", _boom)
    step = _run(monkeypatch, tmp_path, op, heavy=True)
    assert not (_VRAM_KEYS & set(step["legs"])), step["legs"]
    assert step["outputs"], "the op ran as before"


def test_a_card_that_reports_nothing_books_no_vram_legs(monkeypatch, tmp_path, op):
    """`_free_vram_mb` answers None (never a coerced 0) on a box without a readable card."""
    monkeypatch.setattr(runner, "_read_vram_free_mib", lambda: None)
    step = _run(monkeypatch, tmp_path, op, heavy=True)
    assert not (_VRAM_KEYS & set(step["legs"])), step["legs"]
