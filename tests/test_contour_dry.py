"""Contour-dry stand-in (podagent/ops/dry.py) and its pod-side boot lock (podagent.main)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from podagent import main as podagent_main
from podagent.ops import dry, registry

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


def test_armed_reads_only_its_own_env(monkeypatch):
    monkeypatch.delenv(dry.ARM_ENV, raising=False)
    assert dry.armed() is False
    monkeypatch.setenv(dry.ARM_ENV, "1")
    assert dry.armed() is True
    monkeypatch.setenv(dry.ARM_ENV, "0")
    assert dry.armed() is False


@pytest.mark.parametrize("op_name", sorted(p.stem for p in CONTRACTS.glob("ops/*.json")))
def test_every_declared_op_yields_its_declared_outputs_in_dry_mode(tmp_path, op_name):
    op = registry.get(op_name)
    fn = dry.resolve(op)
    outputs = {}
    for port in op.outputs:
        ext = {"video": ".mp4", "audio": ".m4a", "image": ".png", "json": ".json"}[port.kind]
        if port.many:
            outputs[port.id] = [tmp_path / f"{port.id}_{i}{ext}" for i in range(2)]
        else:
            outputs[port.id] = tmp_path / f"{port.id}{ext}"
    fn(params={}, inputs={}, outputs=outputs)
    for port in op.outputs:
        paths = outputs[port.id] if port.many else [outputs[port.id]]
        for path in paths:
            assert path.exists() and path.stat().st_size > 0, f"{op_name}.{port.id}: dry stub wrote nothing"


def test_json_output_is_readable_json(tmp_path):
    dst = tmp_path / "out.json"
    dry._write_json(dst, seed="probe")
    json.loads(dst.read_text())


def test_unknown_output_kind_refuses_by_name(tmp_path):
    with pytest.raises(registry.OpError):
        dry._fill_one(tmp_path / "x", "browser", seed="s")


def test_boot_refuses_dry_arm_off_the_local_contour(monkeypatch):
    monkeypatch.setenv(dry.ARM_ENV, "1")
    monkeypatch.setenv("POD_IMAGE_TAG", "a" * 40)
    with pytest.raises(SystemExit):
        podagent_main._refuse_dry_off_local_contour()


def test_boot_accepts_dry_arm_on_the_local_contour(monkeypatch):
    monkeypatch.setenv(dry.ARM_ENV, "1")
    monkeypatch.delenv("POD_IMAGE_TAG", raising=False)
    podagent_main._refuse_dry_off_local_contour()


def test_boot_is_a_noop_when_the_switch_is_off(monkeypatch):
    monkeypatch.delenv(dry.ARM_ENV, raising=False)
    monkeypatch.setenv("POD_IMAGE_TAG", "a" * 40)
    podagent_main._refuse_dry_off_local_contour()
