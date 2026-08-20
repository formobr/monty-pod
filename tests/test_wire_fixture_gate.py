from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from podagent.wire_generated import WIRE_CONTRACT_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _gate():
    path = ROOT / "tools" / "check_wire_literals.py"
    spec = importlib.util.spec_from_file_location("_pod_wire_literal_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_pod_builds_shared_wire_from_generated_models_or_canonical_goldens() -> None:
    gate = _gate()
    found = gate.problems(gate.live_paths())
    assert found == [], "\n".join(problem.render() for problem in found)


def test_a_hand_built_valid_stream_frame_is_rejected(tmp_path: Path) -> None:
    gate = _gate()
    path = tmp_path / "test_bad.py"
    path.write_text(
        "frame = {'type':'ack','stream_id':'s','seq':1,'status':202,"
        "'server_recv_unix_ns':1,'server_send_unix_ns':2}\n",
        encoding="utf-8",
    )
    found = gate.problems([path])
    assert len(found) == 1 and "hand-built valid pod_stream_server" in found[0].message


def test_canonical_helper_and_local_spec_v5_are_outside_the_refusal(tmp_path: Path) -> None:
    gate = _gate()
    path = tmp_path / "test_good.py"
    path.write_text(
        "frame = valid_wire('pod_stream_server.ack', {'seq': 2})\n"
        "spec = {'spec_version': 5, 'job_id': 'j'}\n",
        encoding="utf-8",
    )
    assert gate.problems([path]) == []


def test_a_hand_owned_wire_version_literal_is_rejected(tmp_path: Path) -> None:
    gate = _gate()
    path = tmp_path / "bad_version.py"
    path.write_text(f"WIRE_VERSION = {WIRE_CONTRACT_VERSION}\n", encoding="utf-8")
    found = gate.problems([path])
    assert len(found) == 1 and "numeric shared-wire version" in found[0].message
