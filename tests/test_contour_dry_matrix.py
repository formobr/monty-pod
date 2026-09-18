"""pod-agent/tests/test_contour_dry_matrix.py — MISC-62: the pod-side half of the dry stub x engine reader
matrix; mirrors tests/test_dry_stub_reader_matrix.py with an independent params roster (lock 4). Pod-only:
no engine-module import, no engine-source read — those value-semantics checks live in the engine's file."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from podagent.ops import dry, pack, registry

REPO = Path(__file__).resolve().parents[1]
# monty-pod ships no media fixture; tests that bind one skip via `_NEEDS_FIXTURE` rather than assume it.
FIXTURE = REPO / "dev/localpod/fixtures/contour-smoke.mp4"

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_FFPROBE = shutil.which("ffprobe") is not None
_NEEDS_FFMPEG = pytest.mark.skipif(
    not (_HAS_FFMPEG and _HAS_FFPROBE), reason="ffmpeg/ffprobe not on this runner")

_ALL_OPS = sorted(p.stem for p in (REPO / "contracts/ops").glob("*.json"))
_STUB_OPS = sorted(n for n in _ALL_OPS if dry._CLASSIFICATION[n][0] == dry.STUB)
_REAL_OPS = sorted(n for n in _ALL_OPS if dry._CLASSIFICATION[n][0] == dry.REAL)

# Deliberately spelled again rather than imported from the engine-side file (lock 4): the two rosters must
# independently agree the params are contract-valid, so a drift between them reds ONE of the two matrices.
_JSON_PARAMS = {
    "cut.apply": {"keep": [[0.0, 2.0], [3.5, 5.75], [9.0, 9.5]], "fps_grid": 30.0},
    "media.sheet": {"cols": 3, "cell_w": 64, "cell_h": 64, "gap": 4, "head": 0, "caption_h": 0,
                     "bg": [0, 0, 0], "captions": [[{"text": "a"}], [{"text": "b"}], [{"text": "c"}]]},
    "media.image_tile": {"urls": ["https://example.com/a.jpg", "https://example.com/b.jpg",
                                   "https://example.com/c.jpg"],
                          "width": 48, "height": 48, "fit": "cover"},
}
_EXTRA_PARAMS = {"media.cut_proxy": {"max_h": 360}}
_UNDERIVABLE = {"media.still"}


def _produce(tmp_path: Path, op_name: str, *, only: set[str] | None = None):
    op = registry.get(op_name)
    fn = dry.resolve(op)
    outputs: dict[str, object] = {}
    for port in op.outputs:
        if only is not None and port.id not in only:
            continue
        ext = {"video": ".mp4", "audio": ".m4a", "image": ".png", "json": ".json"}[port.kind]
        name = f"{op_name.replace('.', '_')}_{port.id}"
        outputs[port.id] = ([tmp_path / f"{name}_{i}{ext}" for i in range(2)] if port.many
                             else tmp_path / f"{name}{ext}")
    inputs = {p.id: FIXTURE for p in op.inputs if p.kind == "video"}
    params = {**_JSON_PARAMS.get(op_name, {}), **_EXTRA_PARAMS.get(op_name, {})}
    try:
        fn(params=params, inputs=inputs, outputs=outputs)
    except dry.DryStubUnderivedField:
        if op_name in _UNDERIVABLE:
            return None
        raise
    return outputs


def _ffprobe_opens(path: Path) -> str | None:
    proc = subprocess.run(["ffprobe", "-v", "error", str(path)], capture_output=True, text=True, timeout=20)
    return None if proc.returncode == 0 else proc.stderr.strip()[:300]


def test_every_contract_op_is_classified():
    missing = [n for n in _ALL_OPS if n not in dry._CLASSIFICATION]
    assert missing == [], f"contracts/ops/*.json op(s) with no dry-tier row: {missing}"


def test_real_ops_resolve_to_the_pack_handler(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(pack, "resolve", lambda handler: sentinel)  # noqa: ARG005
    errors = [name for name in _REAL_OPS if dry.resolve(registry.get(name)) is not sentinel]
    assert errors == [], f"REAL op(s) that did NOT resolve through pack.resolve: {errors}"
    assert {"media.audio", "cut.audio", "measure.audio"} <= set(_REAL_OPS)


@_NEEDS_FFMPEG
def test_dry_stub_output_x_engine_reader_matrix(tmp_path):
    failures: list[str] = []
    informational: list[str] = []
    for op_name in _STUB_OPS:
        op_dir = tmp_path / op_name.replace(".", "_")
        op_dir.mkdir()
        try:
            outputs = _produce(op_dir, op_name)
        except Exception as e:  # noqa: BLE001 — a produce-time crash IS a matrix row failure
            failures.append(f"{op_name}: stub handler itself raised {type(e).__name__}: {e}")
            continue
        if outputs is None:
            informational.append(f"{op_name}: refuses by name (DryStubUnderivedField)")
            continue
        op = registry.get(op_name)
        declared = {p.id: p for p in op.outputs}
        for port_id, dst in outputs.items():
            port = declared[port_id]
            if port.kind == "json":
                continue
            for path in (dst if isinstance(dst, list) else [dst]):
                err = _ffprobe_opens(path)
                if err is not None:
                    failures.append(f"{op_name}.{port_id} ({path.name}): ffprobe refused it — {err}")
    msg = "broken (op, port, reader) pairs:\n" + "\n".join(failures)
    if informational:
        msg += "\n\ninformational: " + "; ".join(informational)
    assert not failures, msg


def test_cut_apply_stub_durs_matches_contract_shape(tmp_path):
    # No engine-source read here (unlike apply_edl.py:306) — that proof lives in the engine's own matrix.
    keep = _JSON_PARAMS["cut.apply"]["keep"]
    outputs = _produce(tmp_path, "cut.apply", only={"durs"})
    doc = json.loads(outputs["durs"].read_text(encoding="utf-8"))
    rdurs = doc["rdurs"]
    assert len(rdurs) == len(keep)
    assert abs(sum(rdurs) - sum(e - s for s, e in keep)) < 1e-3
