"""podagent/ops/dry.py — contour-dry stand-in handler, reached via `runner.py`'s `pack.resolve` seam when
`ARM_ENV` is armed. Fills every declared output port with one arity-correct file so the unmodified arity
check and `plan_match.verdict` still judge it."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from . import registry

ARM_ENV = "MONTY_OPS_CONTOUR_DRY"

# Mirrored byte-for-byte at scripts/plan_match.py::CONTOUR_DRY_CLAIMS (pod-agent is a separate repo, so this
# is the one place both sides must move together — MISC-62 lock 4 refuses a receipt whose tuple has moved).
CONTOUR_DRY_CLAIMS: dict[str, str] = {
    "taps": "plan-derived, not measured", "pixels": "not rendered", "vram": "not exercised",
    "nvenc": "not exercised", "weights": "cache presence only", "graph": "really built",
    "argv": "really built", "store": "real PUT/GET",
}

# A stand-in for real work never legitimately runs longer than the smallest thing that could stall it.
_LAVFI_BUDGET_S = 30.0


def armed() -> bool:
    return os.environ.get(ARM_ENV, "").strip() not in ("", "0")


def _write_lavfi(dst: Path, *, video: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if video:
        src = ["-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=1"]
        codec = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    else:
        src = ["-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=1"]
        codec = ["-c:a", "aac", "-b:a", "8k"]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *src, "-t", "1", *codec, str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=_LAVFI_BUDGET_S)


def _write_image(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=64x64", "-frames:v", "1", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=_LAVFI_BUDGET_S)


def _write_json(dst: Path, *, seed: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text('{"contour_dry": true, "seed": "%s"}' % hashlib.sha256(seed.encode()).hexdigest()[:12],
                   encoding="utf-8")


_WRITER: dict[str, Callable[[Path], None]] = {
    "video": lambda p: _write_lavfi(p, video=True),
    "audio": lambda p: _write_lavfi(p, video=False),
    "image": _write_image,
}


def _fill_one(dst: Path, kind: str, *, seed: str) -> None:
    if kind == "json":
        _write_json(dst, seed=seed)
        return
    fn = _WRITER.get(kind)
    if fn is None:
        raise registry.OpError(f"contour-dry: no synthesis rule for output kind {kind!r}")
    fn(dst)


def _handler(op: registry.Op) -> Callable[..., None]:
    def run(*, params: dict[str, Any], inputs: dict[str, Path], outputs: dict[str, Any]) -> None:  # noqa: ARG001
        declared = {p.id: p for p in op.outputs}
        for port_id, dst in outputs.items():
            port = declared[port_id]
            targets = dst if isinstance(dst, list) else [dst]
            for i, one in enumerate(targets):
                _fill_one(Path(one), port.kind, seed=f"{op.op}:{port_id}:{i}")
    return run


def resolve(op: registry.Op) -> Callable[..., None]:
    # Same call shape as `pack.resolve(op.handler)`, no pack fetched.
    return _handler(op)
