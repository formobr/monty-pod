"""GATE: the pod's head-below slide IS the engine's — same numbers, critical damping, rest is the cap.
This tier renders the MASTER, so an overshoot kept here after the engine dropped it (fbr ff22e5e4) ships in
the delivered cut while the approved preview no longer has it; the owner TSX is absent in a bare pod checkout."""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from podagent import mograph

BEAT = (2.0, 6.0)
# <engine>/pod-agent/tests/ → <engine>/: present only when the pod is checked out as the engine's submodule
OWNER = Path(__file__).resolve().parents[2] / "remotion" / "src" / "MontagePreview.tsx"


def _curve(step: float = 0.01):
    s, e = BEAT
    ts = [s - 0.5 + step * n for n in range(int((e - s + 1.0) / step))]
    return ts, [mograph._hb_settle_progress(t, s, e) for t in ts]


def test_the_constants_are_the_engine_values() -> None:
    """Literals, not a derivation: the pod ships without the engine tree and must still be pinned."""
    assert mograph.HB_DROP_FRAC == 0.33
    assert mograph.HEAD_SETTLE_SEC == 0.4
    assert mograph._HB_W0 == (140 / 0.9) ** 0.5


@pytest.mark.skipif(not OWNER.is_file(), reason="engine tree not around this submodule checkout")
def test_the_constants_still_equal_the_declared_owner() -> None:
    src = OWNER.read_text(encoding="utf-8")

    def ts_const(name: str) -> float:
        m = re.search(rf"^export const {name} = ([0-9.]+);", src, re.M)
        assert m, f"{name} is no longer an exported literal in MontagePreview.tsx — the SSOT moved"
        return float(m.group(1))

    assert mograph.HB_DROP_FRAC == ts_const("HB_DROP_FRAC")
    assert mograph.HEAD_SETTLE_SEC == ts_const("HEAD_SETTLE_SEC")
    assert "2 * Math.sqrt(140 * 0.9)" in src, "the owner's glide is not critically damped"


def test_the_head_settle_never_overshoots_its_rest() -> None:
    """THE TICKET: a settle that leaves rest is not a settle — it reads as the head jumping."""
    _, p = _curve()
    assert max(p) == pytest.approx(1.0, abs=1e-3), max(p)
    assert max(p) <= 1.0 + 1e-9, "the head travelled PAST the seated position"
    assert min(p) >= 0.0


def test_the_progress_is_monotone_in_both_edge_windows() -> None:
    """Same easing at both edges: down at the beat start, up on the reversal — never back and forth."""
    s, e = BEAT
    ts, p = _curve()
    down = [v for t, v in zip(ts, p, strict=False) if s <= t <= s + mograph.HEAD_SETTLE_SEC]
    up = [v for t, v in zip(ts, p, strict=False) if e - mograph.HEAD_SETTLE_SEC <= t <= e]
    assert down and up
    assert down == sorted(down), "the entry edge reverses on itself (overshoot)"
    assert up == sorted(up, reverse=True), "the exit edge reverses on itself (overshoot)"


def test_the_ffmpeg_expression_agrees_that_rest_is_the_cap() -> None:
    """NEGATIVE: a `1.2` clamp or a cos/sin (underdamped) body re-opens the preview↔master split."""
    expr = mograph._hb_settle_y_expr(*BEAT)
    assert expr.endswith("\\,0\\,1)"), expr
    assert "1.2" not in expr
    assert "cos(" not in expr and "sin(" not in expr


def test_the_expression_tracks_the_python_progress() -> None:
    """The y expression is what actually renders — it and _hb_settle_progress must be ONE curve."""
    s, e = BEAT
    body = mograph._hb_settle_y_expr(s, e).replace("\\,", ",")[len(f"{mograph.HB_DROP_FRAC}*H*"):]
    scope = {"exp": math.exp, "max": max, "gte": lambda a, b: a >= b,
             "clip": lambda v, lo, hi: min(max(v, lo), hi),
             "_if": lambda c, a, b: a if c else b}
    for t in (1.9, 2.0, 2.2, 3.0, 5.6, 5.8, 6.0):
        got = eval(body.replace("if(", "_if("), {"__builtins__": {}}, {**scope, "t": t})  # noqa: S307
        assert got == pytest.approx(mograph._hb_settle_progress(t, s, e), abs=1e-6)
