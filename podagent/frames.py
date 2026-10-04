"""The ONE rule for every timed window this renderer emits: an integer FRAME range on the output grid.

The engine hands every timed element of a final on the output frame grid. Re-quantizing those times into
formatted float seconds (an end of 0.067 in one graph, 0.066666 in another) put two co-terminal
elements a frame apart, so no filter window is ever a float second here: `frame_at` rounds a spec time to
the nearest output frame ONCE, a window is the half-open range [a, b) of those indices, and it is spoken to
ffmpeg in frames — `between(N,a,b-1)`, `trim=start_frame=a:end_frame=b`, `setpts=…+a/FR/TB`.

N is the output frame index of the frame being filtered, `round(t*FR)` with FR the exact grid rational —
NOT ffmpeg's timeline `n`: in a two-input filter (overlay, via framesync) `n` is the count of whichever
input was consumed last, so `between(n,75,134)` on a cutaway overlay fires on frame 75 alone (ffmpeg 6.1,
reproduced). The frame's own PTS is the same on every input of a synced pair, so its index is too.
"""
from __future__ import annotations

import math
from fractions import Fraction

from . import finalize as _finalize


def rate(fps: float | str) -> Fraction:
    """The output grid as an exact rational: a declared float fps snaps the way `-r` does (59.94 is
    60000/1001), a rational string (the grid itself) is taken as written."""
    if isinstance(fps, str):
        return Fraction(fps)
    return Fraction(_finalize.declared_grid(float(fps)))


def frame_at(t: float, fps: float | str) -> int:
    """Nearest output frame index of spec time `t` (half rounds up). The 1e-6 snap first kills float
    dust, so an on-grid time can never land on the far side of a .5 by representation error."""
    return math.floor(round(float(Fraction(t) * rate(fps)), 6) + 0.5)


def frame_range(start: float, end: float, fps: float | str) -> tuple[int, int]:
    """[a, b): start inclusive, end exclusive, both via `frame_at`. A non-empty span never collapses to
    zero frames, and the END stays where `frame_at(end)` put it: a sub-frame span takes the one frame
    BEFORE its end frame, so it stays co-terminal with every other element ending at the same time
    (pushing b to a+1 instead ended [0.030,0.034) on frame 2 and [0.0,0.034) on frame 1 at 30 fps).
    Only a span ending on frame 0, which has no frame before it, takes frame 0."""
    a, b = frame_at(start, fps), frame_at(end, fps)
    if end > start and b <= a:
        a, b = (b - 1, b) if b > 0 else (0, 1)
    return a, max(a, b)


def index(fps: float | str) -> str:
    """N: the output frame index of the frame being filtered, from its own PTS on the exact grid."""
    r = rate(fps)
    return f"round(t*{r.numerator})" if r.denominator == 1 else f"round(t*{r.numerator}/{r.denominator})"


def between(a: int, b: int, fps: float | str) -> str:
    """Gate over the half-open output-frame range [a, b)."""
    return f"between({index(fps)},{a},{b - 1})"


def seconds(a: int, fps: float | str) -> str:
    """Frame index `a` as an exact rational time expression (no formatted float): a/FR."""
    r = rate(fps)
    return f"{a * r.denominator}/{r.numerator}"


def pts_at(a: int, fps: float | str) -> str:
    """`setpts` offset that seats a stream's first frame on output frame `a`."""
    return f"{seconds(a, fps)}/TB"
