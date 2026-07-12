"""podagent.captions — the libass ASS builder (no ffmpeg). Proves oneword emits Dialogue events,
the accent lights `hot` words, and the portrait safe-zone clamp. (Phrase parity: engine A/B.)"""
from __future__ import annotations

from pathlib import Path

from podagent import captions

_WORDS = [
    {"text": "true", "start": 0.1, "end": 0.4, "hot": False},
    {"text": "story", "start": 0.45, "end": 0.9, "hot": True},
    {"text": "here", "start": 0.95, "end": 1.3, "hot": False},
]
_FONT = Path("/fonts/Inter-ExtraBold.ttf")  # never opened for oneword (no PIL measuring)


def test_oneword_emits_events_and_lights_hot() -> None:
    ass = captions.build_ass(_WORDS, font=_FONT, w=1080, h=1920, accent="#d6ff3a", style="oneword")
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert ass.count("Dialogue:") == 3
    assert "STORY" in ass and "TRUE" in ass          # words are upper-cased at draw time
    lime = captions._ac("#d6ff3a")
    story = next(ln for ln in ass.splitlines() if "STORY" in ln)
    assert f"\\1c{lime}" in story                     # the hot word carries the accent colour
    true = next(ln for ln in ass.splitlines() if "TRUE" in ln)
    assert f"\\1c{lime}" not in true                  # a non-hot word does not


def test_portrait_center_y_clamped_to_safe_zone() -> None:
    # a center_y past the bottom UI reserve is pulled up; landscape is left untouched
    assert captions._clamp_cy(0.99, 60, 1080, 1920) < 0.99
    assert captions._clamp_cy(0.99, 60, 1920, 1080) == 0.99
