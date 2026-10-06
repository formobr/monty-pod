"""podagent.captions — the libass ASS builder (no ffmpeg). Proves oneword emits Dialogue events,
the accent lights `hot` words, and the portrait safe-zone clamp. (Phrase parity: engine A/B.)

The colours are ARGUMENTS here, never fixtures-as-truth: this file used to feed one tenant's lime in and
assert it came back out, which made the pod's hardcoded palette look verified. The colours below are
deliberately NOT any brand's (see test_no_colour_literal_survives_in_the_module)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from podagent import captions

_WORDS = [
    {"text": "true", "start": 0.1, "end": 0.4, "hot": False},
    {"text": "story", "start": 0.45, "end": 0.9, "hot": True},
    {"text": "here", "start": 0.95, "end": 1.3, "hot": False},
]


def _real_font():
    """A real TTF: every style measures glyph width via PIL (oneword too — its safe-width shrink)."""
    import glob
    for pat in ("/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/DejaVuSans*.ttf"):
        found = glob.glob(pat, recursive=True)
        if found:
            return Path(found[0])
    pytest.skip("no system TTF available to measure caption widths")

_FG = "#123456"       # arbitrary, on purpose: the builder must draw what it is HANDED
_ACCENT = "#abcdef"


def test_oneword_emits_events_and_lights_hot() -> None:
    ass = captions.build_ass(_WORDS, font=_real_font(), w=1080, h=1920, fg=_FG, accent=_ACCENT, style="oneword")
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert ass.count("Dialogue:") == 3
    assert "STORY" in ass and "TRUE" in ass          # words are upper-cased at draw time
    accent = captions._ac(_ACCENT)
    story = next(ln for ln in ass.splitlines() if "STORY" in ln)
    assert f"\\1c{accent}" in story                   # the hot word carries the accent colour
    true = next(ln for ln in ass.splitlines() if "TRUE" in ln)
    assert f"\\1c{accent}" not in true                # a non-hot word does not
    assert captions._ac(_FG) in ass                   # the body style is the fg it was handed


def test_the_body_colour_is_the_one_handed_in_not_a_house_white() -> None:
    """NEGATIVE — reddens the moment a module-level FG (or any other default) draws the body instead of the
    caller's brand value. The pod is the FINAL renderer: a literal here ships in every delivered video."""
    ass = captions.build_ass(_WORDS, font=_real_font(), w=1080, h=1920, fg="#010203", accent=_ACCENT)
    assert captions._ac("#010203") in ass
    assert captions._ac("#f2f2f0") not in ass, "the retired hardcoded off-white is back in the ASS head"


def test_colours_are_required_arguments() -> None:
    """A default would be a second SSOT for a brand value; the seam must fail loudly, not silently recolour."""
    for missing in ({"accent": _ACCENT}, {"fg": _FG}):
        with pytest.raises(TypeError):
            captions.build_ass(_WORDS, font=_real_font(), w=1080, h=1920, **missing)  # type: ignore[arg-type]


def test_no_colour_literal_survives_in_the_module() -> None:
    """NEGATIVE — the file itself must hold no `#rrggbb`. The engine gate (its own brand-literals test)
    proves it is not THIS brand's colour; this proves it is not ANY colour, defaults included."""
    src = Path(captions.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    found = [m for m in re.findall(r"#[0-9a-fA-F]{3,8}\b", code)]
    assert not found, f"colour literal(s) back in podagent/captions.py: {found}"


class _Caps:
    def __init__(self, accent=None):
        self.accent = accent


class _Brand:
    def __init__(self, tokens):
        self.tokens = tokens


class _MP:
    def __init__(self, brand=None):
        self.brand = brand


def _colours(caps, mp):
    from podagent.render import _caption_colours
    return _caption_colours(caps, mp)


def test_colours_come_from_the_data_that_crossed() -> None:
    mp = _MP(_Brand({"color": {"fg": "#111213", "accent": "#141516"}}))
    assert _colours(_Caps("#aabbcc"), mp) == ("#111213", "#aabbcc")   # spec field wins for accent
    assert _colours(_Caps(None), mp) == ("#111213", "#141516")        # else the crossed brand tokens


def test_a_burn_with_no_brand_at_all_fails_instead_of_guessing_a_palette() -> None:
    """NEGATIVE — reddens if `caps.accent or "#d6ff3a"` (or any other house colour) comes back: the pod is the
    final renderer, so a guessed accent is a DELIVERED video in another tenant's brand."""
    with pytest.raises(RuntimeError):
        _colours(_Caps(None), _MP(None))


def test_the_body_fallback_is_neutral_not_a_tenant_off_white() -> None:
    """A brandless job degrades to plain white — never #f2f2f0, which is one specific channel's warm white."""
    from podagent import render
    fg, _ = _colours(_Caps("#aabbcc"), _MP(None))
    assert fg == render._NEUTRAL_FG == "#ffffff"


def test_portrait_center_y_clamped_to_safe_zone() -> None:
    # a center_y past the bottom UI reserve is pulled up; landscape is left untouched
    assert captions._clamp_cy(0.99, 60, 1080, 1920) < 0.99
    assert captions._clamp_cy(0.99, 60, 1920, 1080) == 0.99


# ── bold (4th style): block layout like phrase, heavier weight ───────────────────────────────────────

def test_bold_style_head_carries_bold_flag_and_block_layout() -> None:
    font = _real_font()
    ass = captions.build_ass(_WORDS, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style="bold")
    style_line = next(ln for ln in ass.splitlines() if ln.startswith("Style: Cap,"))
    fields = style_line.split(",")
    assert fields[2] == str(captions.BOLD_SIZE)                 # a dedicated, bigger Fontsize than phrase
    assert fields[7] == "1"                                     # Bold:1 (Format col index 7 after "Style:")
    assert "\\an8" in ass or "\\an5" in ass                     # block-anchored, not the oneword \move look


def test_bold_style_accents_the_spoken_word_like_phrase() -> None:
    font = _real_font()
    ass = captions.build_ass(_WORDS, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style="bold")
    accent = captions._inline_c(_ACCENT)
    assert f"\\1c{accent}" in ass
    assert "\\fscx" not in ass                                  # colour accent, not phrase_jump's scale bounce


def test_bold_size_sits_between_phrase_and_title() -> None:
    assert captions.PHRASE_SIZE < captions.BOLD_SIZE < captions.TITLE


def test_the_three_legacy_styles_keep_their_own_size_and_stay_not_bold() -> None:
    """NEGATIVE — proves the `bold` addition did not leak BOLD_SIZE/Bold:1 into phrase/phrase_jump (byte-exact
    pin for oneword lives in this module's own captions-ASS golden test; this covers the two _build_ass_phrase callers
    the golden test does not hash)."""
    font = _real_font()
    kw = dict(font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, center_y=0.76)
    for style in ("phrase", "phrase_jump"):
        ass = captions.build_ass(_WORDS, style=style, **kw)
        style_line = next(ln for ln in ass.splitlines() if ln.startswith("Style: Cap,"))
        fields = style_line.split(",")
        assert fields[2] == str(captions.PHRASE_SIZE), f"{style}: Fontsize drifted from PHRASE_SIZE"
        assert fields[7] == "0", f"{style}: Bold flag leaked on from the bold addition"


# ── safe width: the same side box as the browser preview ─────────────────────────────────────────────

def test_side_limits_are_the_engine_safe_box() -> None:
    """Twin parity (like _SAFE_BOTTOM): the engine's safe-zone box `_LEFT, _RIGHT = 112, 951`."""
    assert (captions._REF_W, captions._SAFE_LEFT, captions._SAFE_RIGHT) == (1080, 112, 951)
    assert captions._safe_maxw(1080) == 839
    assert captions._safe_maxw(2160) == 1678                    # scales with the frame width


class _FixedFont:
    """Every word measures 400 px, a space 40 px: two words make a line of exactly 840 px."""
    def getlength(self, text: str) -> float:
        return 40.0 if text == " " else 400.0


def test_lines_wrap_to_the_preview_safe_box() -> None:
    """NEGATIVE — an 840 px line fit the old `w - 2*110` = 860 box but crosses the 822-839 safe box."""
    fnt = _FixedFont()
    two = [{"text": "aaaa"}, {"text": "bbbb"}]
    lines = captions._wrap_lines(two, fnt, fnt.getlength(" "), 1080)
    assert [len(ln) for ln in lines] == [1, 1]
    # the same line on a wider frame still fits on one line: the wrap itself is unchanged
    assert [len(ln) for ln in captions._wrap_lines(two, fnt, 40.0, 1300)] == [2]


_LONG = "Supercalifragilisticexpialidoc"   # 30 characters, one token


@pytest.mark.parametrize("style", ["oneword", "phrase", "phrase_jump", "bold"])
def test_an_overlong_word_is_shrunk_to_the_safe_width_in_every_style(style: str) -> None:
    assert len(_LONG) == 30
    font = _real_font()
    words = [{"text": _LONG, "start": 0.1, "end": 0.9, "hot": False}]
    ass = captions.build_ass(words, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style=style)
    size = {"oneword": captions.TITLE, "bold": captions.BOLD_SIZE}.get(style, captions.PHRASE_SIZE)
    maxw = captions._safe_maxw(1080)
    word = captions._clean(_LONG)
    assert captions._font_at(str(font), captions._size_px(size)).getlength(word) > maxw  # needs a shrink
    events = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:") and word in ln]
    assert events
    for ev in events:
        m = re.search(r"\\fs(\d+)", ev)
        assert m, f"{style}: over-wide word drawn at full size: {ev}"
        fs = int(m.group(1))
        assert fs < size
        assert captions._font_at(str(font), captions._size_px(fs)).getlength(word) <= maxw


def test_a_url_long_token_still_fits_and_warns_once(capsys) -> None:
    """No lower floor: a 60-char token lands inside the safe box, however small, and is logged."""
    font = _real_font()
    url = "HTTPSEXAMPLECOMAVERYLONGPATHWITHNOBREAKSATALLINSIDEITWHATSOEVER"
    words = [{"text": url, "start": 0.1, "end": 0.9}]
    ass = captions.build_ass(words, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style="oneword")
    fs = int(re.search(r"\\fs(\d+)", ass).group(1))
    assert captions._font_at(str(font), captions._size_px(fs)).getlength(url) <= captions._safe_maxw(1080)
    err = capsys.readouterr().err
    assert err.count("WARN") == 1 and f"{len(url)}-char" in err


def test_a_word_that_fits_carries_no_size_override() -> None:
    ass = captions.build_ass(_WORDS, font=_real_font(), w=1080, h=1920, fg=_FG, accent=_ACCENT)
    assert "\\fs" not in ass


def test_a_repeated_overlong_word_warns_once_not_per_occurrence(capsys) -> None:
    """NEGATIVE — the same over-long word crossing two phrase blocks used to print one WARN per block
    (each block re-measures independently); it must warn once per distinct word per build_ass call."""
    font = _real_font()
    url = "HTTPSEXAMPLECOMAVERYLONGPATHWITHNOBREAKSATALLINSIDEITWHATSOEVER"
    words = [
        {"text": url, "start": 0.1, "end": 0.9, "hot": False},
        {"text": url, "start": 5.0, "end": 5.8, "hot": False},   # gap > PHRASE_WINDOW_MS: a new block
    ]
    captions.build_ass(words, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style="phrase")
    assert capsys.readouterr().err.count("WARN") == 1


def _positions(ass: str) -> list[int]:
    """Every x coordinate an ASS `\\pos`/`\\move` tag places a caption at."""
    xs = []
    for m in re.finditer(r"\\(?:pos|move)\((-?\d+),-?\d+(?:,(-?\d+),-?\d+)?", ass):
        xs.append(int(m.group(1)))
        if m.group(2) is not None:
            xs.append(int(m.group(2)))
    return xs


def test_a_full_width_line_centres_on_the_safe_box_not_the_frame() -> None:
    """NEGATIVE — centring a max-width line on the frame (540) instead of the safe box (531.5) pushed
    its right edge ~8.5px past _SAFE_RIGHT even though its measured width was exactly `_safe_maxw`."""
    assert captions._safe_center(1080) == (112 + 951) / 2
    font = _real_font()
    word = "A" * 30   # shrunk to land at (about) the full safe width
    words = [{"text": word, "start": 0.1, "end": 0.9}]
    for style in ("phrase", "bold"):
        ass = captions.build_ass(words, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style=style)
        size = captions.BOLD_SIZE if style == "bold" else captions.PHRASE_SIZE
        fs_m = re.search(r"\\fs(\d+)", ass)
        fs = int(fs_m.group(1)) if fs_m else size
        ww = captions._font_at(str(font), captions._size_px(fs)).getlength(word)
        xc = _positions(ass)[0]
        assert captions._SAFE_LEFT <= xc - ww / 2 and xc + ww / 2 <= captions._SAFE_RIGHT + 0.5, style


def test_phrase_jump_lays_out_no_wider_than_its_own_wrap_decided() -> None:
    """CONFIRMED bug — _wrap_lines budgeted `spc` per gap but _phrase_jump_block laid out with
    `spc + 8`, so a line the wrap accepted (<= the safe width) rendered wider than it in the real
    per-word \\pos/\\move layout, crossing the safe box on every extra word in the line."""
    font = _real_font()
    toks = "I AM SO SO SO SO SO SO SO SO SO SO SO SO SO SO".split()
    words = [{"text": t, "start": i * 0.2, "end": i * 0.2 + 0.15} for i, t in enumerate(toks)]
    ass = captions.build_ass(words, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style="phrase_jump")
    xs = _positions(ass)
    assert xs, "no positioned events emitted"
    assert min(xs) >= captions._SAFE_LEFT and max(xs) <= captions._SAFE_RIGHT


def test_phrase_jump_bounce_overshoot_stays_inside_the_safe_box() -> None:
    """CONFIRMED bug — the active word's bounce briefly scales it to JUMP_OVERSHOOT/JUMP_SCALE percent;
    a lone over-long word shrunk to exactly fit at rest still crossed the box mid-bounce."""
    font = _real_font()
    word = _LONG
    words = [{"text": word, "start": 0.1, "end": 0.9}]
    ass = captions.build_ass(words, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style="phrase_jump")
    fs = int(re.search(r"\\fs(\d+)", ass).group(1))
    ww = captions._font_at(str(font), captions._size_px(fs)).getlength(captions._clean(word))
    xc = _positions(ass)[0]
    overshot = ww * captions.JUMP_MAX_SCALE
    assert xc - overshot / 2 >= captions._SAFE_LEFT
    assert xc + overshot / 2 <= captions._SAFE_RIGHT
