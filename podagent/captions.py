"""Subtitle track via libass (no browser): brand-neutral ASS builder for four looks
(oneword/phrase/phrase_jump/bold). The brain bakes words+accent+centerY into the spec; the pod draws
them with a delivered TTF.

NO COLOUR LITERAL LIVES IN THIS FILE. `fg` and `accent` are REQUIRED arguments: this module used to spell
one tenant's warm white and one tenant's lime as defaults, and since the pod is the FINAL renderer every
delivered video burned them — a second tenant's captions came out in the first one's palette with every
test green. The caller resolves both from the crossing brand data (render_onepass._write_ass)."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# ── look, tuned to the brand build_motion captions (A/B-verified) ────────────────────────────────────
TITLE = 92          # libass Fontsize == 64px on-screen (different metric than CSS px)
OUTLINE = 3
SHADOW = 4
BLUR = 4
FADE_IN_MS = 83
RISE_PX = 16
RISE_MS = 120
HOLD_AFTER = 0.8    # keep a word up this long past its end if no next word

PHRASE_SIZE = 70
PHRASE_PX = round(PHRASE_SIZE * 64 / 92)   # libass Fontsize → on-screen px (measure/layout in THIS)
PHRASE_WINDOW_MS = 700
PHRASE_MAX_LINES = 2
JUMP_OVERSHOOT = 120
JUMP_SCALE = 110
JUMP_GROW_MS = 90
JUMP_GAP = 8         # extra px between jump words beyond a plain space (the wrap budget must reserve it too)
JUMP_MAX_SCALE = max(JUMP_OVERSHOOT, JUMP_SCALE) / 100.0   # worst-case bounce scale a jump word ever hits

BOLD_SIZE = 80      # "bold" look: same ≤2-line block/wrap as phrase, heavier weight + bigger than PHRASE_SIZE

# safe-zone bottom reserve (9:16 only), from the 1080×1920 reference; scales with height
_REF_H = 1920
_SAFE_BOTTOM = 1565

# safe-zone sides: the SAME box the browser preview wraps to — engine scripts/safezone.py:23-24
# (`_LEFT, _RIGHT = 112, 951` on 1080×1920, brand safe box in brand_tokens.py). Scales with width.
_REF_W = 1080
_SAFE_LEFT = 112
_SAFE_RIGHT = 951
_SHRINK_WARN = 0.5  # a word drawn below this fraction of its style size is logged (never floored)


def _safe_maxw(w: int) -> float:
    """Widest a caption line may measure: the safe box's side limits scaled to the frame width."""
    return (_SAFE_RIGHT - _SAFE_LEFT) * w / _REF_W


def _safe_center(w: int) -> float:
    """Centre a caption on the SAFE BOX, not the raw frame centre: the box is off-centre in the frame
    (112..951 of 1080 centres on 531.5, not 540), so a line at the full `_safe_maxw` width centred on
    the frame would cross the right limit by the same offset. Centring on the box instead makes a
    full-width line's edges land exactly on _SAFE_LEFT/_SAFE_RIGHT."""
    return (_SAFE_LEFT + _SAFE_RIGHT) / 2 * w / _REF_W


def _caption_max_y(below_px: int, h: int) -> int:
    return round(_SAFE_BOTTOM * h / _REF_H) - below_px


def _ac(hexc: str, aa: str = "00") -> str:
    """#rrggbb → ASS &HaaBBGGRR."""
    h = hexc.lstrip("#")
    return f"&H{aa}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _inline_c(hexc: str) -> str:
    """#rrggbb → ASS inline \\1c form &HBBGGRR&."""
    h = hexc.lstrip("#")
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def _tc(t: float) -> str:
    cs = int(round(t * 100))
    hh = cs // 360000; cs %= 360000
    mm = cs // 6000; cs %= 6000
    ss = cs // 100; cs %= 100
    return f"{hh:d}:{mm:02d}:{ss:02d}.{cs:02d}"


def _clean(text: object) -> str:
    s = str(text).upper().replace("\n", " ").replace("{", "").replace("}", "")
    return s.strip(" .,!?;:…«»\"'()-—–").strip()


def _clamp_cy(center_y: float, below_px: int, w: int, h: int) -> float:
    """Portrait only: cap center_y so `below_px` under the anchor clears the bottom UI reserve."""
    if h <= w:
        return center_y
    return min(center_y, _caption_max_y(below_px, h) / h)


def _ass_head(white: str, w: int, h: int, size: int = TITLE, *, bold: bool = False) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Inter ExtraBold,{size},{white},&H000000FF,&H00000000,&H50000000,{1 if bold else 0},0,0,0,100,100,0,0,1,{OUTLINE},{SHADOW},5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(words: list[dict], *, font: Path, w: int, h: int, fg: str, accent: str,
              center_y: float = 0.76, style: str = "oneword",
              window_ms: int = PHRASE_WINDOW_MS) -> str:
    """words = [{text,start,end,hot?}, …] → ASS text; `style` picks oneword/phrase/phrase_jump.

    `fg` (body colour) and `accent` (hot-word colour) are required — see the module docstring."""
    if style in ("phrase", "phrase_jump"):
        return _build_ass_phrase(words, font, w, h, fg, accent, center_y, window_ms, kind=style)
    if style == "bold":
        return _build_ass_phrase(words, font, w, h, fg, accent, center_y, window_ms, kind="bold",
                                  size=BOLD_SIZE, bold=True)
    white = _ac(fg)
    lime = _ac(accent)
    center_y = _clamp_cy(center_y, TITLE // 2 + RISE_PX, w, h)
    ymid = round(h / 2 + (center_y - 0.5) * h)
    xc = round(_safe_center(w))
    head = _ass_head(white, w, h)
    maxw = _safe_maxw(w)
    warned: set = set()
    n = len(words)
    lines = []
    for i, wd in enumerate(words):
        st = float(wd["start"])
        nxt = float(words[i + 1]["start"]) if i + 1 < n else 1e9
        su = min(nxt, float(wd["end"]) + HOLD_AFTER)
        su = su if su > st else st + 0.1
        txt = _clean(wd["text"])
        fs = _fit_size(font, txt, TITLE, maxw, warned)
        tag = (f"{{\\move({xc},{ymid + RISE_PX},{xc},{ymid},0,{RISE_MS})"
               f"\\fad({FADE_IN_MS},0)\\blur{BLUR}{_fs_tag(fs)}")
        if wd.get("hot"):
            tag += f"\\1c{lime}"
        tag += "}"
        lines.append(f"Dialogue: 0,{_tc(st)},{_tc(su)},Cap,,0,0,0,,{tag}{txt}")
    return head + "\n".join(lines) + "\n"


def _phrase_font(font: Path, px: int = PHRASE_PX):
    return _font_at(str(font), px)   # measure at on-screen px, not the libass Fontsize


@lru_cache(maxsize=64)
def _font_at(font: str, px: int):
    from PIL import ImageFont
    return ImageFont.truetype(font, px)


def _size_px(size: int) -> int:
    """libass Fontsize → on-screen px (the same metric conversion as PHRASE_PX)."""
    return round(size * 64 / TITLE)


def _fit_size(font: Path, text: str, size: int, maxw: float, warned: set | None = None) -> int | None:
    """None when `text` fits `maxw` at `size`; else the largest libass Fontsize at which it does.
    No lower floor: a 40-char URL still lands inside the safe box, however small (logged below 50% —
    once per distinct (text, size) via `warned`, so one repeated over-long word across many blocks
    doesn't spam the log with a line per occurrence)."""
    ww = _font_at(str(font), _size_px(size)).getlength(text)
    if ww <= maxw:
        return None
    fs = max(1, int(size * maxw / ww))
    while fs > 1 and _font_at(str(font), _size_px(fs)).getlength(text) > maxw:
        fs -= 1
    if fs < size * _SHRINK_WARN:
        key = (text, size)
        if warned is None or key not in warned:
            if warned is not None:
                warned.add(key)
            print(f"[captions] WARN a {len(text)}-char word is shrunk to {fs / size:.0%} of its size "
                  f"to fit the safe width", file=sys.stderr, flush=True)
    return fs


def _fs_tag(fs: int | None) -> str:
    return "" if fs is None else f"\\fs{fs}"


def _wrap_lines(block, fnt, spc, w, maxw=None):
    """Greedily wrap words into ≤lines by pixel width. Returns [[(word, w_px), …], …].

    `maxw` overrides the safe-box default (jump passes a shrunk budget — see JUMP_MAX_SCALE) and
    `spc` is the caller's actual inter-word gap (jump passes spc+JUMP_GAP, matching its own layout)."""
    if maxw is None:
        maxw = _safe_maxw(w)
    lines, line, used = [], [], 0.0
    for wd in block:
        ww = fnt.getlength(_clean(wd["text"]))
        adv = ww if not line else spc + ww
        if line and used + adv > maxw:
            lines.append(line); line, used = [], 0.0; adv = ww
        line.append((wd, ww)); used += adv
    if line:
        lines.append(line)
    return lines


def _group_blocks(words, fnt, spc, window_ms, w, maxw=None):
    """Break a new block on a pause > window_ms, or when the next word would need a 3rd line."""
    gap_s = window_ms / 1000.0
    blocks, cur = [], []
    for wd in words:
        if cur:
            gap = float(wd["start"]) - float(cur[-1]["end"])
            if gap > gap_s or len(_wrap_lines(cur + [wd], fnt, spc, w, maxw)) > PHRASE_MAX_LINES:
                blocks.append(cur); cur = []
        cur.append(wd)
    if cur:
        blocks.append(cur)
    return blocks


def _build_ass_phrase(words, font, w, h, fg, accent, center_y, window_ms, *, kind,
                       size=PHRASE_SIZE, bold=False):
    """Stable, centred ≤2-line block pinned at a fixed y (never jumps); kind = phrase | phrase_jump | bold
    (bold reuses the phrase colour-block layout at a bigger, Bold:1 style)."""
    white = _ac(fg)
    white_c = _inline_c(fg)
    accent_c = _inline_c(accent)
    px = _size_px(size)
    fnt = _phrase_font(font, px)
    spc = fnt.getlength(" ")
    line_h = round(px * 1.5)
    center_y = _clamp_cy(center_y, line_h, w, h)
    y_top = round(h * center_y) - line_h
    xc = round(_safe_center(w))
    base_maxw = _safe_maxw(w)
    if kind == "phrase_jump":
        # jump's active word bounces up to JUMP_MAX_SCALE while animating, and lays words out with an
        # extra JUMP_GAP beyond a plain space — the wrap budget must reserve both, or a line/word that
        # measures inside the safe box at rest can still cross it mid-bounce or at its actual layout width.
        wrap_gap, wrap_maxw = spc + JUMP_GAP, base_maxw / JUMP_MAX_SCALE
    else:
        wrap_gap, wrap_maxw = spc, base_maxw
    blocks = _group_blocks(words, fnt, wrap_gap, window_ms, w, wrap_maxw)
    out = []
    warned: set = set()
    for bi, block in enumerate(blocks):
        lines = _wrap_lines(block, fnt, wrap_gap, w, wrap_maxw)
        # the wrap leaves a word wider than the safe box alone on its line: shrink that line to fit
        fits = [_fit_size(font, _clean(ln[0][0]["text"]), size, wrap_maxw, warned) if len(ln) == 1 else None
                for ln in lines]
        b_start = float(block[0]["start"])
        nxt = float(blocks[bi + 1][0]["start"]) if bi + 1 < len(blocks) else 1e9
        b_end = min(nxt, float(block[-1]["end"]) + HOLD_AFTER)
        if kind == "phrase_jump":
            out += _phrase_jump_block(lines, fits, b_start, b_end, y_top, line_h, wrap_gap, xc, white_c,
                                      accent_c, font)
        else:
            out += _phrase_colour_block(lines, fits, b_start, b_end, y_top, line_h, xc, white_c, accent_c)
    return _ass_head(white, w, h, size, bold=bold) + "\n".join(out) + "\n"


def _phrase_colour_block(lines, fits, b_start, b_end, y_top, line_h, xc, white_c, accent_c):
    """Colour-highlight: libass-native centred lines (static white) + a per-word accent overlay."""
    out = []
    for i, line in enumerate(lines):
        yc = y_top + i * line_h
        txt = " ".join((f"{{\\1c{accent_c}}}{_clean(wd['text'])}{{\\1c{white_c}}}" if wd.get("hot")
                        else _clean(wd["text"])) for wd, _ in line)
        out.append(f"Dialogue: 1,{_tc(b_start)},{_tc(b_end)},Cap,,0,0,0,,"
                   f"{{\\an8\\pos({xc},{yc})\\fad({FADE_IN_MS},0)\\blur{BLUR}{_fs_tag(fits[i])}}}{txt}")
    for i, line in enumerate(lines):
        yc = y_top + i * line_h
        m = len(line)
        for k, (wd, _) in enumerate(line):
            st = float(wd["start"])
            nxt = float(line[k + 1][0]["start"]) if k + 1 < m else b_end
            end = nxt if nxt > st else st + 0.1
            parts = [(f"{{\\1c{accent_c}}}{_clean(w2['text'])}{{\\1c{white_c}}}" if (jj == k or w2.get("hot"))
                      else _clean(w2["text"])) for jj, (w2, _) in enumerate(line)]
            out.append(f"Dialogue: 2,{_tc(st)},{_tc(end)},Cap,,0,0,0,,"
                       f"{{\\an8\\pos({xc},{yc})\\blur{BLUR}{_fs_tag(fits[i])}}}{' '.join(parts)}")
    return out


def _jump_bounce(d_ms):
    """Inline scale tags for one word's bounce: spring to overshoot, settle, then ease back to 100."""
    b = (f"\\fscx100\\fscy100\\t(0,{JUMP_GROW_MS},0.4,\\fscx{JUMP_OVERSHOOT}\\fscy{JUMP_OVERSHOOT})"
         f"\\t({JUMP_GROW_MS},{2 * JUMP_GROW_MS},\\fscx{JUMP_SCALE}\\fscy{JUMP_SCALE})")
    if d_ms > 2 * JUMP_GROW_MS + 80:
        b += f"\\t({d_ms - 80},{d_ms},\\fscx100\\fscy100)"
    return b


def _phrase_jump_block(lines, fits, b_start, b_end, y_top, line_h, gap, xc, white_c, accent_c, font):
    """Every word at its own \\pos (fixed y); the spoken word bounces in scale, neighbours slide sideways.
    `gap` is the caller's actual inter-word spacing (spc + JUMP_GAP) — the SAME value the wrap budget
    used, so a line that fit the wrap never lays out wider than the wrap decided."""
    out = []
    grow_f = JUMP_SCALE / 100.0 - 1.0
    for i, line in enumerate(lines):
        cy = round(y_top + i * line_h + PHRASE_PX * 0.5)
        fs = _fs_tag(fits[i])
        ww = [w_px for _, w_px in line]
        if fits[i] is not None:   # a lone over-wide word, shrunk: centre it on its drawn width
            ww = [_font_at(str(font), _size_px(fits[i])).getlength(_clean(line[0][0]["text"]))]
        toks = [_clean(wd["text"]) for wd, _ in line]
        starts = [float(wd["start"]) for wd, _ in line]
        m = len(line)
        total = sum(ww) + gap * (m - 1)
        x = xc - total / 2.0
        cx = []
        for k in range(m):
            cx.append(round(x + ww[k] / 2.0)); x += ww[k] + gap
        segs = []
        if starts[0] > b_start + 0.02:
            segs.append((b_start, starts[0], -1))
        for k in range(m):
            segs.append((starts[k], starts[k + 1] if k + 1 < m else b_end, k))
        for j in range(m):
            prev_x = None
            hot_col = f"\\1c{accent_c}" if line[j][0].get("hot") else ""
            for s, e, act in segs:
                e = e if e > s else s + 0.05
                fade = f"\\fad({FADE_IN_MS},0)" if abs(s - b_start) < 0.02 else ""
                if act == j or act < 0:
                    tx = cx[j]
                else:
                    push = round(grow_f * ww[act] / 2.0)
                    tx = cx[j] + (-push if j < act else push)
                if act == j:
                    tag = f"{{\\an5\\pos({cx[j]},{cy}){fade}{_jump_bounce(round((e - s) * 1000))}{hot_col}\\blur{BLUR}{fs}}}"
                elif prev_x is not None and prev_x != tx:
                    tag = f"{{\\an5\\move({prev_x},{cy},{tx},{cy},0,{JUMP_GROW_MS}){fade}{hot_col}\\blur{BLUR}{fs}}}"
                else:
                    tag = f"{{\\an5\\pos({tx},{cy}){fade}{hot_col}\\blur{BLUR}{fs}}}"
                out.append(f"Dialogue: 1,{_tc(s)},{_tc(e)},Cap,,0,0,0,,{tag}{toks[j]}")
                prev_x = tx
    return out
