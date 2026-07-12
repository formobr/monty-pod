"""Cover (clickbait thumbnail) render for the final composite — a port of add_cover.py to the
brand-agnostic pod: draw over the base frame (darken band, autofit headline, elements, logo), weld the
still ~hold s onto the END of the video. All brand values (colours/font/logo) arrive in SpecCover/inputs."""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

RGB = tuple[int, int, int]
_FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _color(name: Any, colors: dict[str, RGB], default: str = "white") -> RGB:
    if isinstance(name, (list, tuple)):
        return (int(name[0]), int(name[1]), int(name[2]))
    return colors.get(name, colors.get(default, (255, 255, 255)))


def _font_chain(font_path: str | None) -> list[str]:
    return ([font_path] if font_path else []) + _FALLBACK_FONTS


def _load_font(chain: list[str], size: int, weight: int) -> Any:
    from PIL import ImageFont
    for cand in chain:
        if cand and Path(cand).is_file():
            f = ImageFont.truetype(cand, size)
            try:
                axes = f.get_variation_axes() or []
                wmax = axes[0]["maximum"] if axes else weight
                f.set_variation_by_axes([min(weight, wmax)])  # variable-font weight axis
            except Exception:
                pass
            return f
    return ImageFont.load_default()


def _fit_font_size(max_size: int, min_size: int, step: int, fits: Any) -> int:
    size = max_size
    while size > min_size and not fits(size):
        size -= step
    return max(size, min_size)


def _fit_cover(img: Any, w: int, h: int) -> Any:
    from PIL import Image
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def _darken_band(base: Any, pos: str, w: int, h: int) -> None:
    from PIL import Image
    grad = Image.new("L", (1, h), 0)
    px = grad.load()
    for y in range(h):
        if pos == "bottom":
            t = max(0.0, (y - h * 0.55) / (h * 0.45))
        elif pos == "top":
            t = max(0.0, (h * 0.45 - y) / (h * 0.45))
        else:
            t = max(0.0, 1 - abs(y - h / 2) / (h * 0.35))
        px[0, y] = int(165 * min(1.0, t))
    black = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    black.putalpha(grad.resize((w, h)))
    base.alpha_composite(black)


def _draw_headline(base: Any, hl: dict[str, Any], colors: dict[str, RGB], chain: list[str],
                   weight: int, w: int, h: int) -> None:
    from PIL import ImageDraw
    lines = hl.get("lines") or []
    if not lines:
        return
    pos, box = hl.get("pos", "bottom"), hl.get("box", True)
    max_w, max_h = w - 120, int(h * 0.42)
    draw = ImageDraw.Draw(base)

    def line_w(line: Any, font: Any) -> float:
        return sum(draw.textlength(s["t"] + " ", font=font) for s in line)

    def fits(size: int) -> bool:
        font = _load_font(chain, size, weight)
        if not all(line_w(ln, font) <= max_w for ln in lines):
            return False
        asc, desc = font.getmetrics()
        block = len(lines) * (asc + desc) + (len(lines) - 1) * int(size * 0.18) + 2 * int(size * 0.12)
        return block <= max_h

    size = _fit_font_size(int(hl.get("size", 150)), 36, 4, fits)
    font = _load_font(chain, size, weight)
    asc, desc = font.getmetrics()
    lh, gap = asc + desc, int(size * 0.18)
    pad_x, pad_y = int(size * 0.28), int(size * 0.12)
    block_h = len(lines) * lh + (len(lines) - 1) * gap

    cy = hl.get("y")
    if cy is not None:
        y0 = max(40, min(int(cy * h - block_h / 2), h - block_h - 40))
    elif pos == "top":
        y0 = 150
    elif pos == "center":
        y0 = (h - block_h) // 2
    else:
        y0 = h - block_h - 170

    y = y0
    for ln in lines:
        lw = line_w(ln, font)
        x0 = int((w - lw) // 2)
        if box:
            draw.rounded_rectangle([x0 - pad_x, y - pad_y, x0 + lw + pad_x, y + lh + pad_y],
                                   radius=int(size * 0.18), fill=(10, 10, 10, 235))
        x = x0
        for seg in ln:
            t = seg["t"]
            col = _color(seg.get("c", "white"), colors)
            for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:  # crisp black outline
                draw.text((x + dx, y + dy), t, font=font, fill=(0, 0, 0, 255))
            draw.text((x, y), t, font=font, fill=col + (255,))
            x += draw.textlength(t + " ", font=font)
        y += lh + gap


def _el_xy(e: dict[str, Any], ew: int, eh: int, w: int, h: int) -> tuple[int, int]:
    cx, cy = e.get("x", 0.5) * w, e.get("y", 0.5) * h
    return int(cx - ew / 2), int(cy - eh / 2)


def _draw_coin(base: Any, e: dict[str, Any], colors: dict[str, RGB], chain: list[str],
               weight: int, w: int, h: int) -> None:
    from PIL import Image, ImageDraw
    s = int(e.get("size", 180))
    face = _color(e.get("color", "accent"), colors)
    layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([0, 0, s - 1, s - 1], fill=(10, 10, 10, 255))
    m = int(s * 0.08)
    d.ellipse([m, m, s - 1 - m, s - 1 - m], fill=face + (255,))
    m2 = int(s * 0.16)
    d.ellipse([m2, m2, s - 1 - m2, s - 1 - m2], outline=(10, 10, 10, 220), width=max(2, int(s * 0.04)))
    gf = _load_font(chain, int(s * 0.55), weight)
    glyph = e.get("glyph", "$")
    tw = d.textlength(glyph, font=gf)
    asc, desc = gf.getmetrics()
    d.text(((s - tw) / 2, (s - (asc + desc)) / 2), glyph, font=gf, fill=(10, 10, 10, 255))
    if e.get("rotate"):
        layer = layer.rotate(e["rotate"], expand=True, resample=Image.BICUBIC)
    base.alpha_composite(layer, _el_xy(e, layer.size[0], layer.size[1], w, h))


def _draw_badge(base: Any, e: dict[str, Any], colors: dict[str, RGB], chain: list[str],
                weight: int, w: int, h: int) -> None:
    from PIL import Image, ImageDraw
    t = e.get("t", "")
    size = int(e.get("size", 84))
    bg = _color(e.get("bg", "accent"), colors)
    fg = _color(e.get("fg", "black"), colors)
    f = _load_font(chain, size, weight)
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw = tmp.textlength(t, font=f)
    asc, desc = f.getmetrics()
    pad_x, pad_y = int(size * 0.5), int(size * 0.28)
    bw, bh = int(tw + 2 * pad_x), int(asc + desc + 2 * pad_y)
    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=int(bh * 0.5), fill=bg + (255,))
    d.text((pad_x, pad_y), t, font=f, fill=fg + (255,))
    if e.get("rotate"):
        layer = layer.rotate(e["rotate"], expand=True, resample=Image.BICUBIC)
    base.alpha_composite(layer, _el_xy(e, layer.size[0], layer.size[1], w, h))


def _draw_arrow(base: Any, e: dict[str, Any], colors: dict[str, RGB], w: int, h: int) -> None:
    from PIL import ImageDraw
    d = ImageDraw.Draw(base)
    x1, y1 = e["from"][0] * w, e["from"][1] * h
    x2, y2 = e["to"][0] * w, e["to"][1] * h
    col = _color(e.get("color", "accent"), colors) + (255,)
    wid = int(e.get("width", 24))
    d.line([(x1, y1), (x2, y2)], fill=col, width=wid)
    ang = math.atan2(y2 - y1, x2 - x1)
    head = wid * 2.6
    for da in (math.radians(150), math.radians(-150)):
        d.line([(x2, y2), (x2 + head * math.cos(ang + da), y2 + head * math.sin(ang + da))],
               fill=col, width=wid)


def _place_image(base: Any, e: dict[str, Any], input_paths: dict[str, Path], w: int, h: int) -> None:
    from PIL import Image
    asset = e.get("asset")
    p = input_paths.get(asset) if asset else None
    if p is None or not Path(p).is_file():
        print(f"[cover] element image not delivered: {asset}", file=sys.stderr)
        return
    im = Image.open(p).convert("RGBA")
    iw = int(e.get("width", im.width))
    ih = int(im.height * iw / im.width)
    im = im.resize((iw, ih), Image.LANCZOS)
    if e.get("opacity", 1.0) < 1.0:
        im.putalpha(im.getchannel("A").point(lambda v: int(v * e["opacity"])))
    if e.get("rotate"):
        im = im.rotate(e["rotate"], expand=True, resample=Image.BICUBIC)
    base.alpha_composite(im, _el_xy(e, im.size[0], im.size[1], w, h))


def _place_logo(base: Any, lg: dict[str, Any], input_paths: dict[str, Path], w: int, h: int) -> None:
    from PIL import Image
    asset = lg.get("asset")
    p = input_paths.get(asset) if asset else None
    if p is None or not Path(p).is_file():
        return
    im = Image.open(p).convert("RGBA")
    lw = int(lg.get("width", 200))
    lh = int(im.height * lw / im.width)
    im = im.resize((lw, lh), Image.LANCZOS)
    op = lg.get("opacity", 1.0)
    if op < 1.0:
        im.putalpha(im.getchannel("A").point(lambda v: int(v * op)))
    m = int(lg.get("margin", 44))
    corner = lg.get("corner", "tr")
    x = w - lw - m if corner in ("tr", "br") else m
    y = h - lh - m if corner in ("bl", "br") else m
    base.alpha_composite(im, (x, y))


def compose(base_frame: Path, cover: dict[str, Any], input_paths: dict[str, Path],
            out_png: Path, w: int, h: int) -> None:
    """Draw the cover still (darken band → elements → headline → logo) over the base frame → out_png."""
    from PIL import Image
    colors = {k: (int(v[0]), int(v[1]), int(v[2])) for k, v in (cover.get("colors") or {}).items()}
    chain = _font_chain(str(input_paths[cover["font"]]) if cover.get("font") in input_paths else None)
    weight = int(cover.get("display_weight", 800))
    base = _fit_cover(Image.open(base_frame).convert("RGB"), w, h).convert("RGBA")
    hl = cover.get("headline") or {}
    if hl.get("lines"):
        _darken_band(base, hl.get("pos", "bottom"), w, h)
    for e in cover.get("elements", []):
        kind = e.get("type")
        if kind == "coin":
            _draw_coin(base, e, colors, chain, weight, w, h)
        elif kind == "badge":
            _draw_badge(base, e, colors, chain, weight, w, h)
        elif kind == "arrow":
            _draw_arrow(base, e, colors, w, h)
        elif kind == "image":
            _place_image(base, e, input_paths, w, h)
        else:
            print(f"[cover] unknown element type: {kind}", file=sys.stderr)
    if hl.get("lines"):
        _draw_headline(base, hl, colors, chain, weight, w, h)
    _place_logo(base, cover.get("logo") or {}, input_paths, w, h)
    base.convert("RGB").save(out_png)


def _probe(video: Path) -> tuple[str, str, str]:
    def q(stream: str, entries: str) -> list[str]:
        return subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream,
                               "-show_entries", entries, "-of", "default=nw=1:nk=1", str(video)],
                              capture_output=True, text=True).stdout.strip().splitlines()
    v = q("v:0", "stream=r_frame_rate")
    a = q("a:0", "stream=sample_rate,channels")
    return (v[0] if v else "30"), (a[0] if a else "48000"), (a[1] if len(a) > 1 else "2")


def weld(video: Path, cover_png: Path, out: Path, hold: float, gpu: bool, w: int, h: int) -> None:
    """Hold the cover `hold` s and concat it to the END of the video (one re-encode; gapless, seekable)."""
    fps, sr, ch = _probe(video)
    layout = "stereo" if ch == "2" else "mono"
    bt709 = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"
    venc = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20"] if gpu
            else ["-c:v", "libx264", "-preset", "medium", "-crf", "18"])
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "cover_clip.mp4"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-loop", "1", "-i", str(cover_png),
                        "-f", "lavfi", "-i", f"anullsrc=channel_layout={layout}:sample_rate={sr}",
                        "-t", str(hold), "-r", fps,
                        "-vf", f"scale={w}:{h},setsar=1,format=yuv420p,{bt709}",
                        *venc, "-profile:v", "high", "-c:a", "aac", "-ar", sr, "-ac", ch,
                        "-video_track_timescale", "90000", "-shortest", str(clip)], check=True)
        norm = f"fps={fps},scale={w}:{h},setsar=1,format=yuv420p,{bt709}"
        fc = (f"[0:v]{norm}[v0];[1:v]{norm}[v1];[v0][v1]concat=n=2:v=1:a=0[v];"
              f"[0:a]aresample={sr}[a0];[1:a]aresample={sr}[a1];[a0][a1]concat=n=2:v=0:a=1[a]")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(video), "-i", str(clip), "-filter_complex", fc,
                        "-map", "[v]", "-map", "[a]", *venc, "-c:a", "aac", "-ar", sr,
                        "-movflags", "+faststart", str(out)], check=True)


def render_cover(cover: dict[str, Any], base_video: Path, composed_video: Path,
                 input_paths: dict[str, Path], out: Path, gpu: bool, w: int, h: int,
                 hold: float = 0.6) -> None:
    """Extract the base frame at frame_at (clean plate), compose the still, weld it onto the composited
    video. base_video = clean head_dyn (no captions/logo); composed_video = the finished master."""
    with tempfile.TemporaryDirectory() as td:
        frame = Path(td) / "frame.jpg"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", str(cover["frame_at"]), "-i", str(base_video),
                        "-frames:v", "1", "-q:v", "2", str(frame)], check=True)
        png = Path(td) / "cover.png"
        compose(frame, cover, input_paths, png, w, h)
        weld(composed_video, png, out, hold, gpu, w, h)
