"""podagent.mograph — the overlay filtergraph + public-staging (no node/chrome/ffmpeg execution)."""
from __future__ import annotations

from pathlib import Path

from podagent import mograph


def test_overlay_filtergraph_shifts_and_gates() -> None:
    layers = [{"start": 5.0, "dur": 4.0, "glass": False},
              {"start": 12.0, "dur": 3.0, "glass": True}]
    fc, last = mograph.overlay_filtergraph(layers)
    assert last == "v1"
    assert "[1:v]setpts=PTS-STARTPTS+5.0/TB[o0]" in fc            # layer shifted to its start
    assert "overlay=enable='between(t,5.0,9.0)':eof_action=pass" in fc  # gated to its window
    assert "[0:v]gblur" not in fc                                 # first (non-glass) layer: no blur
    assert "gblur=sigma=22:enable='between(t,12.0,15.0)'" in fc   # glass layer blurs the frame behind it


def test_stage_public_copies_by_prefix(tmp_path: Path) -> None:
    rd = tmp_path / "remotion"
    (rd / "public").mkdir(parents=True)
    src = tmp_path / "Inter.ttf"; src.write_bytes(b"ttf")
    media = tmp_path / "pic.png"; media.write_bytes(b"png")
    input_paths = {"mograph/public/Inter.ttf": src, "mograph/public/_photo/pic.png": media, "base": tmp_path / "x"}
    mograph._stage_public(input_paths, rd)
    assert (rd / "public" / "Inter.ttf").read_bytes() == b"ttf"
    assert (rd / "public" / "_photo" / "pic.png").read_bytes() == b"png"  # nested rel staged
    assert not (rd / "public" / "x").exists()                             # non-prefixed input NOT staged


class _Sec:
    def __init__(self, comp: str, start: float, glass: bool = False) -> None:
        self.comp, self.start, self.glass, self.props = comp, start, glass, {"prompt": "a thing"}


def _rd(tmp_path: Path, monkeypatch) -> Path:
    rd = tmp_path / "remotion"
    (rd / "src").mkdir(parents=True)
    (rd / "render_batch.mjs").write_text("//")
    monkeypatch.setenv("MONTY_REMOTION_DIR", str(rd))
    return rd


def test_delivered_bespoke_mov_composites_without_chrome(tmp_path: Path, monkeypatch, capsys) -> None:
    """The pod bakes NO bespoke .tsx, so an un-delivered bespoke section was dropped ("no delivered entry") on a
    green manifest. NEGATIVE: remove the delivered-.mov branch → zero layers + a SKIP line."""
    _rd(tmp_path, monkeypatch)
    mov = tmp_path / "Bespoke-c5ac507d.mov"; mov.write_bytes(b"mov")
    monkeypatch.setattr("podagent.render._probe_dur", lambda p: 4.5)
    monkeypatch.setattr(mograph, "_run_batch",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no Chrome for a delivered .mov")))
    layers = mograph._render_layers([_Sec("Bespoke-c5ac507d", 54.1, glass=True)], None,
                                    {"bespoke/Bespoke-c5ac507d.mov": mov}, tmp_path)
    assert layers == [{"mov": str(mov), "start": 54.1, "dur": 4.5, "glass": True}]
    assert "SKIP" not in capsys.readouterr().err


def test_bespoke_without_mov_or_entry_skips_loud(tmp_path: Path, monkeypatch, capsys) -> None:
    _rd(tmp_path, monkeypatch)
    layers = mograph._render_layers([_Sec("Bespoke-deadbeef", 12.0)], None, {}, tmp_path)
    assert layers == []
    assert "SKIP Bespoke-deadbeef" in capsys.readouterr().err
