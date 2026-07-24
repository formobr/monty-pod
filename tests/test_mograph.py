"""podagent.mograph — the overlay filtergraph + public-staging (no node/chrome/ffmpeg execution)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

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


def _batch_items(tmp_path: Path, comps, *, write_frames=True):
    items = []
    for i, c in enumerate(comps):
        sd = tmp_path / f"seq{i}"; sd.mkdir()
        if write_frames:
            (sd / "element-0001.png").write_bytes(b"png")
        items.append({"comp": c, "props": {}, "seqdir": str(sd)})
    return items


def test_batch_that_aborts_after_writing_every_sequence_is_kept(tmp_path: Path, monkeypatch, capsys) -> None:
    """Chrome SIGABRTs on teardown with all frames already on disk; a returncode check threw the master away."""
    rd = tmp_path / "rd"; rd.mkdir()
    items = _batch_items(tmp_path, ["Compare", "RingStat"])
    monkeypatch.setattr(mograph.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], -6, b"", b"Aborted"))
    mograph._run_batch(rd, items, tmp_path / "spec.json", None)     # must NOT raise
    assert "AFTER writing every sequence" in capsys.readouterr().err


def test_batch_with_a_missing_sequence_still_fails_loud(tmp_path: Path, monkeypatch) -> None:
    rd = tmp_path / "rd"; rd.mkdir()
    items = _batch_items(tmp_path, ["Compare", "Bars"], write_frames=False)
    monkeypatch.setattr(mograph.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], -6, b"", b"boom"))
    with pytest.raises(RuntimeError, match="no frames for Compare, Bars"):
        mograph._run_batch(rd, items, tmp_path / "spec.json", None)
