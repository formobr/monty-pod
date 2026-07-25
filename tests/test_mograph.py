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


def test_overlay_filtergraph_slides_the_head_below_its_layer() -> None:
    layers = [{"start": 5.0, "dur": 4.0, "glass": False, "head_below": True}]
    fc, last = mograph.overlay_filtergraph(layers)
    assert last == "v0"
    assert "[0:v]split[ha0][hb0]" in fc                       # the base head is split for the slide copy
    assert "trim=start=5.0:end=9.0" in fc                     # the copy is windowed to the beat
    assert "overlay=y='0.33*H*clip(" in fc                    # slides down 0.33*H by the glide progress
    assert "[hbv0][o0]overlay=enable='between(t,5.0,9.0)'" in fc  # the layer's picture then rides over the top


def test_overlay_filtergraph_has_no_head_slide_by_default() -> None:
    # a plain layer must NOT split/slide the base head — only an explicit head_below does
    fc, _ = mograph.overlay_filtergraph([{"start": 1.0, "dur": 2.0, "glass": False}])
    assert "split" not in fc and "0.33*H" not in fc


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


def test_render_concurrency_follows_the_box(monkeypatch) -> None:
    """Fixed at 4, a 28-core host idled while a 2-core one thrashed — and Remotion time IS the final render."""
    from podagent import mograph
    monkeypatch.setattr(mograph.os, "cpu_count", lambda: 28)
    monkeypatch.setattr(mograph.os, "sysconf", lambda k: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 16 * (1 << 30) // 4096}[k])
    assert mograph._render_concurrency() == 8      # 16 GB RAM caps it below the 26 cores allow

    monkeypatch.setattr(mograph.os, "cpu_count", lambda: 2)
    assert mograph._render_concurrency() == 2      # never below 2, never (cores-2)=0

    monkeypatch.setattr(mograph.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(mograph.os, "sysconf", lambda k: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 256 * (1 << 30) // 4096}[k])
    assert mograph._render_concurrency() == 16     # hard ceiling: Chrome stops scaling long before 62
