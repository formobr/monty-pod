"""podagent.mograph — the overlay filtergraph + public-staging (no node/chrome/ffmpeg execution)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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


# ── H6/F8: a staged input id must not escape the job's own render workspace ────────────────────────

def test_stage_public_regression_floor_matches_final_dispatch_shape(tmp_path: Path) -> None:
    """The exact id shape scripts/final_dispatch.py._ship_public mints (`f"mograph/public/{dest}"` where
    `dest = f"_photo/{p.name}"`, :433,470,484) must still stage exactly where it did before the guard."""
    rd = tmp_path / "remotion"
    (rd / "public").mkdir(parents=True)
    media = tmp_path / "x.jpg"; media.write_bytes(b"jpg")
    mograph._stage_public({"mograph/public/_photo/x.jpg": media}, rd)
    assert (rd / "public" / "_photo" / "x.jpg").read_bytes() == b"jpg"


def test_stage_public_refuses_a_dotdot_traversal(tmp_path: Path) -> None:
    rd = tmp_path / "job" / "remotion"
    (rd / "public").mkdir(parents=True)
    src_dir = tmp_path / "src"; src_dir.mkdir()
    payload = src_dir / "evil.mjs"; payload.write_bytes(b"evil")
    with pytest.raises(mograph.StagedInputNotAllowed):
        mograph._stage_public({"mograph/../../evil.mjs": payload}, rd)
    assert not (rd.parent.parent / "evil.mjs").exists()  # rd.parent.parent == tmp_path — where it would land


def test_stage_public_refuses_an_absolute_looking_id(tmp_path: Path) -> None:
    rd = tmp_path / "remotion"
    (rd / "public").mkdir(parents=True)
    victim = tmp_path / "outside"
    payload = tmp_path / "evil"; payload.write_bytes(b"evil")
    with pytest.raises(mograph.StagedInputNotAllowed):
        mograph._stage_public({f"mograph//{victim}/evil": payload}, rd)
    assert not victim.exists()


def test_stage_public_refuses_a_write_through_the_node_modules_symlink(tmp_path: Path) -> None:
    """bundle.workspace() symlinks `node_modules` into the SHARED, immutable cache (bundle.py:11-19,34,66-69)
    — an id that walks through it must be refused, not silently persisted across every future job."""
    shared_cache = tmp_path / "cache" / "node_modules"
    shared_cache.mkdir(parents=True)
    rd = tmp_path / "remotion"
    rd.mkdir()
    (rd / "node_modules").symlink_to(shared_cache, target_is_directory=True)
    payload = tmp_path / "evil.txt"; payload.write_bytes(b"evil")
    with pytest.raises(mograph.StagedInputNotAllowed):
        mograph._stage_public({"mograph/node_modules/evil.txt": payload}, rd)
    assert not (shared_cache / "evil.txt").exists()


def test_qtrle_layer_encode_sets_bt709(monkeypatch, tmp_path: Path) -> None:
    seqdir = tmp_path / "seq"; seqdir.mkdir()
    (seqdir / "frame.png").write_bytes(b"png")
    commands = []
    monkeypatch.setattr(mograph.subprocess, "run", lambda cmd, **_kw: commands.append(cmd))

    layers = mograph._pack([{"seqdir": seqdir, "start": 0.0, "glass": False}], tmp_path)

    assert layers and len(commands) == 1
    assert all(token in commands[0] for token in mograph._BT709)


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


def test_batch_failure_message_keeps_head_and_tail_of_a_long_dump(tmp_path: Path, monkeypatch) -> None:
    """Node error dumps carry message+stack at the TOP; a tail-only slice (prod ac5b4614) showed none of it."""
    rd = tmp_path / "rd"; rd.mkdir()
    items = _batch_items(tmp_path, ["SplitScreen"], write_frames=False)
    head_line = "Error: A delayRender() ... was called but not cleared after ~28000ms"
    tail_marker = "END-OF-DUMP-MARKER"
    long_stderr = (head_line + "\n" + ("x" * 2000) + "\n" + tail_marker).encode()
    monkeypatch.setattr(mograph.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, b"", long_stderr))

    with pytest.raises(RuntimeError) as exc_info:
        mograph._run_batch(rd, items, tmp_path / "spec.json", None)

    msg = str(exc_info.value)
    assert head_line in msg    # the message line survives, not buried under 2000 chars of stack
    assert tail_marker in msg  # the tail survives too


def test_render_concurrency_follows_the_container_not_the_host(monkeypatch) -> None:
    """A rented pod is a cgroup slice of someone's machine: sizing Chrome tabs off the HOST's cores/RAM is
    how 2 of 3 paid runs died OOM. These are negative tests — the host numbers must LOSE."""
    monkeypatch.setattr(mograph.os, "sched_getaffinity", lambda pid: set(range(28)))
    monkeypatch.setattr(mograph.os, "sysconf",
                        lambda k: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 256 * (1 << 30) // 4096}[k])
    monkeypatch.setattr(mograph, "_cgroup_bytes",
                        lambda paths: (16 << 30) if paths is mograph._CGROUP_MEM_LIMIT else None)
    assert mograph._render_concurrency() == 8      # 16 GiB cgroup limit caps it, not the 256 GiB host

    monkeypatch.setattr(mograph, "_cgroup_bytes", lambda paths: None)
    assert mograph._render_concurrency() == 16     # no limit → host total, hard ceiling 16

    monkeypatch.setattr(mograph.os, "sched_getaffinity", lambda pid: {0, 1})
    assert mograph._render_concurrency() == 2      # cpuset of 2 caps regardless of RAM; never below 2


def test_a_v1_unlimited_sentinel_does_not_beat_the_host_total(monkeypatch) -> None:
    """cgroup v1 'no limit' is a huge number, not 'max' — reading it as the budget re-opens the OOM."""
    monkeypatch.setattr(mograph.os, "sysconf",
                        lambda k: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 32 * (1 << 30) // 4096}[k])
    monkeypatch.setattr(mograph, "_cgroup_bytes",
                        lambda paths: (1 << 62) if paths is mograph._CGROUP_MEM_LIMIT else None)
    assert mograph._effective_ram_bytes() == 32 << 30


def test_cgroup_max_sentinel_is_not_a_number(tmp_path: Path) -> None:
    v2 = tmp_path / "memory.max"; v2.write_text("max\n")
    assert mograph._cgroup_bytes((v2,)) is None                    # "max" = no limit, NOT zero
    v1 = tmp_path / "limit_in_bytes"; v1.write_text("17179869184\n")
    assert mograph._cgroup_bytes((v2, v1)) == 17179869184          # falls through to the readable file


def test_offthread_cache_is_clamped_and_overridable(monkeypatch) -> None:
    monkeypatch.delenv("MONTY_OFFTHREAD_CACHE_MB", raising=False)
    monkeypatch.setattr(mograph, "_effective_ram_bytes", lambda: 64 << 30)
    assert mograph._offthread_cache_bytes() == 2 << 30             # ceiling: ram/8 would be 8 GiB
    monkeypatch.setattr(mograph, "_effective_ram_bytes", lambda: 2 << 30)
    assert mograph._offthread_cache_bytes() == 512 << 20           # floor: ram/8 would be 256 MiB
    monkeypatch.setenv("MONTY_OFFTHREAD_CACHE_MB", "1024")
    assert mograph._offthread_cache_bytes() == 1 << 30             # explicit knob wins


def test_batch_spec_carries_an_explicit_offthread_cache(tmp_path: Path, monkeypatch) -> None:
    """A spec WITHOUT the bound reverts Remotion to host-meminfo cache sizing — the exact OOM this kills."""
    rd = tmp_path / "rd"; rd.mkdir()
    items = _batch_items(tmp_path, ["Compare"])
    monkeypatch.setattr(mograph, "_offthread_cache_bytes", lambda: 1 << 30)
    monkeypatch.setattr(mograph, "_oom_count", lambda: 0)
    monkeypatch.setattr(mograph.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, b"", b""))
    spec = tmp_path / "spec.json"
    mograph._run_batch(rd, items, spec, None)
    assert json.loads(spec.read_text())["offthreadVideoCacheSizeInBytes"] == 1 << 30


def test_a_frameless_failure_carries_the_box_facts_up_front(tmp_path: Path, monkeypatch) -> None:
    """safe_error cuts the wire message at 500 chars; facts buried behind a 3000-byte Chrome tail never
    reached the control plane — that blindness is the diagnostic half of this wave."""
    rd = tmp_path / "rd"; rd.mkdir()
    items = _batch_items(tmp_path, ["SplitScreen", "TitleFX"], write_frames=False)
    samples = iter([3, 5])
    monkeypatch.setattr(mograph, "_oom_count", lambda: next(samples))
    monkeypatch.setattr(mograph, "_render_concurrency", lambda: 6)
    monkeypatch.setattr(mograph, "_offthread_cache_bytes", lambda: 1536 << 20)
    monkeypatch.setattr(mograph.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, b"", b"x" * 5000 + b"crashed"))
    with pytest.raises(RuntimeError) as ei:
        mograph._run_batch(rd, items, tmp_path / "spec.json", None)
    msg = str(ei.value)
    assert "no frames for SplitScreen, TitleFX" in msg
    assert "oom+2" in msg and "tabs=6" in msg and "cache=1536MiB" in msg
    assert msg.index("cache=1536MiB") < 500        # facts must SURVIVE safe_error's 500-char cut


def test_an_unreadable_oom_counter_is_not_reported_as_zero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mograph, "_oom_count", lambda: None)
    facts = mograph._box_facts(None, 4, 512 << 20, tmp_path)
    assert "oom=?" in facts and "oom+" not in facts


def test_render_layers_attaches_the_per_item_bound(tmp_path: Path, monkeypatch) -> None:
    """render_batch.mjs judges EACH item against its own ceiling — every section must carry it, not just
    the first (D38 follow-up: the pod's batch call had no per-item bound at all)."""
    captured = {}
    monkeypatch.setattr(mograph, "remotion_dir", lambda ref, tmp: tmp_path)
    monkeypatch.setattr(mograph, "_run_batch", lambda rd, items, spec, entry: captured.setdefault("items", items))
    monkeypatch.setattr(mograph, "_pack", lambda metas, tmp: metas)
    sec = SimpleNamespace(comp="Stat", start=1.0, props={}, glass=False)
    mograph._render_layers([sec], None, {}, tmp_path, max_seconds=65.0)
    assert captured["items"][0]["maxSeconds"] == 65.0


def test_render_layers_omits_the_bound_when_none_is_reachable(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(mograph, "remotion_dir", lambda ref, tmp: tmp_path)
    monkeypatch.setattr(mograph, "_run_batch", lambda rd, items, spec, entry: captured.setdefault("items", items))
    monkeypatch.setattr(mograph, "_pack", lambda metas, tmp: metas)
    sec = SimpleNamespace(comp="Stat", start=1.0, props={}, glass=False)
    mograph._render_layers([sec], None, {}, tmp_path)
    assert "maxSeconds" not in captured["items"][0]
