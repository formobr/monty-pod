"""MISC-262: a contour-dry pod needs no GPU, its fakes pass the engine's REAL gates, and a full pod refuses at boot
a card that cannot hold its render ops (main.DRY_POD_NO_GPU_WHY, ops/dry.py DRY_AUDIO_WHY / DRY_STILL_WHY,
infer_lanes.RENDER_HEADROOM_WHY).

The engine gates are VENDORED here as rules (no engine import — this repo ships without it), each citing the
engine source it mirrors, so a dry output is judged by the check it actually meets in a release replay."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from podagent import infer_cliprank
from podagent import main as agent_main
from podagent import render_onepass
from podagent.ops import dry, registry

_NEEDS_FFMPEG = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not on this runner")


@pytest.fixture(autouse=True)
def _isolated_live_mark(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def note(self, ev: dict) -> None:
        self.events.append(ev)

    def send_event(self, ev: dict, *, wait: bool = False) -> bool:
        self.events.append(ev)
        return True

    def announce_ready(self, ev: dict) -> tuple[str, int]:
        self.events.append(dict(ev))
        return ("fake-stream", 1)

    def readiness_wall_s(self) -> float:
        return 0.0

    def await_settled(self, key: tuple[str, int], timeout: float) -> bool:
        return True


# ── vendored engine gates ─────────────────────────────────────────────────────────────────────────────────

# the engine's master check — TARGET_LUFS (brand audio.master_lufs, -14 for the brand), TOL,
# TP_CLIP, SILENT_LUFS, WANT_COLOR are its constants; the audio branch is the MISC-239 «silent audio» refusal.
_TARGET_LUFS, _TOL, _TP_CLIP, _SILENT_LUFS, _WANT_COLOR, _WANT_SR = -14.0, 3.0, 0.0, -40.0, "bt709", 48000


def _loudness(path: Path) -> tuple[float | None, float | None]:
    """Integrated LUFS + true peak, as montyops.measure_master takes them (ffmpeg loudnorm analysis)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
                        "-af", "loudnorm=print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=60)
    doc = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])

    def num(v: str) -> float | None:
        return None if v in ("-inf", "inf", "nan") else float(v)
    return num(doc["input_i"]), num(doc["input_tp"])


def _probe(path: Path, stream: str, entries: str) -> list[str]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", entries,
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=20)
    return r.stdout.strip().split(",") if r.stdout.strip() else []


def _master_contract_issues(path: Path, *, target: float, w: int, h: int, fps: str, dur: float) -> list[str]:
    issues: list[str] = []
    has_audio = bool(_probe(path, "a:0", "stream=sample_rate"))
    lufs, tp = _loudness(path) if has_audio else (None, None)
    if not has_audio:
        issues.append("no audio stream — a final without sound")
    elif lufs is None:
        issues.append("silent audio: loudness measured as -inf — a final without sound")
    elif lufs < _SILENT_LUFS:
        issues.append(f"silent audio: integrated {lufs} LUFS < {_SILENT_LUFS}")
    elif not target - _TOL <= lufs <= target + _TOL:
        issues.append(f"integrated {lufs} LUFS outside {target} ± {_TOL}")
    if tp is not None and tp > _TP_CLIP:
        issues.append(f"true-peak {tp} dBTP > {_TP_CLIP} — clipping")
    space, prim, trc = _probe(path, "v:0", "stream=color_space,color_primaries,color_transfer")
    if not space == prim == trc == _WANT_COLOR:
        issues.append(f"colour untagged: space={space} primaries={prim} transfer={trc}")
    # the delivery grid (check_master `defects`) + the plan's own size/duration the dry master must carry
    vw, vh, rate = _probe(path, "v:0", "stream=width,height,r_frame_rate")
    if (int(vw), int(vh)) != (w, h) or rate != f"{fps}/1":
        issues.append(f"grid drifted: {vw}x{vh}@{rate} != {w}x{h}@{fps}")
    if int(_probe(path, "a:0", "stream=sample_rate")[0] if has_audio else 0) != _WANT_SR:
        issues.append(f"sample rate != {_WANT_SR}")
    got = float(_probe(path, "", "format=duration")[0]) if _probe(path, "", "format=duration") else 0.0
    if abs(got - dur) > 0.1:
        issues.append(f"duration {got} != plan {dur}")
    return issues


def _still_gate_issues(meta: dict, dst: Path, *, vector: bool) -> list[str]:
    """the engine's photo fetcher (finished|plated AND bytes), and its get-photo command
    (a vector not plated is refused), the engine's own mark-backing treatment check (dark, bbox, mark_w/mark_h), and
    the real montyops.media_still.run sidecar shape."""
    issues: list[str] = []
    real_keys = {"width", "height", "vector", "alpha", "plated", "finished", "rasterizer", "width_requested",
                 "dark", "bbox", "mark_w", "mark_h"}
    if set(meta) != real_keys:
        issues.append(f"sidecar keys {sorted(meta)} != the real op's {sorted(real_keys)}")
    if not ((meta.get("finished") or meta.get("plated")) and dst.exists() and dst.stat().st_size):
        issues.append("fetch_still: no finished still at the durable address")
    if vector and not meta.get("plated"):
        issues.append("cmd_get_photo: a VECTOR no host could rasterise")
    if not isinstance(meta.get("dark"), bool):
        issues.append("the engine's mark-backing treatment check: no `dark` verdict")
    bbox = meta.get("bbox") or []
    if len(bbox) != 4 or not all(0.0 <= v <= 1.0 for v in bbox) or not (meta.get("mark_w") and meta.get("mark_h")):
        issues.append(f"the engine's mark-backing rect check: unusable bbox={bbox} mark={meta.get('mark_w')}x{meta.get('mark_h')}")
    if meta.get("vector") is not vector:
        issues.append("probe: vector verdict wrong")
    return issues


def _range_receipt_issues(doc: dict) -> list[str]:
    """the engine op's range transport receipt (closed shape + field bounds) and
    acceptance_snapshot.RangeSampleCounts.strict for one sample."""
    issues: list[str] = []
    if doc.get("status") not in ("ok", "whole_read"):
        issues.append(f"status {doc.get('status')!r} is not green although the strip landed")
    if doc["outputs_present"] != doc["outputs_expected"] or doc["outputs_expected"] <= 0:
        issues.append("outputs_present != outputs_expected")
    if doc["reason"] or not (isinstance(doc["object_bytes"], int) and doc["object_bytes"] > 0):
        issues.append("a green receipt needs object_bytes > 0 and no reason")
    if doc["whole_attempts"] or doc["whole_reads"] or doc["cap_exceeded"] or doc["ignored_range_responses"]:
        issues.append("a green ok receipt attempted or read a whole object")
    if doc["status"] == "ok" and doc["origin_bytes"] + doc["proven_bytes"] >= (doc["object_bytes"] or 0):
        issues.append("an ok range receipt read the whole origin object")
    return issues


def _still_run(tmp_path: Path, name: str, src: Path, plate: str) -> tuple[dict, Path]:
    fn = dry.resolve(registry.get("media.still"))
    out = {"dst": tmp_path / f"{name}.png", "meta": tmp_path / f"{name}.still.json"}
    fn(params={"width": 256, "plate": plate}, inputs={"src": src}, outputs=out)
    return json.loads(out["meta"].read_text()), out["dst"]


# ── the test ──────────────────────────────────────────────────────────────────────────────────────────────

@_NEEDS_FFMPEG
def test_a_dry_pod_boots_without_a_gpu_and_its_fakes_pass_the_engine_gates(monkeypatch, tmp_path):
    # (a) a DRY pod on a host with NO GPU: no NVENC/NVDEC/Vulkan probe, no VRAM floor, and it says so
    monkeypatch.setenv(dry.ARM_ENV, "1")
    monkeypatch.setenv("MONTY_INFER_KINDS", "face_probe")
    free_vram_or_refuse = agent_main._free_vram_or_refuse
    for probe in ("_nvenc_or_refuse", "_nvdec_or_refuse", "_vulkan_preflight", "_free_vram_or_refuse"):
        monkeypatch.setattr(agent_main, probe,
                            lambda *_a, _p=probe, **_k: pytest.fail(f"a dry pod ran {_p}"))
    monkeypatch.setattr(infer_cliprank, "_free_vram_mb", lambda: None)     # no card at all
    monkeypatch.setattr(infer_cliprank, "vram_total_mb", lambda: None)
    real_run = subprocess.run

    def no_gpu_run(cmd, *a, **k):
        argv = " ".join(map(str, cmd))
        assert not re.search(r"nvenc|cuda|vulkan|libplacebo", argv), f"a dry pod touched the GPU: {argv}"
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", no_gpu_run)
    cp = _CP()
    capacity = agent_main.capacity_payload(rank_lanes=1, fetch_workers=1, vram_total_mb=None)
    agent_main._capability_preflight(cp, capacity=capacity)
    boot, ready = cp.events[0], cp.events[-1]
    assert "gpu=not probed (contour-dry" in boot["step"]
    assert ready["phase"] == "ready" and ready["step"] == agent_main.DRY_PREFLIGHT_STEP
    assert "no NVENC/NVDEC/Vulkan probe, no VRAM floor" in ready["step"]
    # not probed is not the `vulkan=false` defect the pool evicts on (the engine's pod boot-defect check)
    assert ready["capacity"]["vulkan"] is None and ready["capacity"]["vulkan"] is not False
    assert ready["capacity"]["gpu_preflight"] == "skipped_contour_dry"
    assert [e for e in cp.events if e.get("status") == "error"] == []
    monkeypatch.setattr(subprocess, "run", real_run)

    # (b1) the dry render_final master passes the master contract at the spec's loudness, size, fps, duration
    spec = SimpleNamespace(overlays=SimpleNamespace(finalize=SimpleNamespace(loudnorm=SimpleNamespace(i=-14.0))))
    lufs = render_onepass.dry_master_lufs(spec)
    assert lufs == -14.0
    assert render_onepass.dry_master_lufs(SimpleNamespace(overlays=None)) == dry.DRY_TONE_LUFS
    master = tmp_path / "master.mp4"
    render_onepass._dry_lavfi(master, w=1080, h=1920, dur=2.0, grid="30", with_audio=True, lufs=lufs)
    assert _master_contract_issues(master, target=lufs, w=1080, h=1920, fps="30", dur=2.0) == []
    # and the bit-exact-silence master it replaces is exactly what the vendored gate refuses
    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=30:d=1",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=1", "-t", "1", "-c:v", "libx264",
                    "-c:a", "aac", str(silent)], check=True, capture_output=True, timeout=60)
    assert any("silent audio" in i for i in _master_contract_issues(silent, target=-14.0, w=64, h=64,
                                                                     fps="30", dur=1.0))
    # a stub video with nothing real to reflect is never silent either
    stub = tmp_path / "stub.mp4"
    dry._write_video(stub, op_name="edit.splice")
    stub_lufs, _ = _loudness(stub)
    assert stub_lufs is not None and stub_lufs > _SILENT_LUFS

    # (b2) media.still: every field its consumers read, for the vector photo lane, the logo lane, a plate
    svg = tmp_path / "fetched.svg"
    dry._write_image(svg, op_name="media.fetch")               # what the dry media.fetch really hands it
    meta, dst = _still_run(tmp_path, "photo_vector", svg, "if_transparent")
    assert _still_gate_issues(meta, dst, vector=True) == []
    assert meta["rasterizer"] == "contour-dry-stub" and meta["dark"] is True   # the placeholder's black fill
    white = tmp_path / "white.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=white:s=80x40",
                    "-frames:v", "1", str(white)], check=True, capture_output=True, timeout=30)
    from PIL import Image
    mark = tmp_path / "logo.png"           # a cut-out wordmark: white, centred, transparent margin
    canvas = Image.new("RGBA", (80, 40), (0, 0, 0, 0))
    canvas.paste((255, 255, 255, 255), (20, 10, 60, 30))
    canvas.save(mark)
    meta, dst = _still_run(tmp_path, "logo", mark, "none")
    assert _still_gate_issues(meta, dst, vector=False) == []
    assert meta["alpha"] is True and meta["dark"] is False and not meta["plated"]
    assert (meta["mark_w"], meta["mark_h"]) == (80, 40) and meta["bbox"] == [0.25, 0.25, 0.75, 0.75]
    meta, dst = _still_run(tmp_path, "plated", white, "always")
    assert _still_gate_issues(meta, dst, vector=False) == []
    # an opaque raster under if_transparent is left alone exactly as the real op leaves it: no dst, says so
    meta, dst = _still_run(tmp_path, "opaque", white, "if_transparent")
    assert not dst.exists() and meta["finished"] is False and "dark" not in meta

    # (b3) media.range_filmstrip: a landed strip is a green receipt the engine's receipt model accepts
    fn = dry.resolve(registry.get("media.range_filmstrip"))
    out = {"strip": tmp_path / "strip.png", "receipt": tmp_path / "receipt.json"}
    fn(params={"url": "https://example.com/c.mp4", "positions": [0.5], "width": 32, "height": 32,
               "fit": "cover", "max_origin_bytes": 1 << 20}, inputs={}, outputs=out)
    assert out["strip"].stat().st_size > 0
    assert _range_receipt_issues(json.loads(out["receipt"].read_text())) == []

    # (c) a FULL pod's floor carries the render headroom, and a 6 GB card that cleared the old floor refuses
    monkeypatch.delenv(dry.ARM_ENV)
    monkeypatch.delenv("MONTY_INFER_KINDS")
    assert agent_main.boot_vram_floor_mib() == 2736.0 + 960.0 + 512.0 == 4208.0
    assert agent_main.boot_vram_floor_mib({"face_probe"}) == 960.0 + 512.0
    monkeypatch.setattr(infer_cliprank, "_free_vram_mb", lambda: 3300.0)    # > old 3248, < 4208
    monkeypatch.setattr(infer_cliprank, "vram_total_mb", lambda: 6144.0)
    full = _CP()
    with pytest.raises(SystemExit) as exc:
        free_vram_or_refuse(full)
    assert exc.value.code == agent_main.BOOT_VRAM_REFUSAL_EXIT
    assert full.events[-1]["step"].startswith("gpu_vram_occupied: free=3300 total=6144 floor=4208")
