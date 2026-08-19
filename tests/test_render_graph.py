"""Pure unit tests for the render translator — filtergraph and argv only, no ffmpeg execution."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from podagent import finalize, render
from podagent.models import MotionKeyframe, RenderSpec

_EXAMPLES = Path(__file__).resolve().parents[1] / "contracts" / "examples"


def _data(name: str) -> dict:
    return json.loads((_EXAMPLES / name).read_text())


def _spec(name: str) -> RenderSpec:
    return RenderSpec.model_validate(_data(name))


def test_preview_golden_graph() -> None:
    g = render.build_filtergraph(_spec("spec.preview.json"), gpu=True)
    assert "trim=start=12.333:end=18.9" in g
    assert "setpts=(PTS-STARTPTS)/1.5" in g
    assert "atempo=1.5" in g
    assert "concat=n=2:v=1:a=1" in g
    assert g.rstrip().endswith(f"[vout]{render._BT709_SET_PARAMS}[vout]")


@pytest.mark.integration
def test_real_cpu_master_output_is_bt709(tmp_path: Path) -> None:
    """Run the real main graph and inspect encoded stream metadata, not argv intent."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe unavailable")

    source = tmp_path / "synthetic.mp4"
    output = tmp_path / "master.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=16x16:rate=8",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=8000",
        "-t", "4", "-frames:v", "32", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(source),
    ], check=True)
    spec = RenderSpec.model_validate({
        "spec_version": 6, "job_id": "colour-proof", "slug": "colour-proof", "mode": "preview",
        "inputs": [{"id": "source", "kind": "video", "sha256": "0" * 64, "url": "unused"}],
        "timeline": {"fps": 8, "width": 16, "height": 16,
                      "segments": [{"src": "source", "in": 0.0, "out": 4.0, "speed": 1.0}]},
        "encode": {"video": "libx264", "preset": "medium", "cq": 23, "pix_fmt": "yuv420p",
                    "audio": "aac", "audio_bitrate": "96k"},
        "outputs": [{"id": "master", "kind": "master", "put_url": "unused"}],
    })
    subprocess.run(render.build_command(spec, {"source": source}, output, gpu=False), check=True,
                   capture_output=True)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=color_space,color_primaries,color_transfer",
        "-of", "json", str(output),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["color_space"] == "bt709"
    assert stream["color_primaries"] == "bt709"
    assert stream["color_transfer"] == "bt709"


def test_anim_expr_smoothstep_and_constant() -> None:
    kfs = [
        MotionKeyframe(t=0.0, rect=[0.1, 0.05, 0.8, 0.8]),
        MotionKeyframe(t=5.566, rect=[0.14, 0.08, 0.72, 0.72]),
    ]
    e = render.anim_expr(kfs, 0, "ease_in_out", "iw")
    assert "if(" in e
    assert "3-2*" in e  # smoothstep p*p*(3-2p)
    assert "0.1" in e and "0.14" in e  # both rect x values present
    assert e.endswith("*iw")

    one = render.anim_expr(kfs[:1], 0, "ease_in_out", "iw")
    assert "if(" not in one
    assert "0.1" in one


def test_atempo_chain_product() -> None:
    data = _data("spec.preview.json")
    data["timeline"]["segments"][0]["speed"] = 2.5  # past the single-atempo ceiling
    g = render.build_filtergraph(RenderSpec.model_validate(data), gpu=False)
    a0 = next(c for c in g.split(";") if c.endswith("[a0]"))
    factors = [float(x) for x in re.findall(r"atempo=([0-9.]+)", a0)]
    assert len(factors) == 2
    prod = 1.0
    for f in factors:
        prod *= f
    assert abs(prod - 2.5) < 1e-6


def test_gpu_vs_cpu_motion() -> None:
    spec = _spec("spec.preview.json")
    g = render.build_filtergraph(spec, gpu=True)
    assert "libplacebo" in g
    assert "hwupload" in g and "hwdownload" in g

    c = render.build_filtergraph(spec, gpu=False)
    assert "libplacebo" not in c
    # static crop at seg0's first keyframe rect [x=0.1, y=0.05, w=0.8, h=0.8]
    assert "crop=w=iw*0.8:h=ih*0.8:x=iw*0.1:y=ih*0.05" in c


def test_motion_refuses_static_crop_without_explicit_escape(monkeypatch) -> None:
    spec = _spec("spec.preview.json")
    events: list[dict] = []
    monkeypatch.delenv("MONTY_ALLOW_STATIC_CAMERA", raising=False)
    monkeypatch.setattr(render, "_gpu_available", lambda: False)

    class _CP:
        def send_event(self, payload: dict, *, wait: bool = False) -> bool:
            events.append(payload)
            return True

    with pytest.raises(RuntimeError, match="refusing a static-crop master"):
        render.render_spec(spec, _CP())
    assert any(e.get("phase") == "gpu_probe_finished" for e in events)
    assert not any(e.get("op") == "download" for e in events)


def test_motion_deliberate_switch_keeps_the_old_degrade_path(monkeypatch, capsys) -> None:
    spec = _spec("spec.preview.json")
    monkeypatch.delenv("MONTY_ALLOW_STATIC_CAMERA", raising=False)
    monkeypatch.setenv("MONTY_GPU_MOTION", "0")
    monkeypatch.setattr(render, "_gpu_available", lambda: pytest.fail("deliberate switch must skip probe"))

    class _CP:
        def send_event(self, payload: dict, *, wait: bool = False) -> bool:
            return True

        def send_result(self, payload: dict, *, wait: bool = True) -> bool:
            return True

    def fake_download(_url: str, dest: Path) -> Path:
        dest.write_bytes(b"media")
        return dest

    class _Done:
        returncode = 0
        stderr = b""

    monkeypatch.setattr(render, "download", fake_download)
    monkeypatch.setattr(render, "upload", lambda *_a, **_kw: None)
    monkeypatch.setattr(render.subprocess, "run", lambda *_a, **_kw: _Done())
    render.render_spec(spec, _CP())
    assert capsys.readouterr().err.splitlines()[0] == (
        "no NVENC: camera motion degrades to a static crop at the first keyframe"
    )


def test_build_command_encode_flags() -> None:
    spec = _spec("spec.preview.json")
    ipaths = {i.id: Path(f"/work/{i.id}") for i in spec.inputs}
    out = Path("/work/out.mp4")

    cpu = render.build_command(spec, ipaths, out, gpu=False)
    assert "libx264" in cpu and "-crf" in cpu
    assert "-movflags" in cpu and "+faststart" in cpu
    assert all(token in cpu for token in render._BT709)

    gpu = render.build_command(spec, ipaths, out, gpu=True)
    assert "h264_nvenc" in gpu and "-cq" in gpu
    assert "-movflags" in gpu and "+faststart" in gpu
    assert all(token in gpu for token in render._BT709)


def test_build_command_declares_the_measured_grid() -> None:
    """render.build_command is the LIVE multi-pass master encode — a missing -r/-fps_mode/-ar here is
    exactly the bug that shipped a 25fps/96kHz master from a 30fps/48kHz source."""
    spec = _spec("spec.preview.json")
    ipaths = {i.id: Path(f"/work/{i.id}") for i in spec.inputs}
    cmd = render.build_command(spec, ipaths, Path("/work/out.mp4"), gpu=False)
    assert cmd[cmd.index("-r") + 1] == "30"
    assert cmd[cmd.index("-fps_mode") + 1] == "cfr"
    assert cmd[cmd.index("-ar") + 1] == "48000"


def test_build_command_emits_the_exact_rational_not_a_rounded_float() -> None:
    data = _data("spec.preview.json")
    data["timeline"]["fps"] = 30000 / 1001
    spec = RenderSpec.model_validate(data)
    ipaths = {i.id: Path(f"/work/{i.id}") for i in spec.inputs}
    cmd = render.build_command(spec, ipaths, Path("/work/out.mp4"), gpu=False)
    assert cmd[cmd.index("-r") + 1] == "30000/1001"


def test_burn_captions_declares_the_grid_but_never_touches_audio(monkeypatch, tmp_path) -> None:
    """The captions pass RE-ENCODES video (libass burn) but copies audio through — it must declare
    -r/-fps_mode and must NOT declare -ar next to a -c:a copy clause."""
    spec = _spec("spec.final.json")
    mp = spec.overlays.motion_plan
    input_paths = {"caption_font": Path("/work/caption_font")}
    cmds: list[list[str]] = []
    monkeypatch.setattr(render.subprocess, "run", lambda cmd, **_kw: cmds.append(cmd))
    render._burn_captions(mp.captions, mp, Path("/work/src.mp4"), input_paths, tmp_path / "out.mp4",
                          False, spec.timeline.width, spec.timeline.height, spec.encode,
                          spec.timeline.fps)
    cmd = cmds[0]
    assert cmd[cmd.index("-r") + 1] == "30"
    assert cmd[cmd.index("-fps_mode") + 1] == "cfr"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-ar" not in cmd


def test_final_overlays_not_implemented() -> None:
    spec = _spec("spec.final.json")

    class _StubCP:
        def send_event(self, payload: dict, *, wait: bool = False) -> bool:
            raise AssertionError("send_event must not be reached before the render runs")

    with pytest.raises(NotImplementedError):
        render.render_spec(spec, _StubCP())  # type: ignore[arg-type]


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf"), None])
def test_render_spec_refuses_an_unusable_grid_before_any_subprocess(fps, monkeypatch) -> None:
    """The ONLY refusal a lost render is worse than — it must fire before gpu_probe's own subprocess."""
    spec = _spec("spec.preview.json")
    spec.timeline.fps = fps
    monkeypatch.setattr(render.subprocess, "run", lambda *_a, **_kw: pytest.fail("subprocess ran"))

    class _StubCP:
        def send_event(self, payload: dict, *, wait: bool = False) -> bool:
            raise AssertionError("send_event must not be reached before the grid is refused")

    with pytest.raises(ValueError):
        render.render_spec(spec, _StubCP())  # type: ignore[arg-type]


def test_input_ids_excludes_non_av_assets() -> None:
    # a caption/cover font is downloaded but must NOT be fed as ffmpeg -i (a TTF is not a decodable stream)
    spec = _spec("spec.final.json")
    ids = render.input_ids(spec)
    assert "caption_font" not in ids
    assert spec.timeline.segments[0].src in ids                 # the base video IS consumed
    assert spec.overlays.broll_final.broll[0].clip in ids       # broll clips ARE consumed
    ipaths = {i.id: Path(f"/work/{i.id}") for i in spec.inputs}
    cmd = render.build_command(spec, ipaths, Path("/work/out.mp4"), gpu=False)
    assert "/work/caption_font" not in [cmd[n + 1] for n, a in enumerate(cmd) if a == "-i"]


def test_multi_source_input_order() -> None:
    data = _data("spec.preview.json")
    src2 = dict(data["inputs"][0])
    src2["id"] = "src2"
    data["inputs"].append(src2)
    data["timeline"]["segments"][1]["src"] = "src2"  # seg1 now reads the second input
    spec = RenderSpec.model_validate(data)

    assert render.input_ids(spec) == ["src", "src2"]
    g = render.build_filtergraph(spec, gpu=False)
    assert "[0:v]" in g and "[1:v]" in g

    ipaths = {i.id: Path(f"/work/{i.id}") for i in spec.inputs}
    cmd = render.build_command(spec, ipaths, Path("/work/out.mp4"), gpu=False)
    i_paths = [cmd[n + 1] for n, a in enumerate(cmd) if a == "-i"]
    assert i_paths == ["/work/src", "/work/src2"]


def test_render_download_ffmpeg_upload_boundaries_are_structured(monkeypatch, tmp_path) -> None:
    spec = _spec("spec.preview.json")
    monkeypatch.setenv("MONTY_ALLOW_STATIC_CAMERA", "1")
    events: list[dict] = []
    results: list[dict] = []

    class _CP:
        def send_event(self, payload: dict, *, wait: bool = False) -> bool:
            events.append(payload)
            return True

        def send_result(self, payload: dict, *, wait: bool = True) -> bool:
            results.append(payload)
            return True

    def fake_download(_url: str, dest: Path) -> Path:
        dest.write_bytes(b"media")
        return dest

    class _Done:
        returncode = 0
        stderr = b""

    monkeypatch.setattr(render, "download", fake_download)
    monkeypatch.setattr(render, "upload", lambda *_a, **_kw: None)
    monkeypatch.setattr(render, "_gpu_available", lambda: False)
    monkeypatch.setattr(render.subprocess, "run", lambda *_a, **_kw: _Done())
    render.render_spec(spec, _CP(), corr_id="c", session_id="s")

    phases = [e for e in events if e.get("op") in {"gpu_probe", "download", "ffmpeg", "upload"}]
    assert [e["phase"] for e in phases] == [
        "gpu_probe_started", "gpu_probe_finished",
        "download_started", "download_finished",
        "ffmpeg_started", "ffmpeg_finished",
        "upload_started", "upload_finished",
    ]
    assert all(e["corr_id"] == "c" and e["session_id"] == "s" for e in phases)
    assert results and results[0]["session_id"] == "s"


# --- the post-render grid verdict: loud, never a reason to withhold the PUT ---

class _Done:
    returncode = 0
    stderr = b""


def _wire_render_spec(monkeypatch, puts: list) -> None:
    def fake_download(_url: str, dest: Path) -> Path:
        dest.write_bytes(b"media")
        return dest
    monkeypatch.setattr(render, "download", fake_download)
    monkeypatch.setattr(render, "upload", lambda path, url, _mime: puts.append((path, url)))
    monkeypatch.setattr(render, "_gpu_available", lambda: False)
    monkeypatch.setattr(render.subprocess, "run", lambda *_a, **_kw: _Done())
    monkeypatch.setenv("MONTY_ALLOW_STATIC_CAMERA", "1")


def test_render_spec_reports_a_grid_mismatch_but_still_puts_the_master(monkeypatch) -> None:
    """NEGATIVE: the verdict is loud but must never withhold an already-paid-for master."""
    spec = _spec("spec.preview.json")
    events: list[dict] = []
    results: list[dict] = []
    puts: list = []

    class _CP:
        def send_event(self, payload: dict, *, wait: bool = False) -> bool:
            events.append(payload)
            return True

        def send_result(self, payload: dict, *, wait: bool = True) -> bool:
            results.append(payload)
            return True

    _wire_render_spec(monkeypatch, puts)
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 25.0, 25, 1, 4.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: False)

    render.render_spec(spec, _CP())

    grid_events = [e for e in events if e.get("op") == "grid_verify"]
    assert len(grid_events) == 1
    ev = grid_events[0]
    assert ev["status"] == "step" and ev["outcome"] == "ok"
    assert ev["phase"] == "grid_verify_degraded"
    assert ev["timings"]["grid_defect"] == {"video_rate": {"declared": "30", "measured": "25"}}
    assert puts, "the master PUT must still happen despite the mismatch"
    assert results and results[0]["status"] == "ok"
    assert results[0]["defects"] == [{"video_rate": {"declared": "30", "measured": "25"}}]


def test_render_spec_never_reports_a_delivered_master_as_a_failed_job(monkeypatch) -> None:
    """The cabinet frontend keys 'failed' off outcome=='error' alone (progress.mjs isErrorActivity),
    not off the phase name — a degraded-but-delivered master must never set outcome/error/error_type."""
    spec = _spec("spec.preview.json")
    events: list[dict] = []
    puts: list = []

    class _CP:
        def send_event(self, payload: dict, *, wait: bool = False) -> bool:
            events.append(payload)
            return True

        def send_result(self, payload: dict, *, wait: bool = True) -> bool:
            return True

    _wire_render_spec(monkeypatch, puts)
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 25.0, 25, 1, 4.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: False)

    render.render_spec(spec, _CP())

    assert not any(e.get("outcome") == "error" for e in events)
    assert not any("error" in e or "error_type" in e for e in events if e.get("op") == "grid_verify")


def test_render_spec_emits_no_event_on_a_matching_grid(monkeypatch) -> None:
    spec = _spec("spec.preview.json")
    events: list[dict] = []
    results: list[dict] = []
    puts: list = []

    class _CP:
        def send_event(self, payload: dict, *, wait: bool = False) -> bool:
            events.append(payload)
            return True

        def send_result(self, payload: dict, *, wait: bool = True) -> bool:
            results.append(payload)
            return True

    _wire_render_spec(monkeypatch, puts)
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, 4.0))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: True)
    monkeypatch.setattr(finalize, "_probe_audio", lambda _p: (48000, 4.0))

    render.render_spec(spec, _CP())

    assert not any(e.get("op") == "grid_verify" for e in events)
    assert "defects" not in results[0]


def test_render_spec_survives_a_post_render_probe_failure(monkeypatch) -> None:
    """NEGATIVE: a probe raising on the FINISHED master must not cost the render its PUT."""
    spec = _spec("spec.preview.json")
    puts: list = []

    class _CP:
        def send_event(self, payload: dict, *, wait: bool = False) -> bool:
            return True

        def send_result(self, payload: dict, *, wait: bool = True) -> bool:
            return True

    def boom(_p):
        raise RuntimeError("ffprobe exploded")

    _wire_render_spec(monkeypatch, puts)
    monkeypatch.setattr(finalize, "_probe", boom)

    render.render_spec(spec, _CP())  # must not raise
    assert puts
