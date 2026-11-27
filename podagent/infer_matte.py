"""person_matte inference: RVM (mobilenetv3, fp16 TorchScript) over ONE whole video → the alpha per frame,
written as the outputs the request asks for and PUT to their own presigned URLs; a receipt
(person_matte.schema.json) goes to put_url. Nothing is decided here: the downsample ratio is the planner's,
which layers use the matte and where is the compositor's.

Settings and their sources: the engine's person-matte research receipt (the R4 step-1 bench + RVM @53d74c6:
its README download table, its inference documentation and its speed-test script).

  load    torch.jit.load(<baked asset>, cuda).eval() → torch.jit.freeze   (the image has no torchvision, so not
          the .pth + MattingNetwork route the bench took; parity measured: max |Δα| 4/255)
  run     cudnn.benchmark; src = [1,3,H,W] fp16 0..1; fgr, pha, *rec = model(src, *rec, ratio); rec reset per
          request (one request = one video); α = round(clamp(pha) * 255)
  alpha   rawvideo gray → ffv1 gray Matroska (lossless), frame-for-frame with the input   — the FINAL tier
  cutout  the INPUT's own frames (format=yuva420p, no RGB round trip) alphamerge'd with α → libvpx-vp9
          -b:v 0 -crf 32 -deadline realtime -cpu-used 8 -row-mt 1, WebM, no audio            — the PREVIEW tier
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .cp import download, upload
from .models import MatteOutput, PersonMatteParams, PersonMattePayload, PersonMatteTimings
from .sanitize import safe_error, safe_text

# THE model (owner decision 2026-10-22). The id is what a request must name; the digest pins the baked bytes.
RVM_MODEL_ID = "rvm_mobilenetv3_fp16"
RVM_SHA256 = "847a8b5139498afbf7abde9cc347b41030540f6498db593a0e3d9dde1eccdd96"
RVM_ENV = "MODEL_RVM"
RVM_DEFAULT_PATH = "/opt/models/rvm_mobilenetv3_fp16.torchscript"

# Decode-ahead depth: the decoder thread may run this many frames ahead of the GPU loop. 8 x 1080x1920x3 = 50 MB.
_DECODE_AHEAD = 8

VP9_ARGS = ("-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "32",
            "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1")
FFV1_ARGS = ("-c:v", "ffv1", "-pix_fmt", "gray")


class MatteRefused(ValueError):
    """The request cannot be served as asked (wrong model, unreadable input, short output). Ends the job
    status=error WITH this reason — a partial matte composited over a full video is a visible defect."""


class MatteNet(Protocol):
    """The forward, injectable so the pipeline is testable without a GPU (tests pass a fake)."""

    def reset(self) -> None: ...
    def alpha(self, rgb: np.ndarray, downsample_ratio: float) -> np.ndarray: ...


class RvmTorch:
    """RVM on CUDA. CUDA or nothing: placement is absolute, a CPU matte would be a silent 50x crawl."""

    def __init__(self, path: Path) -> None:
        import torch

        if not torch.cuda.is_available():
            raise MatteRefused("person_matte needs a CUDA device and this pod reports none — refusing, never CPU")
        torch.backends.cudnn.benchmark = True
        self.torch = torch
        self.model = torch.jit.freeze(torch.jit.load(str(path), map_location="cuda").eval())
        self.rec: list[Any] = [None] * 4

    def reset(self) -> None:
        self.rec = [None] * 4

    def alpha(self, rgb: np.ndarray, downsample_ratio: float) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            src = torch.from_numpy(rgb).to("cuda", non_blocking=False).permute(2, 0, 1)[None].half() / 255.0
            _fgr, pha, *self.rec = self.model(src, *self.rec, downsample_ratio)
            return (pha[0, 0].float().clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()


def model_path() -> Path:
    return Path(os.environ.get(RVM_ENV) or RVM_DEFAULT_PATH)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_file(path: Path) -> None:
    """The baked bytes ARE the model: refuse a missing or different file before any CUDA context is made."""
    if not Path(path).is_file():
        raise MatteRefused(f"person_matte model file {path} is not on this pod ({RVM_ENV}); the image bakes it")
    got = file_sha256(path)
    if got != RVM_SHA256:
        raise MatteRefused(f"person_matte model file {path} has sha256 {got[:12]}…, pinned {RVM_SHA256[:12]}…")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: str          # r_frame_rate verbatim, e.g. "30/1" or "30000/1001"


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
    try:
        s = json.loads(out.stdout)["streams"][0]
        info = VideoInfo(int(s["width"]), int(s["height"]), str(s["r_frame_rate"]))
    except (KeyError, IndexError, ValueError) as e:
        raise MatteRefused(f"person_matte input has no readable video stream ({safe_error(e, 120)})") from e
    num, _, den = info.fps.partition("/")
    if info.width < 2 or info.height < 2 or not num.isdigit() or int(num) <= 0 or (den and int(den) <= 0):
        raise MatteRefused(f"person_matte input probe is unusable: {info}")
    if den == "1":
        info = VideoInfo(info.width, info.height, num + "/1")
    return info


def count_frames(path: Path) -> int:
    """Packets in the first video stream of an output we wrote (one packet = one shown frame for ffv1 and for
    libvpx-vp9 WebM, whose hidden alt-ref frames ride inside superframes)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, timeout=300)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return -1


def alpha_argv(info: VideoInfo, out: Path) -> list[str]:
    return ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{info.width}x{info.height}", "-framerate", info.fps, "-i", "-",
            *FFV1_ARGS, "-f", "matroska", str(out)]


def cutout_argv(info: VideoInfo, src: Path, out: Path) -> list[str]:
    # Both streams are re-timed from 0 on the input's own frame rate, so alphamerge pairs frame i with alpha i.
    graph = ("[0:v:0]setpts=PTS-STARTPTS,format=yuva420p[c];"
             "[1:v]setpts=PTS-STARTPTS,format=gray[a];[c][a]alphamerge,format=yuva420p[v]")
    return ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src),
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{info.width}x{info.height}", "-framerate", info.fps,
            "-i", "-", "-filter_complex", graph, "-map", "[v]", "-an", *VP9_ARGS, "-f", "webm", str(out)]


def decode_argv(src: Path) -> list[str]:
    return ["ffmpeg", "-v", "error", "-nostdin", "-i", str(src), "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


@dataclass
class MatteRun:
    infer_s: float
    timings: dict[str, float] = field(default_factory=dict)
    payload: PersonMattePayload | None = None


class _Encoder:
    def __init__(self, name: str, argv: list[str], out: Path, content_type: str) -> None:
        self.name, self.out, self.content_type = name, out, content_type
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)

    def write(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(data)
        except BrokenPipeError as e:
            raise MatteRefused(f"person_matte {self.name} encoder died mid-stream: {self.stderr()}") from e

    def close(self, timeout_s: float) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            rc = self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            self.proc.kill()
            self.proc.wait()
            raise MatteRefused(f"person_matte {self.name} encoder did not finish in {timeout_s:.0f}s") from e
        if rc != 0:
            raise MatteRefused(f"person_matte {self.name} encoder exited {rc}: {self.stderr()}")

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def stderr(self) -> str:
        self.err.seek(0)
        return safe_text(self.err.read().decode(errors="replace").strip()[-400:] or "(no stderr)", 400)


def _frames(src: Path, info: VideoInfo, stop: threading.Event) -> tuple[queue.Queue, threading.Thread, list]:
    """Decoder thread → bounded queue of RGB frames; None marks the end, an exception object a failure."""
    q: queue.Queue = queue.Queue(maxsize=_DECODE_AHEAD)
    size = info.width * info.height * 3
    box: list = []

    def run() -> None:
        err = tempfile.TemporaryFile()
        proc = subprocess.Popen(decode_argv(src), stdout=subprocess.PIPE, stderr=err)
        box.append(proc)
        try:
            assert proc.stdout is not None
            while not stop.is_set():
                buf = proc.stdout.read(size)
                if not buf:
                    break
                if len(buf) != size:
                    raise MatteRefused(f"person_matte decoder delivered a short frame ({len(buf)}/{size} bytes)")
                q.put(np.frombuffer(bytearray(buf), np.uint8).reshape(info.height, info.width, 3))  # writable: torch.from_numpy
            if proc.wait(timeout=60) != 0 and not stop.is_set():
                err.seek(0)
                raise MatteRefused("person_matte decoder failed: "
                                   + safe_text(err.read().decode(errors="replace").strip()[-300:], 300))
            q.put(None)
        except BaseException as e:  # noqa: BLE001 — handed to the consumer, which re-raises it
            q.put(e)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    th = threading.Thread(target=run, name="matte-decode", daemon=True)
    th.start()
    return q, th, box


class MatteService:
    """Holds the frozen RVM on the card (load ~2.3 s + a ~9 s cudnn autotune on the first frame); serves every
    video after that. `net` is injectable: tests pass a fake, the pod builds RvmTorch from the baked file."""

    def __init__(self, model_name: str, path: Path | None = None, *, net: MatteNet | None = None) -> None:
        if model_name != RVM_MODEL_ID:
            raise MatteRefused(f"person_matte serves {RVM_MODEL_ID!r} only (owner decision 2026-10-22); "
                               f"the request named {model_name!r}")
        self.model_name = model_name
        if net is None:
            p = path or model_path()
            verify_model_file(p)
            net = RvmTorch(p)
        self.net = net

    def run(self, params: PersonMatteParams, put_url: str,
            note: Callable[[str], None] = lambda _s: None) -> MatteRun:
        with tempfile.TemporaryDirectory() as td:
            src = download(params.video_url, Path(td) / "src")
            t0 = time.monotonic()
            info = probe(src)
            encoders: list[_Encoder] = []
            stop = threading.Event()
            q, dec_thread, dec_box = _frames(src, info, stop)
            try:
                if params.alpha_put_url is not None:
                    out = Path(td) / "alpha.mkv"
                    encoders.append(_Encoder("alpha", alpha_argv(info, out), out, "video/x-matroska"))
                if params.cutout_put_url is not None:
                    out = Path(td) / "cutout.webm"
                    encoders.append(_Encoder("cutout", cutout_argv(info, src, out), out, "video/webm"))
                self.net.reset()   # one request = one video: the recurrent state starts empty
                n, model_s, first_s = 0, 0.0, 0.0
                while True:
                    item = q.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    s = time.monotonic()
                    a = self.net.alpha(item, params.downsample_ratio)
                    dt = time.monotonic() - s
                    model_s += dt
                    if n == 0:
                        first_s = dt
                    if a.shape != (info.height, info.width) or a.dtype != np.uint8:
                        raise MatteRefused(f"person_matte forward returned {a.shape}/{a.dtype}, "
                                           f"want ({info.height}, {info.width})/uint8")
                    data = np.ascontiguousarray(a).tobytes()
                    for enc in encoders:
                        enc.write(data)
                    n += 1
                    if n % 300 == 0:
                        note(f"person_matte {n} frames")
                if n == 0:
                    raise MatteRefused("person_matte input decoded to zero frames")
                t_drain = time.monotonic()
                for enc in encoders:
                    enc.close(timeout_s=max(120.0, 2.0 * (time.monotonic() - t0)))
                drain_s = time.monotonic() - t_drain
            except BaseException:
                stop.set()
                for enc in encoders:
                    enc.kill()
                for proc in dec_box:
                    if proc.poll() is None:
                        proc.kill()
                raise
            finally:
                # a decoder blocked on a full queue never sees `stop`: drain until it exits (bounded)
                deadline = time.monotonic() + 30.0
                while dec_thread.is_alive() and time.monotonic() < deadline:
                    try:
                        q.get(timeout=0.2)
                    except queue.Empty:
                        pass

            outs: dict[str, MatteOutput] = {}
            for enc in encoders:
                got = count_frames(enc.out)
                if got != n:
                    raise MatteRefused(f"person_matte {enc.name} carries {got} frames, the input decoded {n} — "
                                       f"refusing a matte that does not cover the video frame for frame")
                outs[enc.name] = MatteOutput(bytes=enc.out.stat().st_size, sha256=file_sha256(enc.out))
            wall_s = time.monotonic() - t0
            payload = PersonMattePayload(
                model=self.model_name, model_sha256=RVM_SHA256, width=info.width, height=info.height, fps=info.fps,
                frames=n, downsample_ratio=params.downsample_ratio, alpha=outs.get("alpha"),
                cutout=outs.get("cutout"),
                timings=PersonMatteTimings(model_s=round(model_s, 3), first_frame_s=round(first_s, 3),
                                           encode_drain_s=round(drain_s, 3), wall_s=round(wall_s, 3)))
            for enc in encoders:
                url = params.alpha_put_url if enc.name == "alpha" else params.cutout_put_url
                assert url is not None
                upload(enc.out, url, enc.content_type)
            receipt = Path(td) / "person_matte.json"
            receipt.write_text(payload.model_dump_json(exclude_none=True))
            upload(receipt, put_url, "application/json")
            note(f"person_matte {n} frames {info.width}x{info.height}: model {model_s:.1f}s, wall {wall_s:.1f}s")
            return MatteRun(infer_s=wall_s, payload=payload,
                            timings={"infer_s": wall_s, "model_s": model_s, "first_frame_s": first_s,
                                     "encode_drain_s": drain_s, "frames": float(n),
                                     "fps_model": n / model_s if model_s > 0 else 0.0})
