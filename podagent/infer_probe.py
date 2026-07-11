"""face_probe inference: YuNet face detection over sampled frames of requested shots → raw boxes
(+ optional frame diff), packed as face_probe.schema.json and PUT to the request's presigned URL.
Nothing is chosen here — speaker pick, medians, keyframes stay upstream."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from .cp import download, upload
from .models import FaceProbeParams, FaceProbePayload, ProbeFrame, ProbeShot

_DIFF_WIDTH = 160


class ProbeService:
    """Loads the YuNet detector once (the dominant cost); serves every shot batch after that."""

    def __init__(self, model_path: Path, model_name: str) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.5, 0.3)

    def run(self, params: FaceProbeParams, put_url: str) -> float:
        """Returns wall seconds spent on inference (reported in InferResult.timing)."""
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as td:
            video_path = download(params.video_url, Path(td) / "video")
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.detector.setInputSize((width, height))

            shots = [
                ProbeShot(a=a, b=b, frames=self._probe_shot(cap, fps, a, b, params.stride, params.frame_diff))
                for a, b in ((s[0], s[1]) for s in params.shots)
            ]
            cap.release()

            payload = FaceProbePayload(model=self.model_name, width=width, height=height, shots=shots)
            out = Path(td) / "face_probe.json"
            out.write_text(payload.model_dump_json(exclude_none=True))
            infer_s = time.monotonic() - t0
            upload(out, put_url, "application/json")
        return infer_s

    def _probe_shot(
        self, cap: cv2.VideoCapture, fps: float, a: float, b: float, stride: int, frame_diff: bool
    ) -> list[ProbeFrame]:
        start_idx = round(a * fps)
        end_idx = round(b * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)  # seek once; every later sample is a grab-skip, not a seek

        frames: list[ProbeFrame] = []
        prev_small: np.ndarray | None = None
        for idx in range(start_idx, end_idx, stride):
            ok, frame = cap.read()
            if not ok:
                break
            for _ in range(stride - 1):
                if not cap.grab():
                    break

            _, faces = self.detector.detect(frame)
            if faces is None:
                boxes: list[list[float]] = []
            else:
                boxes = [[float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])] for f in faces]

            diff = None
            if frame_diff:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape[:2]
                small = cv2.resize(gray, (_DIFF_WIDTH, max(1, round(h * _DIFF_WIDTH / w))), interpolation=cv2.INTER_AREA)
                small = small.astype(np.float32)
                diff = 0.0 if prev_small is None else float(np.mean(np.abs(small - prev_small)))
                prev_small = small

            frames.append(ProbeFrame(t=idx / fps, boxes=boxes, diff=diff))
        return frames
