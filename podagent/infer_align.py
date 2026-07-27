"""align inference: pure wav2vec2 forward over requested windows → log-softmax emission
matrices, packed as one .npz and PUT to the request's presigned URL. Nothing is aligned here —
the emissions are the whole product of this call."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .cp import download, upload
from .models import AlignParams

_SR = 16000


def pack_npz(arrays: dict[str, Any], meta: dict[str, Any]) -> bytes:
    """One .npz: emissions_<i> + meta_json. STORED, not deflated — log-softmax float32 is near-incompressible
    (measured 316.9 MB → 274.8 MB, 13%) and zlib spends ~12s of pod CPU buying it."""
    import numpy as np

    buf = io.BytesIO()
    np.savez(buf, **arrays, meta_json=np.frombuffer(json.dumps(meta).encode(), dtype="uint8"))
    return buf.getvalue()


class AlignService:
    """Loads the checkpoint once (the dominant cost); serves every window batch after that.

    Loads from a LOCAL directory the pod fetched and hash-verified — never from a hub id. `model_id` is
    provenance only (it rides into the payload meta), so a request string can never steer what gets loaded.
    """

    def __init__(self, model_id: str, weights_dir: Path) -> None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.processor = Wav2Vec2Processor.from_pretrained(str(weights_dir), local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            str(weights_dir), local_files_only=True).to(self.device).eval()

    def _emit(self, seg):
        # one window → log-softmax emission. A CUDA runtime error (e.g. an arch this torch build has no kernels
        # for) degrades LOUDLY to CPU once — a rented GPU that can't run is a visible fault, not a silent crawl.
        try:
            logits = self.model(seg.to(self.device)).logits[0]
        except RuntimeError as e:
            if self.device == "cpu":
                raise
            print(f"[podagent] WARNING align CUDA failed ({str(e)[:120]}) → CPU fallback (SLOW)",
                  file=sys.stderr, flush=True)
            self.device = "cpu"
            self.model = self.model.to("cpu")
            logits = self.model(seg.to("cpu")).logits[0]
        return self.torch.log_softmax(logits, dim=-1)

    def _vocab(self) -> list[str]:
        v = self.processor.tokenizer.get_vocab()
        return [tok for tok, _ in sorted(v.items(), key=lambda kv: kv[1])]

    def run(self, params: AlignParams, put_url: str,
            note: Callable[[str], None] | None = None) -> float:
        """Returns wall seconds spent on inference (reported in InferResult.timing)."""
        import numpy as np
        import soundfile as sf
        import torchaudio  # resample only; torchaudio.load on 2.8 dispatches to torchcodec (absent) — decode via soundfile

        say = note or (lambda _m: None)
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as td:
            wav_path = download(params.audio_url, Path(td) / "audio")
            data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # (frames, ch)
            wave = self.torch.from_numpy(data.T.copy())                          # (ch, frames), channels-first
            if wave.shape[0] > 1:
                wave = wave.mean(0, keepdim=True)
            if sr != _SR:
                wave = torchaudio.functional.resample(wave, sr, _SR)

            keep = params.keep_ids
            # index on DEVICE: slicing after .cpu() would still materialise the full-vocab matrix we are here
            # to avoid (~40 MB per 20s window at a 9913-token vocab)
            cols = None if keep is None else self.torch.as_tensor(keep, device=self.device)
            arrays: dict[str, "np.ndarray"] = {}
            with self.torch.inference_mode():
                for i, (a, b) in enumerate((w[0], w[1]) for w in params.windows):
                    seg = wave[:, int(a * _SR): int(b * _SR)]
                    emission = self._emit(seg)
                    if cols is not None:
                        emission = emission.index_select(-1, cols)
                    arrays[f"emissions_{i}"] = emission.cpu().numpy().astype("float32")

            meta: dict[str, Any] = {
                "model": self.model_id,
                "sr": _SR,
                "frame_stride_s": 0.02,
                "vocab": self._vocab(),
            }
            if keep is not None:
                meta["keep_ids"] = list(keep)
            payload = pack_npz(arrays, meta)
            out = Path(td) / "align.npz"
            out.write_bytes(payload)
            infer_s = time.monotonic() - t0
            say(f"align {len(params.windows)} window(s) done in {infer_s:.0f}s, "
                f"uploading {len(payload) / 1e6:.1f} MB")
            upload(out, put_url, "application/octet-stream")
        return infer_s
