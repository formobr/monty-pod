"""clip_rank inference: both SigLIP towers over the request's (intent, images) groups → cosine
relevance + L2-normalized image embeddings, packed as clip_rank.schema.json and PUT to the request's
presigned URL. Nothing is ranked here — the reorder, the relevance floor and the MMR dedup stay upstream.

The cosine runs HERE, inside the same fp16/no_grad block as the towers, so one forward yields both
numbers the planner needs and only the numbers cross back."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .cp import download, upload
from .models import ClipRankGroup, ClipRankGroupResult, ClipRankParams, ClipRankPayload

_MISS = -1.0   # unreadable image, or an embed-only group where a score has nothing to mean
_DP = 4        # cosine error is ~1e-3; 4dp keeps the payload ~¼ the size


class ClipRankService:
    """Loads SigLIP once (the dominant cost); serves every group batch after that."""

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_id = model_id
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

    def run(self, params: ClipRankParams, put_url: str) -> float:
        """Returns wall seconds spent on inference (reported in InferResult.timing)."""
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as td:
            groups = [self._run_group(g, Path(td) / f"g{i}") for i, g in enumerate(params.groups)]
            payload = ClipRankPayload(model=self.model_id, groups=groups)
            out = Path(td) / "clip_rank.json"
            out.write_text(payload.model_dump_json())
            infer_s = time.monotonic() - t0
            upload(out, put_url, "application/json")
        return infer_s

    def _run_group(self, group: ClipRankGroup, workdir: Path) -> ClipRankGroupResult:
        images, ok = self._fetch(group.image_urls, workdir)
        n = len(group.image_urls)
        if not images:
            return ClipRankGroupResult(scores=[_MISS] * n, embeds=[None] * n)

        scores, embeds = self._forward(group.intent, images)
        # re-align onto the REQUESTED order: an image we could not fetch keeps -1.0/None so it sorts last
        by_idx = dict(zip(ok, scores))
        emb_by_idx = dict(zip(ok, embeds))
        return ClipRankGroupResult(
            scores=[by_idx.get(i, _MISS) for i in range(n)],
            embeds=[emb_by_idx.get(i) for i in range(n)],
        )

    def _forward(self, intent: str, images: list) -> tuple[list[float], list[list[float]]]:
        torch = self.torch
        with torch.no_grad():
            iin = self.proc(images=images, return_tensors="pt").to(self.device)
            iin = {k: (v.to(self.dtype) if v.dtype == torch.float32 else v) for k, v in iin.items()}
            ie = torch.nn.functional.normalize(self.model.get_image_features(**iin), dim=-1)
            embeds = [[round(x, _DP) for x in e] for e in ie.float().cpu().tolist()]
            # NO intent = an embed-only caller (the image tower is text-independent). Bailing here would hand it
            # Nones and silently blind the dedup/MMR that asked ONLY for embeddings.
            if not intent:
                return [_MISS] * len(images), embeds
            tin = self.proc(text=[intent], return_tensors="pt", padding="max_length", truncation=True)
            te = torch.nn.functional.normalize(self.model.get_text_features(**tin.to(self.device)), dim=-1)
            sims = (ie @ te.T).squeeze(-1).float().cpu().tolist()
        return [round(s, _DP) for s in sims], embeds

    @staticmethod
    def _fetch(urls: list[str], workdir: Path) -> tuple[list, list[int]]:
        """Download+decode each url; returns the decoded images and their REQUEST indices. A url that will
        not fetch or decode is dropped here and scored -1.0/None by the caller, never raised — one dead tile
        must not fail a whole batch of beats."""
        from PIL import Image

        images, ok = [], []
        for i, url in enumerate(urls):
            try:
                path = download(url, workdir / f"{i}.img")
                images.append(Image.open(path).convert("RGB"))
                ok.append(i)
            except Exception:  # noqa: BLE001 — a broken tile is data, not a fault
                continue
        return images, ok
