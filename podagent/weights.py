"""Model weights as a job INPUT, not image ballast.

The pod is a dumb executor that already receives every input as a presigned URL and holds no keys — a
checkpoint arrives the same way. `InferRequest.weights` carries a presigned GET for a tar of the model
directory plus that tar's sha256; the pod fetches it once, verifies it, extracts it, and hands the local
directory to `from_pretrained`.

Fetch/verify/extract/cache-by-content-hash now live in `artifact.py`, shared with the Remotion bundle.
What stays here is the only weights-specific question: which directory inside the tar `from_pretrained`
should actually be pointed at.
"""
from __future__ import annotations

from pathlib import Path

from . import artifact
from .models import WeightsRef


def cache_root() -> Path:
    return artifact.cache_root("WEIGHTS_CACHE", "/var/cache/monty/weights")


def model_dir(root: Path) -> Path:
    """The directory inside an extracted tar that `from_pretrained` should be pointed at.

    Layout-agnostic on purpose: the seeded tars are HF-hub shaped (`<repo>/snapshots/<rev>/…`) but a flat
    tar of a model directory is just as valid. We locate the one directory holding a config.json rather
    than hard-coding either shape, so re-exporting the weights never silently breaks the pod.
    """
    if (root / "config.json").is_file():
        return root
    hits = sorted(p.parent for p in root.rglob("config.json"))
    # A hub tar carries refs/ + snapshots/<rev>/; deeper nesting means sub-configs, so prefer the shallowest.
    if not hits:
        raise ValueError(f"weights tar holds no config.json under {root}")
    return min(hits, key=lambda p: len(p.relative_to(root).parts))


def ensure(ref: WeightsRef, model_id: str = "") -> Path:
    """Return a local directory holding the model, fetching it only if this exact content is not cached.

    Idempotent and safe to call per job: a warm pod pays the transfer exactly once per checkpoint.
    """
    label = f"weights {model_id or ref.sha256[:12]}"
    return model_dir(artifact.ensure_tree(ref, cache_root(), label))
