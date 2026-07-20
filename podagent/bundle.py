"""The Remotion bundle as a job INPUT — the mograph half of "heavy artifacts are inputs, not image ballast".

Same delivery shape as weights (presigned tar + sha256, cached by content hash in `artifact.py`), because
it is the same problem: 500 MB that only SOME jobs need, changing on a cadence the image should not be
chained to. Baking it would put node_modules on every align/probe pod that never renders a frame, and
would rebuild the image every time a comp changes.

WHERE IT DIFFERS FROM WEIGHTS, and why this file exists at all: a checkpoint is READ. A Remotion bundle is
WRITTEN INTO — `mograph._stage_public` copies each job's fonts and section media into `<bundle>/public/`,
and a bespoke batch drops its own entry into `<bundle>/src/`. Handing out the cached directory would let
one job's assets leak into the next job's render, and a `copy2` onto a hardlinked file would mutate the
cached inode itself. So the cache stays IMMUTABLE and every job renders out of its own `workspace()`:

  node_modules  -> SYMLINK to the cached tree   (504 MB, never written into, so sharing is safe)
  src/, *.mjs,  -> real COPY                    (~2 MB — the mutable surface, per job)
  configs
  public/       -> fresh empty dir              (job assets only; nothing inherited)

That is ~2 MB of copying per job against 504 MB, and a job cannot corrupt the cache even in principle.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import artifact
from .models import BundleRef

# Copied per job because staging writes into them. `src/` carries the comps plus the generated
# brandFonts/brandTokens defaults; the configs are what `render_batch.mjs` and Remotion read.
_COPY = ("src", "render_batch.mjs", "package.json", "package-lock.json",
         "tsconfig.json", "remotion.config.ts")
# Shared by symlink: enormous, and nothing stages into it.
_LINK = ("node_modules",)


def cache_root() -> Path:
    return artifact.cache_root("REMOTION_BUNDLE_CACHE", "/var/cache/monty/remotion")


def ensure(ref: BundleRef) -> Path:
    """Fetch+verify+cache this exact bundle, returning its IMMUTABLE root. Never render out of this."""
    root = artifact.ensure_tree(ref, cache_root(), f"remotion bundle {ref.sha256[:12]}")
    # The tar must actually be a Remotion project. Checking here means a mis-built bundle fails at fetch
    # with a clear message, not later as a Node module-resolution error inside a subprocess.
    missing = [n for n in ("render_batch.mjs", "src", "node_modules") if not (root / n).exists()]
    if missing:
        raise RuntimeError(
            f"remotion bundle {ref.sha256[:12]} is not a Remotion project — missing {', '.join(missing)} "
            f"under {root}. Rebuild it with scripts/render/build_remotion_bundle.py.")
    return root


def workspace(root: Path, dest: Path) -> Path:
    """A per-job, writable Remotion project backed by the immutable cached bundle. See the module docstring
    for why this is a copy-and-symlink rather than a plain path handout or a hardlink farm."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in _COPY:
        src = root / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dest / name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / name)
    for name in _LINK:
        link = dest / name
        if not link.exists():
            link.symlink_to(root / name, target_is_directory=True)
    (dest / "public").mkdir(exist_ok=True)
    return dest
