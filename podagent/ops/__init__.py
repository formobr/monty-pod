"""The operation seam — public executor skeleton.

`registry` reads the declarations, `pack` fetches the private handlers by presigned URL, `runner`
executes a chain on one box with local-disk handoff. Nothing here knows how any operation actually works.
"""
from . import pack, registry, runner

__all__ = ["pack", "registry", "runner"]
