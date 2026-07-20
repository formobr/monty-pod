"""The ops pack: the third tenant of the content-addressed artifact cache.

Weights were the first (`weights.py`), the Remotion bundle the second (`bundle.py`), and this is the
same class of thing — a CI-built tar that arrives by presigned URL, is verified by sha256, unpacked once
and reused while the pod is warm. `artifact.ensure_tree` does all of that; this module only adds what is
genuinely its own: putting the extracted tree on `sys.path` and resolving `module:callable`.

WHY THIS EXISTS. `pod-agent` is a PUBLIC repo and stays public, so the cut algorithm, the prompts and the
plan decisions must never appear in it. But the handlers ARE the tuned implementations — filtergraph
construction, encoder profiles, camera-expression building — and shipping those publicly would publish the
algorithm. The pack splits it: the public repo keeps the executor skeleton (fetch, verify, dispatch,
transport), the pack carries the implementations.

WHY NO KEY IS NEEDED. The obvious alternative — make the image private and pull it with registry auth —
puts a CREDENTIAL ON THE POD and breaks the keyless invariant outright. A presigned URL is a
capability, not a credential: it is scoped to one object, expires, and is handed over in the job envelope
the pod already receives. So the pod's entire credential surface stays exactly CP_URL + JOB_TOKEN, and
private code still reaches it. That is the same reason weights and the bundle travel this way.
"""
from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from ..artifact import cache_root, ensure_tree, log

PACK_CACHE_ENV = "OPS_PACK_CACHE"
PACK_CACHE_DEFAULT = "/var/cache/monty/ops"

# A pack is imported ONCE per process: importlib caches modules by name, so two different packs in one
# process would silently serve whichever loaded first. The lock makes the check-and-import atomic; the
# recorded digest makes a second, DIFFERENT pack a loud error instead of a wrong answer.
_lock = threading.Lock()
_loaded_sha: str | None = None


class PackError(RuntimeError):
    pass


def ensure(ref: Any) -> Path:
    """Fetch+verify+unpack this exact pack (or hit the warm cache) and return its root."""
    return ensure_tree(ref, cache_root(PACK_CACHE_ENV, PACK_CACHE_DEFAULT), label="ops-pack")


def activate(ref: Any) -> Path:
    """Make the pack's handlers importable, idempotently.

    Re-activating the SAME pack is free — that is the warm-pod case and it must stay free. Activating a
    DIFFERENT pack in a process that already imported one raises: Python would otherwise keep serving the
    first pack's modules from sys.modules and the job would run the wrong handlers while reporting
    success, which is precisely the silent-wrong-answer failure the content-addressed cache exists to
    make unrepresentable.
    """
    global _loaded_sha
    with _lock:
        root = ensure(ref)
        if _loaded_sha == ref.sha256:
            return root
        if _loaded_sha is not None:
            raise PackError(
                f"this process already activated ops-pack {_loaded_sha[:12]} and cannot switch to "
                f"{ref.sha256[:12]} — imported handler modules would be served from the FIRST pack. "
                f"Restart the agent, or keep one pack per pod.")
        p = str(root)
        if p not in sys.path:
            sys.path.insert(0, p)
        _loaded_sha = ref.sha256
        log(f"ops-pack {ref.sha256[:12]} on sys.path → {root}")
        return root


def resolve(handler: str) -> Callable[..., Any]:
    """`module:callable` → the callable, from the activated pack.

    This is THE reason `podagent/main.py` needs no `if/elif` branch per op: the registry names the
    handler, the pack provides it, and dispatch is a lookup. A new op adds no code here.
    """
    mod_name, _, fn_name = handler.partition(":")
    if not mod_name or not fn_name:
        raise PackError(f"handler {handler!r} is not `module:callable`")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise PackError(
            f"handler module {mod_name!r} is not in the ops pack ({e}). Either the pack predates this op "
            f"or the pack was built without it — check scripts/build_ops_pack.py output.") from e
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise PackError(f"{mod_name}:{fn_name} is not callable")
    return fn


def reset_for_tests() -> None:
    """Drop the one-pack-per-process latch. Tests activate several fixture packs in one interpreter; prod
    never calls this."""
    global _loaded_sha
    with _lock:
        _loaded_sha = None
