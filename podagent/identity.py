"""WHICH PHYSICAL WORKER this process is — the one fact a pod-retained artifact cannot be read without.

THE NAME MUST BE THE CONTROL PLANE'S, NOT ONE THIS BOX INVENTS. The API fences a targeted delivery by
comparing the envelope's `target_worker_id` against the claims of the bearer that claimed it
(`decoded.TargetWorkerID != claims.JobID` in api/cmd/api/pod_stream.go), and the corr→owner index it writes
carries that same string. So the worker's identity IS its JOB_TOKEN's `jid` claim, and reading it here is the
only way a refusal on this side can name the same worker the brain does. A hostname would be a second
vocabulary: every cross-worker check would either always match or never, and both are silent wrongness.

NO SIGNATURE IS CHECKED AND NONE IS NEEDED. This does not AUTHORISE anything — it answers "who am I", and a
process lying to itself about that gains nothing. The token stays unprinted: only the `jid` claim is ever
returned, never the bearer.

AND THERE IS NO OVERRIDE KNOB. The first draft had a `POD_WORKER_ID` escape hatch; the pod-orphan gate
(`tests/test_knob_registry.py`) refused it, correctly — the box writes the pod's environment WHOLE, so a knob
nothing in the seam assigns can only ever be its default, and a switch nobody can throw is worse than none
here: it would look like a way to repair a mis-routed retention while doing nothing. The bearer already names
this worker, and if it does not, the answer is local-only and says so.
"""
from __future__ import annotations

import base64
import json
import os
import socket
from functools import lru_cache

TOKEN_ENVS = ("JOB_TOKEN", "MONTY_JOB_TOKEN")
UNKNOWN = "unknown-worker"


def _claim_jid(token: str) -> str:
    body = token.rsplit(".", 1)[0]
    try:
        claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception:      # noqa: BLE001 — an unreadable bearer is not an identity, and never an exception
        return ""
    jid = claims.get("jid") if isinstance(claims, dict) else None
    return str(jid).strip() if jid else ""


@lru_cache(maxsize=1)
def worker_identity() -> str:
    """This worker's control-plane name. Cached: it cannot change inside a process's life."""
    for env in TOKEN_ENVS:
        if jid := _claim_jid((os.environ.get(env) or "").strip()):
            return jid
    # A dev/local contour runs the same code with no minted bearer. The hostname is not the control plane's
    # name for this box, so it is deliberately LAST and deliberately MARKED: a retained artifact claimed
    # under it can only ever be read back on the same machine, which is exactly what local means. Read from
    # the kernel, not from `HOSTNAME` — the container runtime sets the hostname, our env seam does not, so
    # an env read here would be a pod knob with no author.
    try:
        host = (socket.gethostname() or "").strip()
    except OSError:
        host = ""
    return f"local:{host}" if host else UNKNOWN


def _reset() -> None:
    """Tests only: re-read the environment after monkeypatching it."""
    worker_identity.cache_clear()
