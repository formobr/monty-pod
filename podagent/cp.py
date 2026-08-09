"""Control-plane client. The pod dials OUT only: poll a job, report events/results, move bytes
via presigned URLs. Auth = the single job token from the environment; no other credentials exist
on this machine."""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse
from urllib.request import url2pathname

import requests
from requests.adapters import HTTPAdapter

from podagent import event_stream
from podagent.sanitize import safe_error, safe_text

_TIMEOUT = 30

# How long ONE claim waits on the live socket before the dispatch loop comes round again. It is not a poll
# interval — nothing is spent while it waits — so it is sized to keep the loop responsive to shutdown, not to
# ration round trips the way `_MIN_POLL_INTERVAL_S` had to.
_CLAIM_WALL_S = 30.0
_CHUNK = 1 << 20
_XFER_ATTEMPTS = 3

# `timeout=` is PER-READ: an origin that drips one byte inside every window never trips it, so a stalled
# transfer's only real bound is the job lease — of RENTED time. This is the wall-clock ceiling on ONE
# transfer attempt. Arithmetic: the largest legitimate object is a ~1 GB master; at the ~25 MB/s we measure
# that is 40 s, and at a 4 MB/s floor (6x worse than observed, still a working link) it is ~256 s. 300 s is
# a generous round-up over that floor, leaves room for all 3 attempts inside a render lease, and cuts a
# dripping origin off well before the ops timeout would.
_XFER_DEADLINE_S = 300.0

# The object store is a DIFFERENT host with DIFFERENT auth (each presigned url carries its own signature),
# so it gets its own keep-alive session — the CP's Bearer token must never travel to a third party. No
# urllib3 Retry on it: the retry/resume policy below is body-aware, urllib3's is not.
STORE_POOL_WHY = """
A CONNECTION POOL SMALLER THAN THE CONCURRENCY THAT USES IT IS NOT A POOL — IT IS A CHURN ENGINE.

`pool_maxsize` was 4 while the runner ran up to `OPS_MAX_PARALLEL` (16 on the rented box) steps at once, and
after the transport split up to `OPS_MAX_TRANSFERS` (64) binds and puts. urllib3 does not WAIT for a free
slot when `pool_block` is false — it mints a connection, uses it, and throws it away. So 12 of every 16 puts,
and later 60 of every 64, paid a fresh TCP+TLS handshake and left a socket in TIME_WAIT.

WHY THAT IS NOT MERELY WASTE. Measured 2026-08-06: 595 media steps in one run, 391 of them putting an object.
On a RENTED container that is ~400 short-lived connections per run against one host, and the failure it
produces is the one we could not explain — the box fails a busy connection FAST (`ConnectionError` in 0.4 s,
a RST), while the pod goes SILENT for 45 s. Silence means the SYN got no answer at all: packets DROPPED, not
refused, which is what a full conntrack/NAT table does. The same urls answer in 0.1-0.6 s when asked alone.

SO THE POOL IS SIZED BY THE SAME NUMBER THE BOX ASSIGNS FOR TRANSFERS. One source of truth
(`broker.pod_lane.OPS_MAX_TRANSFERS_VAR`), so widening the transport can never again silently outgrow the
pool that serves it. `pool_block` stays FALSE — a step waiting on a connection slot would be an undeclared
wait — and it now has nothing to fall back to, because the pool is as wide as the callers.

NOT PROVEN TO BE THE WHOLE DISEASE. The hang is consistent with conntrack exhaustion and this removes the
churn we ourselves create; whether the rented network drops for other reasons as well is a question the
`connect` leg answers on the next run, now that it fails in 10 s instead of going quiet for 45.
"""


def _store_pool() -> int:
    """As wide as the transfers this pod was told it may run (STORE_POOL_WHY)."""
    raw = (os.environ.get("OPS_MAX_TRANSFERS") or "").strip()
    try:
        return max(4, int(raw))
    except ValueError:
        return 16      # the step cap this image ships with; never the old 4


_store = requests.Session()
for _scheme in ("http://", "https://"):
    _store.mount(_scheme, HTTPAdapter(pool_connections=_store_pool(), pool_maxsize=_store_pool(),
                                      pool_block=False))
def _log(msg: str) -> None:
    print(f"[podagent] {safe_text(msg)}", file=sys.stderr, flush=True)


def _file_path(url: str) -> str | None:
    """A file:// url → its local path, else None. Lets the SAME render_spec run on a keyless pod
    (presigned http/https) and on the origin/laptop (local files, no R2, no CP) — the local backend
    hands the pod file:// urls, so download/upload degrade to a copy."""
    if not url.startswith("file:"):
        return None
    return url2pathname(urlparse(url).path)


class ControlPlane:
    def __init__(self, base_url: str, job_token: str) -> None:
        self.base = base_url.rstrip("/")
        # One typed socket carries jobs, progress and terminal results. EventStream persists every client frame
        # before sending, so this layer never invents an HTTP fallback or a second retry policy.
        self._stream = event_stream.EventStream(self.base, job_token)

    def poll_job(self) -> dict[str, Any] | None:
        """One envelope of work, or None if none arrived inside the wall.

        NOT a poll any more, despite the name the call sites still use: the control plane claims from the
        session lane and pushes the envelope down the socket this pod already holds. `GET /pod/job` is
        deleted. The name stays so the dispatch loop reads the same; what it does is wait on a live wire.
        """
        return self._stream.claim(_CLAIM_WALL_S)

    @staticmethod
    def _stamped(payload: dict[str, Any]) -> dict[str, Any]:
        stamped = dict(payload)
        stamped.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return stamped

    def send_event(self, payload: dict[str, Any], *, wait: bool = False) -> bool:
        """Durably enqueue a structured event; optionally wait for its typed ACK."""
        return self._stream.send_event(self._stamped(payload), wait=wait)

    def send_result(self, payload: dict[str, Any], *, wait: bool = True) -> bool:
        """Durably deliver one correlated result; EventStream persists its result_acked transition."""
        accepted = self._stream.send_result(dict(payload), wait=wait)
        if wait and not accepted:
            raise event_stream.DeliveryPending(
                "result ACK remains ambiguous; durable outbox retains it for replay")
        return accepted

    def close_stream(self) -> None:
        """Say out loud what was never acknowledged, and let the socket go."""
        self._stream.close()

    def note(self, payload: dict[str, Any]) -> None:
        """Append progress durably; persistence failure is fatal to admission and propagates."""
        self.send_event(payload, wait=False)

    def set_bootstrap_event(self, payload: dict[str, Any]) -> None:
        """Replay a worker admission event after each WebSocket reconnect."""
        self._stream.set_bootstrap_event(self._stamped(payload))

    def report_infer_result(self, payload: dict[str, Any]) -> bool:
        """Deliver the real InferResult on the typed result frame. No result→event downgrade exists."""
        return self.send_result(dict(payload))


class UrlNotAllowed(ValueError):
    """A url the transport may not dereference — it is neither local nor a capability we minted."""


# A presigned GET carries its signature in the query, whichever object store minted it (SigV4 prefixes,
# plus SigV2's bare pair). This is what tells "our own R2" from "somewhere on the internet".
_SIGNED_PREFIXES = ("x-amz-", "x-goog-", "x-ms-", "x-obs-")
_SIGNED_NAMES = frozenset({"signature", "awsaccesskeyid", "expires", "token"})


def _is_presigned(url: str) -> bool:
    names = {n.lower() for n, _ in parse_qsl(urlparse(url).query, keep_blank_values=True)}
    return any(n.startswith(_SIGNED_PREFIXES) or n in _SIGNED_NAMES for n in names)


def assert_fetchable(url: str) -> str:
    """THE transport allowlist. A presigned url is a CAPABILITY — one object, expiring, minted by the
    control plane; a bare url is an address the pod was merely told to visit, and this pod is RENTED on
    somebody else's network where `http://169.254.169.254/...` is one string away."""
    if _file_path(url) is not None or _is_presigned(url):
        return url
    parts = urlparse(url)
    raise UrlNotAllowed(
        f"REFUSED to fetch {parts.scheme}://{parts.hostname or ''}{parts.path}: it is neither a file:// "
        f"path nor a PRESIGNED url. The pod dereferences capabilities the control plane minted, not "
        f"addresses it was handed. An ORIGIN url (a stock CDN) belongs in an op's params, where the ops "
        f"pack runs its own fetch guard on it (montyops.stock_hosts) — not in a binding, where this "
        f"transport would fetch whatever a third party's search response happened to contain.")


class TransferTimeout(requests.RequestException):
    """A transfer attempt outlived its wall-clock deadline. A RequestException so the retry loops around it
    treat a stalled origin exactly like a dropped one."""


def _pump(resp: Any, fh: Any, already: int) -> int:
    """Stream a response body to an open file under the wall-clock deadline. Returns bytes written."""
    t0 = time.monotonic()
    moved = 0
    for chunk in resp.iter_content(_CHUNK):
        if not chunk:
            continue
        fh.write(chunk)
        moved += len(chunk)
        elapsed = time.monotonic() - t0
        if elapsed > _XFER_DEADLINE_S:
            raise TransferTimeout(
                f"download stalled: aborted after {elapsed:.1f}s (deadline {_XFER_DEADLINE_S:.0f}s) with "
                f"{moved} bytes moved this attempt, {already + moved} on disk")
    return moved


class _DeadlineBody:
    """A file read as chunks under the same wall-clock deadline. `__len__` keeps requests on Content-Length —
    a presigned PUT is signed for a plain body, not for chunked transfer-encoding."""

    def __init__(self, fh: Any, size: int) -> None:
        self._fh, self._size = fh, size

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Any:
        t0 = time.monotonic()
        sent = 0
        while True:
            chunk = self._fh.read(_CHUNK)
            if not chunk:
                return
            yield chunk
            sent += len(chunk)
            elapsed = time.monotonic() - t0
            if elapsed > _XFER_DEADLINE_S:
                raise TransferTimeout(
                    f"upload stalled: aborted after {elapsed:.1f}s (deadline {_XFER_DEADLINE_S:.0f}s) with "
                    f"{sent} of {self._size} bytes sent")


def download(url: str, dest: Path) -> Path:
    """Presigned GET → file, streamed, 3 attempts, RESUMING via Range. A file:// url copies from local disk.

    A fetch that raises unwinds through the whole step chain and discards every sibling step's finished work,
    so a dropped connection must cost the remaining bytes, never the job."""
    assert_fetchable(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    local = _file_path(url)
    if local is not None:
        shutil.copyfile(local, dest)
        return dest
    for attempt in range(_XFER_ATTEMPTS):
        # attempt 0 always truncates: only bytes THIS call wrote are known to belong to this object.
        have = dest.stat().st_size if attempt and dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with _store.get(url, stream=True, timeout=_TIMEOUT, headers=headers) as r:
                r.raise_for_status()
                resumed = bool(have) and r.status_code == 206
                if have and not resumed:
                    _log(f"download: server ignored Range (status {r.status_code}) — restarting from 0, "
                         f"{have} bytes discarded")
                    have = 0
                with dest.open("ab" if resumed else "wb") as f:
                    _pump(r, f, have)
            return dest
        except requests.RequestException as e:
            if attempt + 1 == _XFER_ATTEMPTS:
                raise
            _log(f"download failed (attempt {attempt + 1}/{_XFER_ATTEMPTS}): {safe_error(e)} — resuming from "
                 f"{dest.stat().st_size if dest.exists() else 0} bytes")
            time.sleep(2**attempt)
    return dest


def _upload_content_type(src: Path) -> str:
    """Infer the small closed set of browser media types from bytes, not the workspace filename.

    The ops input cache uploads an immutable ``*.put-snapshot`` copy, so suffix-only MIME inference loses
    the original ``.jpg``/``.png`` name.  R2 then stores ``application/octet-stream`` and the preview API
    correctly refuses to hand that object to an ``<img>``.  Magic bytes survive the snapshot and are the
    authority for these formats; unknown outputs stay binary.
    """
    with src.open("rb") as f:
        head = f.read(16)
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def upload(src: Path, put_url: str, content_type: str | None = None) -> None:
    """Presigned PUT ← file, streamed, 3 attempts. A file:// url copies to local disk (local backend).

    A retry re-sends the object FROM BYTE 0 — a presigned PUT is one signed request and has no resume. The
    real fix is presigned MULTIPART (the brain mints an uploadId plus per-part urls), which makes a reset
    cost one part instead of the object; until that mint exists the waste is at least MEASURED, not unknown."""
    local = _file_path(put_url)
    if local is not None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local)
        return
    size = src.stat().st_size
    if content_type is None:
        content_type = _upload_content_type(src)
    resent = 0
    for attempt in range(_XFER_ATTEMPTS):
        if attempt:
            resent += size
            _log(f"upload retry {attempt + 1}/{_XFER_ATTEMPTS} for {src.name}: re-sending all {size} bytes "
                 f"from 0 ({resent} bytes re-sent so far, no resume on a presigned PUT)")
        try:
            with src.open("rb") as f:
                r = _store.put(
                    put_url, data=_DeadlineBody(f, size),
                    headers={"Content-Type": content_type, "Content-Length": str(size)},
                    timeout=max(_TIMEOUT, size // (1 << 20)),
                )
            r.raise_for_status()
            return
        except requests.RequestException:
            if attempt + 1 == _XFER_ATTEMPTS:
                raise
            time.sleep(2**attempt)
