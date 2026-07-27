"""Control-plane client. The pod dials OUT only: poll a job, report events/results, move bytes
via presigned URLs. Auth = the single job token from the environment; no other credentials exist
on this machine."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse
from urllib.request import url2pathname

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_TIMEOUT = 30
_CHUNK = 1 << 20
# How long a terminal report keeps trying before the pod gives up on being heard. A result nobody receives
# is the same to the brain as a pod that died, so this is worth far more patience than an ordinary call.
_REPORT_ATTEMPTS = 6
_REPORT_BACKOFF_S = 5.0


def _log(msg: str) -> None:
    print(f"[podagent] {msg}", file=sys.stderr, flush=True)


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
        self.sess = requests.Session()
        self.sess.headers["Authorization"] = f"Bearer {job_token}"
        # Minutes of work between calls ⇒ the pooled socket is always dead by the terminal POST, and urllib3
        # will not replay an unsafe method: that POST never reaches the server and leaves no access-log line.
        retry = Retry(total=5, connect=5, read=0, status=3, status_forcelist=(502, 503, 504),
                      allowed_methods=False, backoff_factor=0.5, raise_on_status=False)
        adapter = HTTPAdapter(max_retries=retry)
        self.sess.mount("http://", adapter)
        self.sess.mount("https://", adapter)

    def poll_job(self) -> dict[str, Any] | None:
        """One long-poll for work. Returns the job envelope or None on timeout/no-work."""
        r = self.sess.get(f"{self.base}/pod/job", timeout=_TIMEOUT + 35)
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json()  # type: ignore[no-any-return]

    def post_event(self, payload: dict[str, Any]) -> None:
        self.sess.post(f"{self.base}/pod/event", json=payload, timeout=_TIMEOUT).raise_for_status()

    def post_infer_result(self, payload: dict[str, Any]) -> None:
        self.sess.post(f"{self.base}/pod/infer-result", json=payload, timeout=_TIMEOUT).raise_for_status()

    def note(self, payload: dict[str, Any]) -> None:
        """Best-effort progress event; never raises — a pod must not die of saying what it is doing."""
        try:
            self.post_event(payload)
        except Exception as e:  # noqa: BLE001 — a progress ping is never worth failing a job over
            _log(f"progress event dropped ({payload.get('step', '')!r}): {e}")

    def report_infer_result(self, payload: dict[str, Any], wake_key: str | None = None) -> bool:
        """Deliver a terminal InferResult, retrying, then fall back to an error event. Never raises.
        It is the brain's ONLY wake-up: undelivered, it costs a full INFER_TIMEOUT_S of billed silence."""
        for attempt in range(_REPORT_ATTEMPTS):
            try:
                self.post_infer_result(payload)
                return True
            except Exception as e:  # noqa: BLE001 — every failure mode here is retryable from our side
                _log(f"infer-result post failed (attempt {attempt + 1}/{_REPORT_ATTEMPTS}): {e}")
                if attempt + 1 < _REPORT_ATTEMPTS:
                    time.sleep(_REPORT_BACKOFF_S * (attempt + 1))
        # corr_id carries the awaited result_key: the CP echoes it onto the terminal it synthesizes from a
        # chain-level error event, which is what lets this reach the keyed awaiter instead of just a log.
        ev: dict[str, Any] = {
            "job_id": payload.get("job_id", "unknown"), "stage": "infer", "status": "error",
            "error": f"result undeliverable: {str(payload.get('error') or payload)[:400]}"}
        key = payload.get("result_key") or payload.get("corr_id") or wake_key
        if key:
            ev["corr_id"] = key
        try:
            self.post_event(ev)
        except Exception as e:  # noqa: BLE001 — nothing left to try; stderr is the last channel
            _log(f"FATAL both /pod/infer-result and /pod/event are unreachable: {e}")
            return False
        _log("infer-result undeliverable — reported as a /pod/event instead")
        return True


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
        f"pack checks it against its own host allowlist (montyops.stock_hosts) — not in a binding, where "
        f"this transport would fetch whatever a third party's search response happened to contain.")


def download(url: str, dest: Path) -> Path:
    """Presigned GET → file, streamed. A file:// url copies from local disk (local backend)."""
    assert_fetchable(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    local = _file_path(url)
    if local is not None:
        shutil.copyfile(local, dest)
        return dest
    with requests.get(url, stream=True, timeout=_TIMEOUT) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(_CHUNK):
                f.write(chunk)
    return dest


def upload(src: Path, put_url: str, content_type: str = "application/octet-stream") -> None:
    """Presigned PUT ← file, streamed, 3 attempts. A file:// url copies to local disk (local backend)."""
    local = _file_path(put_url)
    if local is not None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local)
        return
    size = src.stat().st_size
    for attempt in range(3):
        try:
            with src.open("rb") as f:
                r = requests.put(
                    put_url, data=f,
                    headers={"Content-Type": content_type, "Content-Length": str(size)},
                    timeout=max(_TIMEOUT, size // (1 << 20)),
                )
            r.raise_for_status()
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
