"""Control-plane client. The pod dials OUT only: poll a job, report events/results, move bytes
via presigned URLs. Auth = the single job token from the environment; no other credentials exist
on this machine."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

_TIMEOUT = 30
_CHUNK = 1 << 20


class ControlPlane:
    def __init__(self, base_url: str, job_token: str) -> None:
        self.base = base_url.rstrip("/")
        self.sess = requests.Session()
        self.sess.headers["Authorization"] = f"Bearer {job_token}"

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


def download(url: str, dest: Path) -> Path:
    """Presigned GET → file, streamed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=_TIMEOUT) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(_CHUNK):
                f.write(chunk)
    return dest


def upload(src: Path, put_url: str, content_type: str = "application/octet-stream") -> None:
    """Presigned PUT ← file, streamed, 3 attempts."""
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
