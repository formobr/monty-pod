"""Secret-safe diagnostics for failures that may contain presigned URLs or bearer credentials."""
from __future__ import annotations

import re
import traceback
from typing import Any
from urllib.parse import urlsplit

_URL = re.compile(r"(?i)\b(?:https?|wss?)://[^\s\"'<>]+")
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s,;\"']+")
_SECRET_PARAM = re.compile(
    r"(?i)(x-amz-(?:signature|credential|security-token)|signature|token|access_token|"
    r"authorization)=([^&\s,;]+)")


def safe_endpoint(value: str) -> str:
    """Keep only scheme, host, port and path; userinfo, query and fragment never reach diagnostics."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return "[redacted-url]"
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = f":{parts.port}" if parts.port is not None else ""
    except ValueError:
        port = ""
    return f"{host}{port}{parts.path}"


def safe_text(value: Any, limit: int | None = None) -> str:
    text = str(value)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _URL.sub("[redacted-url]", text)
    text = _SECRET_PARAM.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    return text if limit is None else text[:limit]


def safe_error(exc: BaseException, limit: int = 500) -> str:
    return safe_text(f"{type(exc).__name__}: {exc}", limit)


def safe_traceback(exc: BaseException) -> str:
    return safe_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
