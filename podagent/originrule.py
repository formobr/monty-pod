"""The pod's ONE origin-url guard: may a keyless RENTED pod dereference this third-party url, and fetch it if so.

A stock provider's thumbnail url comes out of a third party's search response, not out of the control plane,
so it is not a capability we minted (cp.assert_fetchable refuses it, and still must) — it is an ADDRESS the
pod was merely told to visit, on somebody else's network where `http://169.254.169.254/...` is one string
away. This module is the rule for that class of url, ported from the engine's own guards so the two cannot
disagree: the address half from montyops.stock_hosts (scheme, userinfo, port, IP literal, public unicast),
the credential half from the engine's origin-binding guard (userinfo, credential-named parameters, AND the
box's secret-scanner shapes over every parameter value, path and fragment — a pre-signed url's own signing fields
exempt, exactly as the box exempts them), and the redirect
+ retry shape from montyops.media_fetch.resolve (every hop checked BEFORE it is followed; 408/429/5xx
retried, paced per host by the limiter below).

FETCH CONNECTS TO THE EXACT ADDRESS check() JUST VETTED, NEVER ASKING THE RESOLVER AGAIN. A DNS answer with a
short TTL can legitimately change between `check()`'s lookup and whatever `requests`/`urllib` would resolve a
second time at connect() — the public-unicast verdict and the socket's actual peer would then belong to two
different DNS answers (review finding, 2026-10-15). `fetch()` resolves each hop exactly once, through
`check()`'s own helper, and pins the raw connection to that address; TLS SNI and the Host header still carry
the real hostname, so certificate verification is unchanged. The same rewrite also closes the other finding:
a redirect's body is never read at all (the connection is simply closed), so there is no unbounded buffer
tied to a 3xx response the way a stock redirect-handler's own `fp.read()` would be.

Standard library only: it is the guard, so it may not depend on anything that is not already the interpreter.
Its caller is infer_cliprank, for every tile of a clip_rank origin_urls group.
"""
from __future__ import annotations

import email.utils
import http.client
import ipaddress
import math
import random
import re
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import unquote_plus, urljoin, urlsplit

# Copied verbatim from the engine's montyops.stock_hosts._METADATA_ADDRS. Every one is already caught by the
# public-unicast check below; they are written out so the reader sees WHAT is refused, not a CIDR.
_METADATA_ADDRS: Final[frozenset[str]] = frozenset({
    "169.254.169.254",   # AWS / GCP / Azure / OpenStack / DigitalOcean — the canonical one
    "169.254.170.2",     # AWS ECS task-role credentials
    "100.100.100.200",   # Alibaba Cloud
    "192.0.0.192",       # Oracle Cloud
    "fd00:ec2::254",     # AWS IMDS over IPv6
})

# Copied verbatim from the engine's origin-binding guard's own credential-parameter set.
_CRED_PARAMS: Final[frozenset[str]] = frozenset({
    "key", "api_key", "apikey", "auth", "auth_token", "token", "access_token", "refresh_token",
    "password", "passwd", "pwd", "secret", "client_secret", "credential", "credentials", "sig", "signature",
})
# Copied verbatim from the same guard's signing-prefix tuple: a pre-signed url's own signing fields
# are the capability, not a leak.
_SIGNING_PREFIXES: Final[tuple[str, ...]] = ("x-amz-", "x-goog-", "x-ms-", "x-obs-")

# VENDORED from the engine's secret scanner at engine commit 3d919bab156ebeab9b1f6695f3f0c3b898f83395 — its shape
# list, its secret-name and not-a-value patterns and its key-material test: everything its shape scan reads with
# an empty allowlist (one rule id renamed to describe the role, not the provider; the pattern is unchanged).
# The box's origin-binding guard runs its shape scan on `{name} = "{value}"` over every query pair and over the
# url's scheme://host/path; a pod that skipped it accepted a secret under a neutral name the box refuses.
# tests/test_originrule_refuses_a_secret_under_any_name.py pins these against that commit's literals.
SECRET_SHAPES_SOURCE: Final[str] = "engine secret scanner@3d919bab156ebeab9b1f6695f3f0c3b898f83395"
_MIN_RUN: Final[int] = 16
_MIN_ENTROPY: Final[float] = 3.0
_RUN: Final = re.compile(r"[A-Za-z0-9_-]{%d,}" % _MIN_RUN)
_SHAPES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("tiktok",              re.compile(r"\b(?:act|rft)\.[A-Za-z0-9!._-]{20,}")),
    ("google_oauth_secret", re.compile(r"\bGOCSPX-[A-Za-z0-9_-]{10,}")),
    ("google_refresh",      re.compile(r"(?<![\w/])1//[A-Za-z0-9_-]{20,}")),
    ("google_access",       re.compile(r"\bya29\.[A-Za-z0-9_.-]{20,}")),
    ("meta",                re.compile(r"\bEAA[A-Za-z0-9]{20,}")),
    ("instagram",           re.compile(r"\bIGQ[A-Za-z0-9_-]{20,}")),
    ("chat_bot_token",      re.compile(r"\b\d{8,12}:AA[A-Za-z0-9_-]{30,}")),
    ("huggingface",         re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("github",              re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("vk",                  re.compile(r"\bvk1\.a\.[A-Za-z0-9_-]{20,}")),
    ("openai_style",        re.compile(r"\bsk-(?:ant-|or-)?[A-Za-z0-9_-]{20,}")),
    ("aws_access_key_id",   re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
)
_SECRET_NAME: Final = re.compile(
    r"""(?xi)
    ((?:[A-Za-z_][A-Za-z0-9_.\[\]"'-]*)?
     (?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|(?-i:[A-Z0-9_]KEY))
     [A-Za-z0-9_]*)
    \s*[:=]{1,2}\s*
    (?:r|f|b|u)?["'`]([^"'`\n]{20,})["'`]
    """
)
_NOT_A_VALUE: Final = re.compile(
    r"(?i)\$\{|\$[a-z_]|%\(|%s\b|\{\{|<[a-z_]+>|os\.environ|getenv|process\.env|your[_-]|example|"
    r"placeholder|redact|xxxx|\.\.\.|(?-i:^[a-z0-9_./-]+$)"
)


def _is_key_material(run: str) -> bool:
    classes = sum(bool(re.search(r, run)) for r in (r"[a-z]", r"[A-Z]", r"\d"))
    n = len(run)
    entropy = -sum(c / n * math.log2(c / n) for c in Counter(run).values()) if n else 0.0
    return n >= _MIN_RUN and classes >= 2 and entropy >= _MIN_ENTROPY


def _secret_shape(text: str) -> str | None:
    """The rule id of the first secret shape in `text` — the box's shape scan with an empty allowlist, made
    a predicate. Never the matched value: the name is what a refusal may print."""
    for line in text.splitlines():
        for rule, rx in _SHAPES:
            if rx.search(line):
                return rule
        for m in _SECRET_NAME.finditer(line):
            val = m.group(2)
            if _NOT_A_VALUE.search(val) or not _is_key_material(max(_RUN.findall(val) or [""], key=len)):
                continue
            return f"assigned:{m.group(1)[:40]}"
    return None


# The engine's montyops.media_fetch status set. Three attempts (PROD 2026-10-28: two attempts 0.5 s apart
# hammered a throttling CDN and gave up inside the same second); the pause between them is the host limiter's
# hold below, never a fixed sleep.
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
_ORIGIN_ATTEMPTS: Final[int] = 3

# PER-HOST PACING. One state per origin host for the whole process, shared by every fetch: AIMD on the number of
# fetches in flight to that host (a throttle or a dead connection halves it, a success adds one), and a 429/503
# also holds NEW fetches to that host. Neither pexels nor pixabay publishes CDN pacing numbers, so the limiter
# finds the rate instead of guessing one. Every wait is capped: an uncapped Retry-After would rebuild the endless
# wait this replaces.
_POOL_WIDTH: Final[int] = 32       # the ceiling: infer_cliprank._FETCH_WORKERS, the widest the pod ever fetches
_HOLD_MIN_S: Final[float] = 0.5
_HOLD_MAX_S: Final[float] = 8.0
_BACKOFF_BASE_S: Final[float] = 1.0   # 1, 2, 4, 8 s when the host names no Retry-After
_JITTER: Final[float] = 0.25
_SLOT_POLL_S: Final[float] = 0.25     # a fetch waiting for a free slot re-checks its deadline this often
_MAX_IDLE_PER_HOST: Final[int] = 8    # persistent connections kept per host

# Standard HTTP redirect statuses — a protocol literal, not a tunable; the same set urllib's own
# HTTPRedirectHandler special-cases by name (http_error_301/302/303/307/308).
_REDIRECT_STATUSES: Final[frozenset[int]] = frozenset({301, 302, 303, 307, 308})

# NAT64's well-known prefix, and the deprecated IPv4-compatible IPv6 form: both carry a v4 address in their
# low 32 bits, exactly like the ipv4_mapped case ipaddress itself unwraps — judged as that v4 address or a
# pod sitting behind NAT64/DNS64 could be handed 169.254.169.254 wearing a v6 prefix ipaddress calls global.
_NAT64_PREFIX: Final = ipaddress.ip_network("64:ff9b::/96")
_V4_COMPAT_PREFIX: Final = ipaddress.ip_network("::/96")

_FIX: Final[str] = (
    "This url came from a third party's search response, and a KEYLESS RENTED pod may dereference the public "
    "internet but nothing that is only reachable from where it happens to sit, and never a url that IS a key.")

# Seams: the tests fake them, nothing else touches them. `_sleep` and `_now` (monotonic) / `_wall` (epoch, for an
# HTTP-date Retry-After) so a pacing test never waits on the clock; `_jitter` so a backoff is exact;
# `_connect` so a test can stand in for the TLS socket where the real one would open a connection.
_sleep = time.sleep
_now = time.monotonic
_wall = time.time


def _jitter() -> float:
    return random.uniform(-_JITTER, _JITTER)


class DeadlineExceeded(TimeoutError):
    """The caller's overall deadline passed before this fetch could start or finish pacing. Never retried."""


def _retry_after_s(value: str | None) -> float | None:
    """Retry-After as seconds from now: delta-seconds or an HTTP-date; None when absent or unreadable."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return when.timestamp() - _wall()


class _Host:
    """One origin host's pacing state and idle connections, shared by every fetch of the process."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.limit = _POOL_WIDTH
        self.inflight = 0
        self.hold_until = 0.0
        self.backoff_n = 0
        self.idle: list[tuple[str, Any]] = []

    def acquire(self, deadline: float | None) -> None:
        """Wait out the host's hold and for a free slot, then take it. Raises DeadlineExceeded instead of
        waiting past `deadline`."""
        while True:
            with self.cond:
                now = _now()
                if deadline is not None and now >= deadline:
                    raise DeadlineExceeded("deadline reached before the host would take another fetch")
                hold = self.hold_until - now
                if hold <= 0:
                    if self.inflight < self.limit:
                        self.inflight += 1
                        return
                    self.cond.wait(_SLOT_POLL_S)
                    continue
                if deadline is not None and now + hold > deadline:
                    raise DeadlineExceeded(f"deadline reached before the {hold:.1f}s hold on this host ended")
            _sleep(hold)

    def release(self, outcome: str, retry_after: str | None = None) -> None:
        """Give the slot back and learn from how the fetch went: ok | throttle | block | conn_error | neutral."""
        with self.cond:
            self.inflight -= 1
            if outcome == "ok":
                self.limit = min(_POOL_WIDTH, self.limit + 1)
                self.backoff_n = 0
            elif outcome in ("throttle", "block", "conn_error"):
                self.limit = max(1, self.limit // 2)
            if outcome == "throttle":
                named = _retry_after_s(retry_after)
                if named is not None:
                    hold = min(max(named, _HOLD_MIN_S), _HOLD_MAX_S)
                else:
                    hold = min(_BACKOFF_BASE_S * 2 ** self.backoff_n * (1 + _jitter()), _HOLD_MAX_S)
                    self.backoff_n = min(self.backoff_n + 1, 4)     # 2**4 is already past the cap
                self.hold_until = max(self.hold_until, _now() + hold)
            self.cond.notify_all()

    def take_idle(self, addrs: tuple[str, ...]) -> tuple[str, Any] | None:
        """A persistent connection pinned to one of the addresses just vetted; one to any other is closed."""
        with self.cond:
            while self.idle:
                addr, conn = self.idle.pop()
                if addr in addrs:
                    return addr, conn
                conn.close()
        return None

    def put_idle(self, addr: str, conn: Any) -> None:
        with self.cond:
            if len(self.idle) < _MAX_IDLE_PER_HOST:
                self.idle.append((addr, conn))
                return
        conn.close()


_HOSTS: dict[str, _Host] = {}
_hosts_lock = threading.Lock()


def _host_state(host: str) -> _Host:
    with _hosts_lock:
        return _HOSTS.setdefault(host, _Host())


@dataclass(frozen=True)
class OriginFailure:
    """Where and why ONE origin fetch failed — diagnostics only, attached to the exception as `origin_failure`
    and never read by a retry decision. `phase` is dns | connect | tls | http | read; `host` is the final hop's;
    `cf_code` is the first `error code: 1xxx` of the first 2 KB of an error body (Cloudflare's own verdict)."""
    phase: str
    host: str
    addr: str | None = None
    family: str | None = None       # "v4" | "v6" of `addr`
    status: int | None = None
    retry_after: str | None = None
    cf_mitigated: str | None = None
    cf_ray: str | None = None
    cf_code: str | None = None
    exc: str = ""


_BODY_PEEK: Final[int] = 2048
_HEADER_CAP: Final[int] = 64
_CF_CODE: Final = re.compile(r"error code: (1\d{3})")


def _family(addr: str | None) -> str | None:
    return None if addr is None else ("v6" if ":" in addr else "v4")


def _attach(exc: BaseException, **fields: Any) -> BaseException:
    """Record `exc`'s failure on itself, once: the innermost site knows the most."""
    if getattr(exc, "origin_failure", None) is None:
        fields.setdefault("family", _family(fields.get("addr")))
        exc.origin_failure = OriginFailure(exc=type(exc).__name__, **fields)  # type: ignore[attr-defined]
    return exc


class OriginBody(bytes):
    """A 2xx body whose Content-Type said text/html: returned as today, flagged so the caller can name it."""
    html = True


def _http_failure(resp: Any, host: str, addr: str, status: int, exc_name: str) -> OriginFailure:
    """The record of an error status: selected headers and the first 2 KB of the body, both best effort."""
    def header(name: str) -> str | None:
        value = resp.getheader(name)
        return str(value)[:_HEADER_CAP] if value is not None else None
    try:
        peek = resp.read(_BODY_PEEK)
    except (OSError, http.client.HTTPException):
        peek = b""
    m = _CF_CODE.search(peek.decode("utf-8", "replace"))
    return OriginFailure(phase="http", host=host, addr=addr, family=_family(addr), status=status,
                         retry_after=header("Retry-After"), cf_mitigated=header("cf-mitigated"),
                         cf_ray=header("cf-ray"), cf_code=m.group(1) if m else None, exc=exc_name)


class OriginRefused(ValueError):
    """A url the pod may not dereference, or a response it may not accept. `reason` never echoes a query
    string or userinfo — a refusal that prints the credential is the leak it prevents."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Transient(Exception):
    """Internal: the origin answered THIS hop with a status `fetch()`'s outer retry loop should retry."""

    def __init__(self, status: int, reason: str, failure: OriginFailure | None = None) -> None:
        super().__init__(status)
        self.status = status
        self.reason = reason
        self.failure = failure


def _safe(parts: Any) -> str:
    # userinfo, port, query and path parameters DROPPED: any of them may BE the credential. Each segment is
    # percent-decoded first so a `;name=value` spelled `%3Bname=value` is dropped too, same as `_params`.
    path = "/".join(unquote_plus(seg).split(";", 1)[0] for seg in parts.path.split("/"))
    return f"{parts.scheme}://{parts.hostname or ''}{path}"


def resolve_addrs(host: str) -> tuple[str, ...]:
    """Every address `host` resolves to, v4 and v6. A name that does not resolve is REFUSED rather than
    returning empty — "I could not tell where this points" must not read as "it points nowhere dangerous"."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError as e:
        refused = OriginRefused(f"REFUSED: {host} does not resolve ({e}). A host we cannot place is a host we "
                                f"cannot clear. {_FIX}")
        _attach(refused, phase="dns", host=host)
        raise refused from e
    return tuple(dict.fromkeys(str(info[4][0]) for info in infos))


def _public_unicast(addr: str) -> bool:
    """Is this address on the PUBLIC unicast internet? Private, loopback, link-local, CGNAT, multicast,
    unspecified and otherwise reserved addresses are somewhere only this rented box can reach."""
    try:
        ip: Any = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return False
    # An IPv4-mapped v6 address is the v4 one wearing a hat: judge it as the v4 address it is. NAT64 and the
    # deprecated IPv4-compatible form wear the same hat over a different prefix.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    elif ip.version == 6 and (ip in _NAT64_PREFIX or ip in _V4_COMPAT_PREFIX):
        ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if str(ip) in _METADATA_ADDRS:
        return False
    return bool(ip.is_global) and not ip.is_multicast


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


def _params(parts: Any) -> list[tuple[str, str]]:
    """Every name=value the url carries: query pairs split on BOTH '&' and ';' (parse_qsl no longer splits on
    ';', but plenty of servers still do), plus ';name=value' path parameters in any path segment — a path
    segment is percent-DECODED before that split, so a credential spelled `%3Btoken=x` is not missed."""
    out = []
    for p in parts.query.replace(";", "&").split("&"):
        if not p:
            continue
        name, _, value = p.partition("=")
        out.append((unquote_plus(name), unquote_plus(value)))
    for seg in parts.path.split("/"):
        for p in unquote_plus(seg).split(";")[1:]:
            if not p:
                continue
            name, _, value = p.partition("=")
            out.append((name, value))
    return out


def _resolve_checked(host: str) -> tuple[str, ...]:
    """Resolve `host` exactly once and hand back its addresses, refusing if any of them is not
    public-unicast. The ONE result both `check()`'s verdict and `fetch()`'s pinned connection are built
    from — never resolved a second time, so the two can never see two different DNS answers."""
    addrs = resolve_addrs(host)
    bad = [a for a in addrs if not _public_unicast(a)]
    if bad:
        raise OriginRefused(
            f"REFUSED: {host} resolves to {', '.join(bad)}, which is not on the public unicast internet. "
            f"{_FIX}")
    return addrs


def _check_and_resolve(url: str) -> tuple[str, tuple[str, ...]]:
    """`check()`'s full body, also handing back the resolved addresses so a caller that is about to connect
    never resolves `host` again to get them."""
    parts = urlsplit(url)
    safe = _safe(parts)
    if parts.scheme != "https":
        raise OriginRefused(
            f"REFUSED: scheme {parts.scheme!r} is not https ({safe}). A keyless RENTED pod fetches over TLS "
            f"or not at all.")
    if parts.username is not None or parts.password is not None:
        raise OriginRefused(f"REFUSED: the authority carries userinfo ({safe}). {_FIX}")
    for name, value in _params(parts):
        low = name.lower()
        if low.startswith(_SIGNING_PREFIXES):
            continue
        if value and low in _CRED_PARAMS:
            raise OriginRefused(f"REFUSED: {safe} carries the credential-shaped parameter {name!r}. {_FIX}")
        if rule := _secret_shape(f'{name} = "{value}"'):
            raise OriginRefused(
                f"REFUSED: {safe} carries secret-shaped material ({rule}) in its parameter {name!r}. {_FIX}")
    # The box splits its query on '&' only, so `w=640;X-Amz-Credential=AKIA…` is ONE pair named `w` there —
    # never exempt. Every '&' pair is therefore scanned whole as well, unless its own name is a signing field.
    for p in parts.query.split("&"):
        name, _, value = p.partition("=")
        name, value = unquote_plus(name), unquote_plus(value)
        if p and not name.lower().startswith(_SIGNING_PREFIXES):
            if rule := _secret_shape(f'{name} = "{value}"'):
                raise OriginRefused(
                    f"REFUSED: {safe} carries secret-shaped material ({rule}) in its parameter {name!r}. {_FIX}")
    # The url itself — host, path (raw and decoded, path parameters included) and fragment: the box scans
    # scheme://host/path; the fragment is scanned too because a refusal costs nothing a real CDN url carries.
    for where, text in (("host or path", f"{parts.scheme}://{parts.hostname or ''}{parts.path}"),
                        ("path", unquote_plus(parts.path)), ("fragment", unquote_plus(parts.fragment))):
        if rule := _secret_shape(text):
            # Host only: the path IS the secret here, so even the path-parameter-free `safe` would echo it.
            raise OriginRefused(
                f"REFUSED: a url on {parts.hostname or ''} carries secret-shaped material ({rule}) in its "
                f"{where}. {_FIX}")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise OriginRefused(f"REFUSED: no host in {safe!r}.")
    try:
        port = parts.port
    except ValueError as e:
        raise OriginRefused(f"REFUSED: unreadable port on {host} ({e}).") from e
    if port not in (None, 443):
        raise OriginRefused(
            f"REFUSED: explicit port {port} on {host} is not the https port. No public CDN is on one; "
            f"plenty of internal services are. {_FIX}")
    if _is_ip_literal(host):
        raise OriginRefused(
            f"REFUSED: {host} is an IP literal. No stock CDN is addressed by number, and the metadata "
            f"service is exactly what that would be for. {_FIX}")
    return url, _resolve_checked(host)


def check(url: str) -> str:
    """Refuse a url a keyless RENTED pod may not dereference, else hand it back unchanged."""
    return _check_and_resolve(url)[0]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """An HTTPSConnection that connects to `addr` — the address `check()` already vetted — instead of asking
    the resolver to turn `host` into one again. SNI and the Host header still carry `host`, so certificate
    verification checks the real domain; only the socket's peer is pinned."""

    def __init__(self, host: str, addr: str, *, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self._pinned_addr = addr

    def connect(self) -> None:
        try:
            sock = socket.create_connection((self._pinned_addr, self.port), timeout=self.timeout)
        except OSError as e:
            _attach(e, phase="connect", host=self.host, addr=self._pinned_addr)
            raise
        try:
            self.sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
        except OSError as e:     # ssl.SSLError is one
            _attach(e, phase="tls", host=self.host, addr=self._pinned_addr)
            raise


def _connect(host: str, addr: str, *, timeout_s: float) -> Any:
    conn = _PinnedHTTPSConnection(host, addr, timeout=timeout_s)
    conn.connect()
    return conn


def _open(state: _Host, host: str, addrs: tuple[str, ...], timeout_s: float, *, fresh: bool) -> tuple[str, Any, bool]:
    """A connection to `host` pinned to a vetted address: an idle one when there is one, else a new one, trying
    each vetted address in turn (v4 first, then v6) — only when ALL fail does the attempt fail, naming the
    address that failed last. Returns (addr, connection, reused)."""
    ordered = sorted(addrs, key=lambda a: ":" in a)
    if not fresh and (idle := state.take_idle(tuple(ordered))) is not None:
        return idle[0], idle[1], True
    last: OSError | None = None
    for addr in ordered:
        try:
            return addr, _connect(host, addr, timeout_s=timeout_s), False
        except OSError as e:
            _attach(e, phase="connect", host=host, addr=addr)
            last = e
    assert last is not None
    raise last


def _send(state: _Host, host: str, addrs: tuple[str, ...], path: str, timeout_s: float) -> tuple[str, Any, Any]:
    """Open (or reuse) a connection and send the GET. A REUSED connection the server closed while it sat idle
    is not a failure: it is dropped and the request goes once more on a fresh one."""
    fresh = False
    while True:
        addr, conn, reused = _open(state, host, addrs, timeout_s, fresh=fresh)
        try:
            conn.request("GET", path, headers={"User-Agent": "monty-pod-originrule/1"})
            return addr, conn, conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            conn.close()
            if reused:
                fresh = True
                continue
            _attach(e, phase="read", host=host, addr=addr)
            raise


def _fetch_once(url: str, *, max_bytes: int, timeout_s: float, deadline: float | None = None) -> bytes:
    """Follow redirects, re-checking (and re-resolving) every hop before connecting to it, and connect to an
    address just resolved — never a second lookup. Every hop goes through its host's limiter. `max_hops`
    borrows urllib's own redirect bound rather than authoring a new number."""
    nxt = url
    max_hops = urllib.request.HTTPRedirectHandler.max_redirections
    for hop in range(max_hops):
        checked, addrs = _check_and_resolve(nxt)
        parts = urlsplit(checked)
        host = (parts.hostname or "").lower().rstrip(".")
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        state = _host_state(host)
        state.acquire(deadline)
        outcome, retry_after = "neutral", None
        conn = None
        keep = False
        try:
            try:
                addr, conn, resp = _send(state, host, addrs, path, timeout_s)
            except (OSError, http.client.HTTPException):
                outcome = "conn_error"
                raise
            status = resp.status
            if status in _REDIRECT_STATUSES:
                outcome = "ok"
                location = resp.getheader("Location")
                if not location:
                    raise OriginRefused(
                        f"REFUSED: redirect hop {hop + 1} from {host} carried no Location header to follow.")
                # The body of a redirect is never read: this connection is simply closed, so no byte cap
                # applies here because none is needed — nothing is buffered.
                nxt = urljoin(checked, location)
                continue
            if status in (429, 503):
                outcome, retry_after = "throttle", resp.getheader("Retry-After")
            elif status == 403:
                outcome = "block"      # halves the host's width, and is never retried (it is not in _RETRYABLE)
            if status in _RETRYABLE_STATUS:
                raise _Transient(status, resp.reason, _http_failure(resp, host, addr, status, "HTTPError"))
            if status >= 400:
                err = urllib.error.HTTPError(checked, status, resp.reason, http.client.HTTPMessage(), None)
                err.origin_failure = _http_failure(resp, host, addr, status, "HTTPError")  # type: ignore[attr-defined]
                raise err
            try:
                data = resp.read(max_bytes + 1)
            except (OSError, http.client.HTTPException) as e:
                outcome = "conn_error"
                _attach(e, phase="read", host=host, addr=addr)
                raise
            outcome = "ok"
            html = str(resp.getheader("Content-Type") or "").lstrip().lower().startswith("text/html")
            # Back to the pool only when the body was read to its end on a connection the server keeps open.
            keep = (len(data) <= max_bytes and not getattr(resp, "will_close", False)
                    and getattr(resp, "isclosed", lambda: True)())
        finally:
            if conn is not None:
                if keep:
                    state.put_idle(addr, conn)
                else:
                    conn.close()
            state.release(outcome, retry_after)
        if len(data) > max_bytes:
            raise OriginRefused(
                f"REFUSED: the body from {_safe(urlsplit(checked))} is over the {max_bytes}-byte cap.")
        return OriginBody(data) if html else data
    raise OriginRefused(f"REFUSED: more than {max_hops} redirect hops fetching {urlsplit(url).hostname or ''}")


def fetch(url: str, *, max_bytes: int, timeout_s: float, deadline: float | None = None) -> bytes:
    """GET a checked url, refusing every unsafe hop before it is followed and any body over `max_bytes`.

    A connection that dies is retried, and so is a status that says the origin could NOT answer — 408, 429,
    5xx — up to `_ORIGIN_ATTEMPTS` in all, each one paced by the host's limiter. An origin that delivers a
    verdict on the request (403, 404) is not, and a refusal never is. `deadline` is an absolute `_now()` time:
    no attempt starts, and no pacing wait runs, past it (DeadlineExceeded)."""
    last: BaseException | None = None
    for attempt in range(_ORIGIN_ATTEMPTS):
        if deadline is not None and _now() >= deadline:
            raise DeadlineExceeded("deadline reached before the next attempt")
        try:
            return _fetch_once(url, max_bytes=max_bytes, timeout_s=timeout_s, deadline=deadline)
        except DeadlineExceeded:
            raise                       # caught ahead of OSError below: TimeoutError is one
        except _Transient as e:
            last = urllib.error.HTTPError(url, e.status, e.reason, http.client.HTTPMessage(), None)
            last.origin_failure = e.failure  # type: ignore[attr-defined]
            why = f"origin could not answer ({e.status})"
        except urllib.error.HTTPError:
            raise                       # the origin's verdict on THIS request — a second one reads the
            # same; caught ahead of OSError below because HTTPError is itself an OSError subclass.
        except (OSError, TimeoutError) as e:
            last = e
            why = f"origin connection died ({type(e).__name__})"
        print(f"[originrule] {why} on attempt {attempt + 1}/{_ORIGIN_ATTEMPTS} for {_safe(urlsplit(url))}",
              file=sys.stderr, flush=True)
    assert last is not None
    raise last
