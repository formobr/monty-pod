"""The pod's ONE origin-url guard: may a keyless RENTED pod dereference this third-party url, and fetch it if so.

A stock provider's thumbnail url comes out of a third party's search response, not out of the control plane,
so it is not a capability we minted (cp.assert_fetchable refuses it, and still must) — it is an ADDRESS the
pod was merely told to visit, on somebody else's network where `http://169.254.169.254/...` is one string
away. This module is the rule for that class of url, ported from the engine's own guards so the two cannot
disagree: the address half from montyops.stock_hosts (scheme, userinfo, port, IP literal, public unicast),
the credential half from the engine's origin-binding guard (userinfo, credential-named parameters, a
pre-signed url's own signing fields exempt — its box-only secret-scanning pass stays box-side), and the redirect
+ retry shape from montyops.media_fetch.resolve (every hop checked BEFORE it is followed; 408/429/5xx
retried once).

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

import http.client
import ipaddress
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
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

# Copied verbatim from the engine's montyops.media_fetch (_RETRYABLE_STATUS, _ORIGIN_RETRY_PAUSE_S,
# _ORIGIN_ATTEMPTS): two attempts, not three — the observed transient failure does not repeat.
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
_ORIGIN_RETRY_PAUSE_S: Final[float] = 0.5
_ORIGIN_ATTEMPTS: Final[int] = 2

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

# Seams: the tests fake them, nothing else touches them. `_sleep` so a retry test never waits on the clock;
# `_connect` so a test can stand in for the TLS socket where the real one would open a connection.
_sleep = time.sleep


class OriginRefused(ValueError):
    """A url the pod may not dereference, or a response it may not accept. `reason` never echoes a query
    string or userinfo — a refusal that prints the credential is the leak it prevents."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Transient(Exception):
    """Internal: the origin answered THIS hop with a status `fetch()`'s outer retry loop should retry."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(status)
        self.status = status
        self.reason = reason


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
        raise OriginRefused(f"REFUSED: {host} does not resolve ({e}). A host we cannot place is a host we "
                            f"cannot clear. {_FIX}") from e
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
        sock = socket.create_connection((self._pinned_addr, self.port), timeout=self.timeout)
        self.sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)


def _connect(host: str, addr: str, *, timeout_s: float) -> Any:
    conn = _PinnedHTTPSConnection(host, addr, timeout=timeout_s)
    conn.connect()
    return conn


def _fetch_once(url: str, *, max_bytes: int, timeout_s: float) -> bytes:
    """Follow redirects, re-checking (and re-resolving) every hop before connecting to it, and connect to the
    exact address just resolved — never a second lookup. `max_hops` borrows urllib's own redirect bound
    rather than authoring a new number."""
    nxt = url
    max_hops = urllib.request.HTTPRedirectHandler.max_redirections
    for hop in range(max_hops):
        checked, addrs = _check_and_resolve(nxt)
        parts = urlsplit(checked)
        host = (parts.hostname or "").lower().rstrip(".")
        addr = next((a for a in addrs if ":" not in a), addrs[0])
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        conn = _connect(host, addr, timeout_s=timeout_s)
        try:
            conn.request("GET", path, headers={"User-Agent": "monty-pod-originrule/1", "Connection": "close"})
            resp = conn.getresponse()
            status = resp.status
            if status in _REDIRECT_STATUSES:
                location = resp.getheader("Location")
                if not location:
                    raise OriginRefused(
                        f"REFUSED: redirect hop {hop + 1} from {host} carried no Location header to follow.")
                # The body of a redirect is never read: this connection is simply closed, so no byte cap
                # applies here because none is needed — nothing is buffered.
                nxt = urljoin(checked, location)
                continue
            if status in _RETRYABLE_STATUS:
                raise _Transient(status, resp.reason)
            if status >= 400:
                raise urllib.error.HTTPError(checked, status, resp.reason, http.client.HTTPMessage(), None)
            data = resp.read(max_bytes + 1)
        finally:
            conn.close()
        if len(data) > max_bytes:
            raise OriginRefused(
                f"REFUSED: the body from {_safe(urlsplit(checked))} is over the {max_bytes}-byte cap.")
        return data
    raise OriginRefused(f"REFUSED: more than {max_hops} redirect hops fetching {urlsplit(url).hostname or ''}")


def fetch(url: str, *, max_bytes: int, timeout_s: float) -> bytes:
    """GET a checked url, refusing every unsafe hop before it is followed and any body over `max_bytes`.

    A connection that dies is retried once, and so is a status that says the origin could NOT answer —
    408, 429, 5xx. An origin that delivers a verdict on the request (403, 404) is not, and a refusal never is."""
    last: BaseException | None = None
    for attempt in range(_ORIGIN_ATTEMPTS):
        try:
            return _fetch_once(url, max_bytes=max_bytes, timeout_s=timeout_s)
        except _Transient as e:
            last = urllib.error.HTTPError(url, e.status, e.reason, http.client.HTTPMessage(), None)
            why = f"origin could not answer ({e.status})"
        except urllib.error.HTTPError:
            raise                       # the origin's verdict on THIS request — a second one reads the
            # same; caught ahead of OSError below because HTTPError is itself an OSError subclass.
        except (OSError, TimeoutError) as e:
            last = e
            why = f"origin connection died ({type(e).__name__})"
        print(f"[originrule] {why} on attempt {attempt + 1}/{_ORIGIN_ATTEMPTS} for {_safe(urlsplit(url))}",
              file=sys.stderr, flush=True)
        if attempt + 1 < _ORIGIN_ATTEMPTS:
            _sleep(_ORIGIN_RETRY_PAUSE_S)
    assert last is not None
    raise last
