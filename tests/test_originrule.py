"""podagent.originrule — the pod's one stdlib guard for a third-party (stock thumbnail) origin url.

Every test fakes DNS (`resolve_addrs`) and the connection (`_connect`): no socket is ever opened, and the
retry pause goes through the `_sleep` seam, so no verdict here depends on the network or the clock."""
from __future__ import annotations

import ast
import http.client
import io
import sys
import urllib.error
from pathlib import Path

import pytest

from podagent import originrule

_REAL_RESOLVE = originrule.resolve_addrs    # the autouse fixture below replaces the module attribute

PUBLIC = "93.184.215.14"
DNS = {
    "cdn.example.com": (PUBLIC, "2606:2800:21f:cb07:6820:80da:af6b:8b2c"),
    "meta.example.com": ("169.254.169.254",),
    "lan.example.com": ("10.0.0.7",),
    "home.example.com": ("127.0.0.1",),
    "mixed.example.com": (PUBLIC, "192.168.1.1"),
    "v6local.example.com": ("::1",),
    "mapped.example.com": ("::ffff:169.254.169.254",),
    "mcast.example.com": ("224.0.0.1",),
    "zero.example.com": ("0.0.0.0",),
    "cgnat.example.com": ("100.100.100.200",),
    "nat64.example.com": ("64:ff9b::a9fe:a9fe",),  # NAT64-embedded 169.254.169.254
    "other.example.com": ("93.184.215.15",),
}
V6 = DNS["cdn.example.com"][1]


@pytest.fixture(autouse=True)
def _fake_dns(monkeypatch):
    def resolve(host):
        if host not in DNS:
            raise originrule.OriginRefused(f"REFUSED: {host} does not resolve")
        return DNS[host]
    monkeypatch.setattr(originrule, "resolve_addrs", resolve)


class _Clock:
    """The injected time: `sleep` advances it, so a hold is a number in `sleeps`, never a wait."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def wall(self) -> float:
        return 1_800_000_000.0 + self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """A fresh clock, no jitter and no host state carried over from another test."""
    c = _Clock()
    monkeypatch.setattr(originrule, "_now", c.now)
    monkeypatch.setattr(originrule, "_wall", c.wall)
    monkeypatch.setattr(originrule, "_sleep", c.sleep)
    monkeypatch.setattr(originrule, "_jitter", lambda: 0.0)
    monkeypatch.setattr(originrule, "_HOSTS", {})
    return c


class _FakeResponse:
    """A canned (status, headers, body) answered as http.client.HTTPResponse would."""

    def __init__(self, status: int, headers: dict, body: bytes):
        self.status = status
        self.reason = http.client.responses.get(status, "")
        self._headers = {k.lower(): v for k, v in headers.items()}
        self._buf = io.BytesIO(body)

    def getheader(self, name, default=None):
        return self._headers.get(name.lower(), default)

    def read(self, n=-1):
        return self._buf.read(n)


class _Origin:
    """An in-process https origin: "https://host/path" -> list of (status, headers, body) answered in order.
    `_connect(host, addr, timeout_s=...)` is faked entirely — no real socket, no real DNS, no real TLS — so
    these tests also double as the proof that `fetch()` never connects to an address it has not resolved
    and checked for itself (the fake records `host` AND `addr` on every connect)."""

    def __init__(self, script):
        self.script = {u: list(v) for u, v in script.items()}   # an entry may be an exception to raise instead
        self.opened: list[str] = []
        self.connected: list[tuple[str, str]] = []
        self.refuse: set[str] = set()      # addresses whose connect is refused
        self.conns: list = []
        self.headers: list = []

    def connect(self, host: str, addr: str, *, timeout_s: float):
        self.connected.append((host, addr))
        if addr in self.refuse:
            raise ConnectionRefusedError(f"refused {addr}")
        origin, origin_host = self, host

        class _FakeConnection:
            closed = False

            def request(self, method, path, headers=None):
                url = f"https://{origin_host}{path}"
                origin.opened.append(url)
                origin.headers.append(headers)
                entry = origin.script[url].pop(0)
                if isinstance(entry, BaseException):
                    raise entry
                status, headers_, body = entry
                self._resp = _FakeResponse(status, headers_, body)

            def getresponse(self):
                return self._resp

            def close(self):
                self.closed = True

        conn = _FakeConnection()
        self.conns.append(conn)
        return conn


@pytest.fixture
def origin(monkeypatch, clock):
    def install(script):
        o = _Origin(script)
        o.sleeps = clock.sleeps
        monkeypatch.setattr(originrule, "_connect", o.connect)
        return o
    return install


@pytest.mark.parametrize("host", [
    "meta.example.com", "lan.example.com", "home.example.com", "mixed.example.com", "v6local.example.com",
    "mapped.example.com", "mcast.example.com", "zero.example.com", "cgnat.example.com", "nat64.example.com",
    "nowhere.example.com",
])
def test_metadata_private_and_loopback_addresses_are_refused(host):
    with pytest.raises(originrule.OriginRefused):
        originrule.check(f"https://{host}/t.jpg")
    url = "https://cdn.example.com/photos/1/t.jpg?w=320&h=180"
    assert originrule.check(url) == url


@pytest.mark.parametrize("url", [
    "https://169.254.169.254/latest/meta-data/",
    "https://93.184.215.14/t.jpg",
    "https://[::1]/t.jpg",
    "https://[fd00:ec2::254]/t.jpg",
    "http://cdn.example.com/t.jpg",
    "ftp://cdn.example.com/t.jpg",
    "file:///etc/passwd",
    "https://cdn.example.com:8443/t.jpg",
    "https://cdn.example.com:80/t.jpg",
    "https:///t.jpg",
])
def test_ip_literal_plain_http_and_a_non_443_port_are_refused(url):
    with pytest.raises(originrule.OriginRefused):
        originrule.check(url)
    assert originrule.check("https://cdn.example.com:443/t.jpg")


@pytest.mark.parametrize("url", [
    "https://user:pw@cdn.example.com/t.jpg",
    "https://user@cdn.example.com/t.jpg",
    "https://cdn.example.com/t.jpg?api_key=abc",
    "https://cdn.example.com/t.jpg?w=1&TOKEN=abc",
    "https://cdn.example.com/t.jpg?Signature=abc",
    "https://cdn.example.com/t.jpg?t%6Fken=abc",
])
def test_userinfo_and_credential_params_are_refused_but_presign_params_pass(url):
    with pytest.raises(originrule.OriginRefused) as e:
        originrule.check(url)
    assert "abc" not in e.value.reason and "pw" not in e.value.reason
    signed = ("https://cdn.example.com/t.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA%2Fx"
              "&X-Amz-Signature=0a79&X-Goog-Signature=ff&x-ms-signature=1&x-obs-credential=2&token=")
    assert originrule.check(signed) == signed


@pytest.mark.parametrize("url", [
    "https://cdn.example.com/t.jpg?w=1;token=abc",
    "https://cdn.example.com/t.jpg?w=1;api_key=abc&h=2",
    "https://cdn.example.com/t.jpg;token=abc",
    "https://cdn.example.com/photos;sig=abc/preview",
    "https://cdn.example.com/photos;v=1;Password=abc/preview",
    "https://cdn.example.com/t.jpg%3Btoken=abc",
])
def test_semicolon_and_path_parameter_credentials_are_refused(url):
    with pytest.raises(originrule.OriginRefused) as e:
        originrule.check(url)
    assert "abc" not in e.value.reason
    assert originrule.check("https://cdn.example.com/photos;v=1/preview?w=1;h=2")


def test_a_redirect_to_a_refused_host_is_refused_before_it_is_followed(origin):
    o = origin({
        "https://cdn.example.com/a.jpg": [(302, {"Location": "https://meta.example.com/latest/"}, b"")],
        "https://cdn.example.com/b.jpg": [(301, {"Location": "/c.jpg?token=abc"}, b"")],
        "https://cdn.example.com/d.jpg": [(302, {"Location": "http://cdn.example.com/e.jpg"}, b"")],
        "https://cdn.example.com/ok.jpg": [(302, {"Location": "/final.jpg"}, b"")],
        "https://cdn.example.com/final.jpg": [(200, {}, b"JPEG")],
    })
    for first in ("a", "b", "d"):
        with pytest.raises(originrule.OriginRefused):
            originrule.fetch(f"https://cdn.example.com/{first}.jpg", max_bytes=100, timeout_s=5)
    assert o.opened == [f"https://cdn.example.com/{x}.jpg" for x in ("a", "b", "d")]
    # A safe relative hop is joined against the hop that sent it and followed.
    assert originrule.fetch("https://cdn.example.com/ok.jpg", max_bytes=100, timeout_s=5) == b"JPEG"
    # And the FIRST url is checked before any socket: nothing is opened for it.
    with pytest.raises(originrule.OriginRefused):
        originrule.fetch("https://lan.example.com/x.jpg", max_bytes=100, timeout_s=5)
    assert "https://lan.example.com/x.jpg" not in o.opened
    # Every connection made pins to an address this run actually resolved and checked for that host — never
    # some other host's address, and never an address DNS did not just hand back for the hop being fetched.
    for host, addr in o.connected:
        assert addr in DNS[host]


def test_a_body_over_the_cap_is_refused(origin):
    origin({
        "https://cdn.example.com/big.jpg": [(200, {}, b"x" * 11)],
        "https://cdn.example.com/fit.jpg": [(200, {}, b"x" * 10)],
    })
    with pytest.raises(originrule.OriginRefused):
        originrule.fetch("https://cdn.example.com/big.jpg", max_bytes=10, timeout_s=5)
    assert originrule.fetch("https://cdn.example.com/fit.jpg", max_bytes=10, timeout_s=5) == b"x" * 10


@pytest.mark.parametrize("status", sorted(originrule._RETRYABLE_STATUS))
def test_a_transient_status_is_retried_up_to_three_attempts(origin, status):
    o = origin({
        "https://cdn.example.com/t.jpg": [(status, {}, b""), (status, {}, b""), (200, {}, b"JPEG")],
        "https://cdn.example.com/thrice.jpg": [(status, {}, b"")] * 3 + [(200, {}, b"never")],
        "https://cdn.example.com/404.jpg": [(404, {}, b""), (200, {}, b"never")],
    })
    assert originrule.fetch("https://cdn.example.com/t.jpg", max_bytes=100, timeout_s=5) == b"JPEG"
    # Three in all, not more: a third transient status is the answer.
    with pytest.raises(urllib.error.HTTPError) as e:
        originrule.fetch("https://cdn.example.com/thrice.jpg", max_bytes=100, timeout_s=5)
    assert e.value.code == status
    assert o.opened.count("https://cdn.example.com/thrice.jpg") == originrule._ORIGIN_ATTEMPTS == 3
    # A verdict on the request is not retried.
    with pytest.raises(urllib.error.HTTPError):
        originrule.fetch("https://cdn.example.com/404.jpg", max_bytes=100, timeout_s=5)
    assert o.opened.count("https://cdn.example.com/404.jpg") == 1


def test_a_cloudflare_403_keeps_its_type_and_records_why(origin):
    o = origin({"https://cdn.example.com/cf.jpg": [
        (403, {"cf-mitigated": "challenge", "cf-ray": "8a1b-FRA", "Retry-After": "x" * 100},
         b"<html>error code: 1020</html>"),
        (200, {}, b"never")]})
    with pytest.raises(urllib.error.HTTPError) as e:
        originrule.fetch("https://cdn.example.com/cf.jpg", max_bytes=100, timeout_s=5)
    assert e.value.code == 403
    f = e.value.origin_failure
    assert (f.phase, f.status, f.cf_code, f.cf_mitigated, f.cf_ray) == ("http", 403, "1020", "challenge", "8a1b-FRA")
    assert (f.host, f.addr, f.family, f.exc) == ("cdn.example.com", PUBLIC, "v4", "HTTPError")
    assert f.retry_after == "x" * 64
    # a verdict is still not retried: one attempt, no pause
    assert o.opened == ["https://cdn.example.com/cf.jpg"] and o.sleeps == []


def test_a_429_records_retry_after_and_is_retried_under_the_cap(origin):
    o = origin({"https://cdn.example.com/slow.jpg": [(429, {"Retry-After": "30"}, b"")] * 3})
    with pytest.raises(urllib.error.HTTPError) as e:
        originrule.fetch("https://cdn.example.com/slow.jpg", max_bytes=100, timeout_s=5)
    assert e.value.code == 429 and e.value.origin_failure.retry_after == "30"
    assert e.value.origin_failure.phase == "http" and e.value.origin_failure.cf_code is None
    # the host said 30 s; the pod waits the 8 s cap between attempts, and not after the last one
    assert len(o.opened) == originrule._ORIGIN_ATTEMPTS and o.sleeps == [8.0, 8.0]


def test_a_2xx_html_body_is_returned_and_flagged(origin):
    origin({"https://cdn.example.com/h.jpg": [(200, {"Content-Type": "text/html; charset=utf-8"}, b"<html>")],
            "https://cdn.example.com/i.jpg": [(200, {"Content-Type": "image/jpeg"}, b"JPEG")]})
    html = originrule.fetch("https://cdn.example.com/h.jpg", max_bytes=100, timeout_s=5)
    assert html == b"<html>" and getattr(html, "html", False)
    assert not getattr(originrule.fetch("https://cdn.example.com/i.jpg", max_bytes=100, timeout_s=5), "html", False)


def test_a_refused_connect_and_a_tls_failure_record_their_phase_and_address(monkeypatch):
    import ssl
    import types

    attempts: list[tuple] = []

    def refuse(address, timeout=None):
        attempts.append(address)
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(originrule.socket, "create_connection", refuse)
    with pytest.raises(ConnectionRefusedError) as e:
        originrule.fetch("https://cdn.example.com/c.jpg", max_bytes=100, timeout_s=5)
    f = e.value.origin_failure
    # the record names the address that failed LAST: v4 was tried first, v6 after it
    assert (f.phase, f.host, f.addr, f.family, f.exc) == ("connect", "cdn.example.com", V6, "v6",
                                                         "ConnectionRefusedError")
    assert attempts == [(PUBLIC, 443), (V6, 443)] * originrule._ORIGIN_ATTEMPTS

    def bad_handshake(*a, **k):
        raise ssl.SSLError("handshake failure")

    monkeypatch.setattr(originrule.socket, "create_connection", lambda address, timeout=None: object())
    monkeypatch.setattr(originrule.ssl, "create_default_context",
                        lambda: types.SimpleNamespace(wrap_socket=bad_handshake))
    with pytest.raises(ssl.SSLError) as e:
        originrule.fetch("https://cdn.example.com/t.jpg", max_bytes=100, timeout_s=5)
    assert e.value.origin_failure.phase == "tls" and e.value.origin_failure.addr == V6


def _get(path: str, **kw):
    return originrule.fetch(f"https://cdn.example.com/{path}", max_bytes=100, timeout_s=5, **kw)


def _width(host: str = "cdn.example.com") -> int:
    return originrule._host_state(host).limit


def test_a_429_holds_the_host_for_retry_after_halves_its_width_and_a_success_adds_one_back(origin, clock):
    o = origin({"https://cdn.example.com/t.jpg": [(429, {"Retry-After": "3"}, b""), (200, {}, b"JPEG")],
                "https://cdn.example.com/u.jpg": [(200, {}, b"JPEG")] * 2})
    assert _width() == originrule._POOL_WIDTH == 32
    assert _get("t.jpg") == b"JPEG"
    assert o.sleeps == [3.0], "the retry waited out the host's Retry-After, and nothing else"
    assert _width() == 17, "32 halved to 16 by the 429, one added back by the success"
    _get("u.jpg")
    _get("u.jpg")
    assert _width() == 19 and o.sleeps == [3.0]


@pytest.mark.parametrize("retry_after, held", [
    ("3600", 8.0),
    ("0", 0.5),
    ("-5", 0.5),
    ("Wed, 21 Oct 2037 07:28:00 GMT", 8.0),
    ("nonsense", 1.0),                      # unreadable: the exponential backoff's first step
])
def test_retry_after_is_clamped_to_half_a_second_to_eight(origin, retry_after, held):
    o = origin({"https://cdn.example.com/t.jpg": [(503, {"Retry-After": retry_after}, b""), (200, {}, b"JPEG")]})
    assert _get("t.jpg") == b"JPEG"
    assert o.sleeps == [held]


def test_an_http_date_retry_after_is_read_against_the_wall_clock(origin, clock):
    import email.utils

    date = email.utils.formatdate(clock.wall() + 4, usegmt=True)
    o = origin({"https://cdn.example.com/t.jpg": [(429, {"Retry-After": date}, b""), (200, {}, b"JPEG")]})
    _get("t.jpg")
    assert o.sleeps == [pytest.approx(4.0, abs=1.0)]


def test_without_retry_after_the_hold_backs_off_1_2_4_8_and_resets_on_success(origin, monkeypatch):
    o = origin({"https://cdn.example.com/x.jpg": [(429, {}, b"")] * 3,
                "https://cdn.example.com/y.jpg": [(429, {}, b""), (200, {}, b"Y")],
                "https://cdn.example.com/z.jpg": [(429, {}, b"")] * 3,
                "https://cdn.example.com/j.jpg": [(429, {}, b""), (200, {}, b"J")]})
    with pytest.raises(urllib.error.HTTPError):
        _get("x.jpg")
    assert o.sleeps == [1.0, 2.0]       # the third 429 set a 4 s hold that no attempt of x.jpg was left to wait out
    assert _get("y.jpg") == b"Y"
    assert o.sleeps == [1.0, 2.0, 4.0, 8.0], "y.jpg waited the 4 s left, then its own 429's 8 s"
    with pytest.raises(urllib.error.HTTPError):
        _get("z.jpg")                   # the success reset the backoff
    assert o.sleeps[4:] == [1.0, 2.0]


def test_the_backoff_hold_is_jittered_down_by_a_quarter(origin, monkeypatch):
    monkeypatch.setattr(originrule, "_jitter", lambda: -0.25)
    o = origin({"https://cdn.example.com/j.jpg": [(429, {}, b""), (200, {}, b"J")]})
    assert _get("j.jpg") == b"J"
    assert o.sleeps == [0.75]


def test_the_jittered_backoff_hold_never_passes_the_cap(origin, monkeypatch):
    monkeypatch.setattr(originrule, "_jitter", lambda: 0.25)
    o = origin({"https://cdn.example.com/x.jpg": [(429, {}, b"")] * 3,
                "https://cdn.example.com/y.jpg": [(429, {}, b""), (200, {}, b"Y")]})
    with pytest.raises(urllib.error.HTTPError):
        _get("x.jpg")
    assert _get("y.jpg") == b"Y"
    # 1.25, 2.5 inside x; then y waits x's 5.0, and its own 429 is the 4th step: 8 * 1.25 = 10 clamped to 8
    assert o.sleeps == [1.25, 2.5, 5.0, 8.0]
    assert max(o.sleeps) <= originrule._HOLD_MAX_S


def test_a_403_is_never_retried_and_halves_the_width(origin):
    o = origin({"https://cdn.example.com/cf.jpg": [(403, {}, b"error code: 1020"), (200, {}, b"never")]})
    with pytest.raises(urllib.error.HTTPError) as e:
        _get("cf.jpg")
    assert e.value.code == 403 and o.opened == ["https://cdn.example.com/cf.jpg"] and o.sleeps == []
    assert _width() == 16


@pytest.mark.parametrize("status", [404, 410])
def test_a_404_or_410_is_never_retried_and_leaves_the_width_alone(origin, status):
    o = origin({"https://cdn.example.com/gone.jpg": [(status, {}, b""), (200, {}, b"never")]})
    with pytest.raises(urllib.error.HTTPError) as e:
        _get("gone.jpg")
    assert e.value.code == status and len(o.opened) == 1 and o.sleeps == []
    assert _width() == 32


def test_a_dead_connection_halves_the_width(origin):
    origin({"https://cdn.example.com/r.jpg": [ConnectionResetError("reset")] * 3})
    with pytest.raises(ConnectionResetError):
        _get("r.jpg")
    assert _width() == 4, "each of the three failed attempts halved it: 32 -> 16 -> 8 -> 4"


def test_one_hosts_429_never_slows_another_host(origin, clock):
    o = origin({"https://cdn.example.com/t.jpg": [(429, {"Retry-After": "5"}, b""), (200, {}, b"JPEG")],
                "https://other.example.com/o.jpg": [(200, {}, b"OTHER")]})
    _get("t.jpg")
    assert o.sleeps == [5.0]
    hold_a = originrule._host_state("cdn.example.com").hold_until
    clock.t = hold_a - 4.0                       # A is still held for 4 s
    assert originrule.fetch("https://other.example.com/o.jpg", max_bytes=100, timeout_s=5) == b"OTHER"
    assert o.sleeps == [5.0], "host B waited for nothing"
    assert _width("other.example.com") == 32 and originrule._host_state("other.example.com").hold_until == 0.0


def test_a_refused_address_is_followed_by_the_next_vetted_address_of_the_host(origin):
    o = origin({"https://cdn.example.com/t.jpg": [(200, {}, b"JPEG")]})
    o.refuse = {PUBLIC}
    assert _get("t.jpg") == b"JPEG"
    assert o.connected == [("cdn.example.com", PUBLIC), ("cdn.example.com", V6)]


def test_two_tiles_to_one_host_reuse_one_connection_and_an_error_closes_it(origin):
    o = origin({"https://cdn.example.com/a.jpg": [(200, {}, b"A")],
                "https://cdn.example.com/b.jpg": [(200, {}, b"B")],
                "https://cdn.example.com/c.jpg": [(404, {}, b"")],
                "https://cdn.example.com/d.jpg": [(200, {}, b"D")]})
    assert (_get("a.jpg"), _get("b.jpg")) == (b"A", b"B")
    assert len(o.connected) == 1 and not o.conns[0].closed, "the second tile rode the first one's connection"
    with pytest.raises(urllib.error.HTTPError):
        _get("c.jpg")
    assert o.conns[0].closed and len(o.connected) == 1
    assert _get("d.jpg") == b"D"
    assert len(o.connected) == 2, "the connection that saw an error was not offered again"


def test_a_server_that_dropped_an_idle_connection_costs_a_reconnect_not_an_attempt(origin):
    o = origin({"https://cdn.example.com/a.jpg": [(200, {}, b"A")],
                "https://cdn.example.com/b.jpg": [ConnectionResetError("idle drop"), (200, {}, b"B")]})
    _get("a.jpg")
    assert _get("b.jpg") == b"B"
    assert len(o.connected) == 2 and o.conns[0].closed and _width() == 32


def test_a_request_keeps_the_connection_open_and_names_the_pod(origin):
    o = origin({"https://cdn.example.com/a.jpg": [(200, {}, b"A")]})
    _get("a.jpg")
    (headers,) = o.headers
    assert headers == {"User-Agent": "monty-pod-originrule/1"}


def test_a_hold_that_outlasts_the_deadline_raises_instead_of_waiting(origin, clock):
    o = origin({"https://cdn.example.com/t.jpg": [(429, {"Retry-After": "8"}, b"")] * 3})
    with pytest.raises(originrule.DeadlineExceeded):
        _get("t.jpg", deadline=clock.t + 5)
    assert o.sleeps == [] and len(o.opened) == 1


def test_a_deadline_already_passed_starts_no_attempt(origin, clock):
    o = origin({"https://cdn.example.com/t.jpg": [(200, {}, b"x")]})
    with pytest.raises(originrule.DeadlineExceeded):
        _get("t.jpg", deadline=clock.t)
    assert o.opened == [] and o.connected == []


def test_an_unresolvable_host_records_the_dns_phase(monkeypatch):
    import socket

    def nxdomain(*a, **k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(originrule.socket, "getaddrinfo", nxdomain)
    with pytest.raises(originrule.OriginRefused) as e:
        _REAL_RESOLVE("gone.example.com")
    f = e.value.origin_failure
    assert (f.phase, f.host, f.addr, f.family, f.exc) == ("dns", "gone.example.com", None, None, "OriginRefused")


def test_an_unresolvable_host_counts_as_egress_failed_and_a_policy_refusal_does_not(monkeypatch):
    """The dns phase is the network's; a private address or a credential-shaped url is policy and never counts."""
    import socket

    monkeypatch.setattr(originrule, "_EGRESS", {})
    monkeypatch.setattr(originrule, "resolve_addrs", _REAL_RESOLVE)

    def nxdomain(*a, **k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(originrule.socket, "getaddrinfo", nxdomain)
    with pytest.raises(originrule.OriginRefused) as e:
        originrule.fetch("https://gone.example.com/a.jpg", max_bytes=100, timeout_s=1)
    assert e.value.origin_failure.phase == "dns" and originrule.egress_failed(e.value) is True
    assert list(originrule._EGRESS["gone.example.com"]) == [True]

    monkeypatch.setattr(originrule.socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("10.0.0.7", 443))])
    for url in ("https://lan.example.com/a.jpg", "https://cdn.example.com/a.jpg?api_key=abcdefghijklmnop"):
        with pytest.raises(originrule.OriginRefused) as e:
            originrule.fetch(url, max_bytes=100, timeout_s=1)
        assert originrule.egress_failed(e.value) is None
    assert set(originrule._EGRESS) == {"gone.example.com"}, "a policy refusal is not an attempt on any host"


def test_the_egress_verdict_needs_twenty_attempts_ninety_percent_failed_and_every_host(monkeypatch):
    monkeypatch.setattr(originrule, "_EGRESS", {})
    for _ in range(19):
        originrule._record_egress("a.example.com", True)
    assert not originrule.host_dead("a.example.com"), "19 attempts is under the minimum"
    originrule._record_egress("a.example.com", True)
    assert originrule.host_dead("a.example.com") and originrule.egress_dead(["a.example.com"])
    for _ in range(40):
        originrule._record_egress("b.example.com", False)
    assert not originrule.egress_dead(["a.example.com", "b.example.com"]), "one healthy host: the network works"
    for _ in range(36):
        originrule._record_egress("c.example.com", True)
    for _ in range(4):                           # 4 successes in the last 40 -> 90 % failed: still dead
        originrule._record_egress("c.example.com", False)
    assert originrule.host_dead("c.example.com")
    originrule._record_egress("c.example.com", False)     # 5 in 40 -> 87.5 % failed: alive
    assert not originrule.host_dead("c.example.com")


def test_the_module_imports_only_the_standard_library():
    tree = ast.parse(Path(originrule.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import is a repo import"
            roots.add((node.module or "").split(".")[0])
    assert roots and roots <= set(sys.stdlib_module_names), roots - set(sys.stdlib_module_names)
