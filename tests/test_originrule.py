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
}


@pytest.fixture(autouse=True)
def _fake_dns(monkeypatch):
    def resolve(host):
        if host not in DNS:
            raise originrule.OriginRefused(f"REFUSED: {host} does not resolve")
        return DNS[host]
    monkeypatch.setattr(originrule, "resolve_addrs", resolve)


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
        self.script = {u: list(v) for u, v in script.items()}
        self.opened: list[str] = []
        self.connected: list[tuple[str, str]] = []

    def connect(self, host: str, addr: str, *, timeout_s: float):
        self.connected.append((host, addr))
        origin, origin_host = self, host

        class _FakeConnection:
            def request(self, method, path, headers=None):
                url = f"https://{origin_host}{path}"
                origin.opened.append(url)
                status, headers_, body = origin.script[url].pop(0)
                self._resp = _FakeResponse(status, headers_, body)

            def getresponse(self):
                return self._resp

            def close(self):
                pass

        return _FakeConnection()


@pytest.fixture
def origin(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(originrule, "_sleep", sleeps.append)

    def install(script):
        o = _Origin(script)
        o.sleeps = sleeps
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
def test_a_transient_status_is_retried_once(origin, status):
    o = origin({
        "https://cdn.example.com/t.jpg": [(status, {}, b""), (200, {}, b"JPEG")],
        "https://cdn.example.com/twice.jpg": [(status, {}, b""), (status, {}, b""), (200, {}, b"never")],
        "https://cdn.example.com/404.jpg": [(404, {}, b""), (200, {}, b"never")],
    })
    assert originrule.fetch("https://cdn.example.com/t.jpg", max_bytes=100, timeout_s=5) == b"JPEG"
    assert o.sleeps == [originrule._ORIGIN_RETRY_PAUSE_S]
    # Once, not more: a second transient status is the answer.
    with pytest.raises(urllib.error.HTTPError) as e:
        originrule.fetch("https://cdn.example.com/twice.jpg", max_bytes=100, timeout_s=5)
    assert e.value.code == status
    assert o.opened.count("https://cdn.example.com/twice.jpg") == 2
    # A verdict on the request is not retried.
    with pytest.raises(urllib.error.HTTPError):
        originrule.fetch("https://cdn.example.com/404.jpg", max_bytes=100, timeout_s=5)
    assert o.opened.count("https://cdn.example.com/404.jpg") == 1


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
