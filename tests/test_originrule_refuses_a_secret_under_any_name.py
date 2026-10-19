"""The pod's origin guard never accepts a media url the box's guard refuses for a secret shape (MISC-783).

The box (the engine's origin-binding guard) runs its secret scanner's shapes over every
query pair and over the url's scheme://host/path; the pod used to check credential NAMES only, so a key under
a neutral name (`ref=sk-…`), behind ';' or in a path parameter reached a rented pod. DNS is faked: no socket
is opened and no verdict depends on the network or the clock."""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import pytest

from podagent import originrule

PUBLIC = "93.184.215.14"
HOSTS = {"cdn.example-stock.test", "bucket.s3.us-east-1.amazonaws.com"}

SK = "sk-SyntheticValue0123456789abcdefghijklmn"
# Built from two pieces so this FILE never carries the contiguous AKIA+16 shape tests/test_leak_gate.py scans
# for — the ASSEMBLED value is what originrule._secret_shape (and the box's own shape scan) must still catch.
AKIA = "AKIA" + "0123456789ABCDEF"

# Every one of these the box refuses (proven below against the box's own rule), and the pod used to accept.
SECRET_URLS = [
    f"https://cdn.example-stock.test/a.jpg?w=640&ref={SK}",
    f"https://cdn.example-stock.test/a.jpg?w=640;X-Amz-Credential={AKIA}",
    f"https://cdn.example-stock.test/a.jpg?w=640;ref={SK}",
    f"https://cdn.example-stock.test/a.jpg;ref={SK}",
    f"https://cdn.example-stock.test/a;x={AKIA}/b.jpg",
    f"https://cdn.example-stock.test/{SK}/a.jpg",
    f"https://cdn.example-stock.test/a.jpg?u=ghp_Synthetic0123456789abcdefXYZ",
    f"https://cdn.example-stock.test/a.jpg?u=hf_Synthetic0123456789abcdefXYZ",
    "https://cdn.example-stock.test/a.jpg?u=ya29.Synthetic0123456789abcdefXYZ",
    "https://cdn.example-stock.test/a.jpg?u=GOCSPX-Synthetic0123",
    "https://cdn.example-stock.test/a.jpg?u=EAASynthetic0123456789abcdef",
    "https://cdn.example-stock.test/a.jpg?u=1//0gSynthetic0123456789abcdef",
    "https://cdn.example-stock.test/a.jpg?u=123456789:AASynthetic0123456789abcdefghijklmnop",
    f"https://cdn.example-stock.test/a.jpg?u=x%3B{SK}",
    "https://cdn.example-stock.test/a.jpg?myToken=Q7xZ9pL2mN4vB8kR3tW6yH1j",
]

PRESIGNED = ("https://bucket.s3.us-east-1.amazonaws.com/renders/a.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256"
             f"&X-Amz-Credential={AKIA}%2F20261019%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20261019T000000Z"
             "&X-Amz-Expires=3600&X-Amz-SignedHeaders=host"
             "&X-Amz-Signature=0a79d3c1f2e4b5a6978877665544332211ffeeddccbbaa009988776655443322")


@pytest.fixture(autouse=True)
def _fake_dns(monkeypatch):
    def resolve(host):
        if host not in HOSTS:
            raise originrule.OriginRefused(f"REFUSED: {host} does not resolve")
        return (PUBLIC,)
    monkeypatch.setattr(originrule, "resolve_addrs", resolve)


def _box_refuses_for_a_secret_shape(url: str) -> bool:
    """The box's rule, transcribed from the engine's origin-binding guard at the pinned commit (query split by
    parse_qsl — '&' only — then the shape scan on each pair and on scheme://host/path), over the vendored shapes."""
    parts = urlsplit(url)
    for name, value in parse_qsl(parts.query, keep_blank_values=True):
        if name.lower().startswith(originrule._SIGNING_PREFIXES):
            continue
        if originrule._secret_shape(f'{name} = "{value}"'):
            return True
    return originrule._secret_shape(urlunsplit((parts.scheme, parts.hostname or "", parts.path, "", ""))) is not None


@pytest.mark.parametrize("url", SECRET_URLS)
def test_a_secret_under_a_neutral_name_is_refused(url):
    assert _box_refuses_for_a_secret_shape(url), url   # the box refuses it: the pod must too
    with pytest.raises(originrule.OriginRefused) as e:
        originrule.check(url)
    # Refused BY NAME, and the refusal never echoes the secret it refused.
    assert "secret-shaped material (" in e.value.reason
    assert SK not in e.value.reason and AKIA not in e.value.reason


def test_a_presigned_url_is_still_accepted():
    assert originrule.check(PRESIGNED) == PRESIGNED
    assert originrule.check("https://cdn.example-stock.test/photos/a-b_c.jpg?w=640&h=480&fm=jpg#top") \
        == "https://cdn.example-stock.test/photos/a-b_c.jpg?w=640&h=480&fm=jpg#top"


# The vendored list equals the box's at engine commit 3d919bab156ebeab9b1f6695f3f0c3b898f83395 — literals copied
# from that commit's secret scanner (verbose-regex comments dropped; they do not change the pattern; the one
# provider-named rule id is renamed to its role, since this repo is public).
_BOX_SHAPES_AT_PIN = [
    ("tiktok",              r"\b(?:act|rft)\.[A-Za-z0-9!._-]{20,}"),
    ("google_oauth_secret", r"\bGOCSPX-[A-Za-z0-9_-]{10,}"),
    ("google_refresh",      r"(?<![\w/])1//[A-Za-z0-9_-]{20,}"),
    ("google_access",       r"\bya29\.[A-Za-z0-9_.-]{20,}"),
    ("meta",                r"\bEAA[A-Za-z0-9]{20,}"),
    ("instagram",           r"\bIGQ[A-Za-z0-9_-]{20,}"),
    ("chat_bot_token",      r"\b\d{8,12}:AA[A-Za-z0-9_-]{30,}"),
    ("huggingface",         r"\bhf_[A-Za-z0-9]{20,}"),
    ("github",              r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    ("vk",                  r"\bvk1\.a\.[A-Za-z0-9_-]{20,}"),
    ("openai_style",        r"\bsk-(?:ant-|or-)?[A-Za-z0-9_-]{20,}"),
    ("aws_access_key_id",   r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
]
_BOX_SECRET_NAME_AT_PIN = (
    r"""(?xi)((?:[A-Za-z_][A-Za-z0-9_.\[\]"'-]*)?(?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|"""
    r"""(?-i:[A-Z0-9_]KEY))[A-Za-z0-9_]*)\s*[:=]{1,2}\s*(?:r|f|b|u)?["'`]([^"'`\n]{20,})["'`]""")
_BOX_NOT_A_VALUE_AT_PIN = (
    r"(?i)\$\{|\$[a-z_]|%\(|%s\b|\{\{|<[a-z_]+>|os\.environ|getenv|process\.env|your[_-]|example|"
    r"placeholder|redact|xxxx|\.\.\.|(?-i:^[a-z0-9_./-]+$)")


def test_the_vendored_shapes_equal_the_boxs_at_the_pinned_commit():
    assert originrule.SECRET_SHAPES_SOURCE.endswith("@3d919bab156ebeab9b1f6695f3f0c3b898f83395")
    assert [(r, p.pattern) for r, p in originrule._SHAPES] == _BOX_SHAPES_AT_PIN
    assert re.sub(r"\s+", "", originrule._SECRET_NAME.pattern) == re.sub(r"\s+", "", _BOX_SECRET_NAME_AT_PIN)
    assert originrule._NOT_A_VALUE.pattern == _BOX_NOT_A_VALUE_AT_PIN
    assert (originrule._MIN_RUN, originrule._MIN_ENTROPY) == (16, 3.0)
