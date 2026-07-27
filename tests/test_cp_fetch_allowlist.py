"""cp.download's transport allowlist — the pod dereferences CAPABILITIES, not addresses. It was a bare
requests.get on whatever a binding named, fine while every url came from the control plane; origin urls now
come out of third-party search responses, so that class stopped being ours to bound."""
from __future__ import annotations

import pytest

from podagent import cp

PRESIGNED = ("https://acct.r2.cloudflarestorage.com/bucket/work/fleet/s/cut.mp4"
             "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=deadbeef&X-Amz-Expires=3600")


@pytest.mark.parametrize("url", [
    PRESIGNED,
    "https://minio.local:9000/b/k?X-Amz-Signature=abc",
    "https://storage.googleapis.com/b/k?X-Goog-Signature=abc",
    "https://s3.example/b/k?AWSAccessKeyId=A&Signature=s&Expires=1",
    "file:///var/lib/monty/work/x.mp4",
])
def test_every_url_the_pod_legitimately_fetches_still_passes(url):
    """Every store url is generate_presigned_url and every local-backend url is file:// — so this rule
    needs no deployment variable and cannot be wrong about an operator's own bucket host."""
    assert cp.assert_fetchable(url) == url


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "https://videos.pexels.com/video-files/1/a.mp4",
    "https://raw.githubusercontent.com/x/y/z.sh",
    "https://acct.r2.cloudflarestorage.com/bucket/k",
])
def test_a_bare_address_is_refused_however_innocent_it_looks(url):
    """NEGATIVE. Drop assert_fetchable from download() and the last of these — an UNSIGNED url on our own
    bucket host — shows why a host-name allowlist would not have been enough."""
    with pytest.raises(cp.UrlNotAllowed):
        cp.assert_fetchable(url)


def test_the_refusal_names_where_an_origin_url_belongs_instead():
    """A refusal a caller cannot act on becomes a --no-verify. This one says: params, not a binding."""
    with pytest.raises(cp.UrlNotAllowed) as e:
        cp.assert_fetchable("https://videos.pexels.com/a.mp4")
    assert "stock_hosts" in str(e.value) and "params" in str(e.value)


def test_download_refuses_before_it_opens_a_socket(tmp_path, monkeypatch):
    """The check must run BEFORE requests, or the SSRF is already delivered by the time we complain."""
    monkeypatch.setattr(cp.requests, "get",
                        lambda *a, **k: pytest.fail("a socket was opened for a refused url"))
    with pytest.raises(cp.UrlNotAllowed):
        cp.download("https://169.254.169.254/latest/meta-data/", tmp_path / "x")
    assert not (tmp_path / "x").exists()
