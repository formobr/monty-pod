"""cp.download's transport allowlist — the pod dereferences CAPABILITIES, not addresses. It was a bare
requests.get on whatever a binding named, fine while every url came from the control plane; origin urls now
come out of third-party search responses, so that class stopped being ours to bound."""
from __future__ import annotations

import pytest

from podagent import cp

# The real shape both in-repo presigners emit — Go's r2.go:70,76,133,211,228 and Python's
# store_s3.py:127,149,256 — never an invented spelling (docs/TESTING.md §3b.3).
PRESIGNED = ("https://acct.r2.cloudflarestorage.com/bucket/work/fleet/s/cut.mp4"
             "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
             "&X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request"
             "&X-Amz-Date=20260829T000000Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host"
             "&X-Amz-Signature=0a79a76965ec9f93e27e38d15ddf76978aae86c8450da425ebffe684e062bbc0")
# The local contour's own MinIO (dev/localpod/.env.example:14) — plain http on loopback, a real
# production path per decisions.yaml `one-local-mode-and-it-is-production`.
MINIO_LOCAL = ("http://127.0.0.1:9000/monty/work/fleet/s/cut.mp4"
               "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
               "&X-Amz-Credential=minioadmin%2F20260829%2Fauto%2Fs3%2Faws4_request"
               "&X-Amz-Date=20260829T000000Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host"
               "&X-Amz-Signature=9470cdf79b72213d658f8cfa0ea21df45d008cbba833c5767180fc25cc20e17e")


@pytest.mark.parametrize("url", [
    PRESIGNED,
    MINIO_LOCAL,
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


# ── F4/C4: a query-parameter NAME is not a signature ────────────────────────────────────────────────

def test_a_bare_parameter_name_that_merely_names_a_signature_scheme_is_refused():
    """THE audit's exact bypass: `_SIGNED_NAMES` used to contain bare "token", so a metadata-endpoint url
    that merely NAMES that parameter walked straight through `_is_presigned`."""
    with pytest.raises(cp.UrlNotAllowed):
        cp.assert_fetchable("http://169.254.169.254/latest/meta-data/?token=1")


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/?expires=1",
    "https://evil.example/x?x-amz-foo=1",
])
def test_a_prefix_or_bare_name_match_is_not_a_signature(url):
    """A NAME match — bare or merely prefixed — proves nothing; only the real signed-url SHAPE does."""
    with pytest.raises(cp.UrlNotAllowed):
        cp.assert_fetchable(url)


def test_download_refuses_a_redirect_hop_instead_of_following_it(tmp_path, monkeypatch):
    """`allow_redirects=True` (the old default) means the allowlist only ever sees the FIRST url — a
    presigned GET that 302s to link-local metadata must be refused at the hop, no real socket involved."""
    calls: list[tuple[str, dict]] = []

    class _Redirect:
        status_code = 302
        headers = {"Location": "http://169.254.169.254/"}

        def close(self) -> None:
            pass

    def _fake_get(url, **kwargs):
        calls.append((url, kwargs))
        assert kwargs.get("allow_redirects") is False, "must not auto-follow redirects"
        return _Redirect()

    monkeypatch.setattr(cp._store, "get", _fake_get)
    dest = tmp_path / "x"
    with pytest.raises(cp.UrlNotAllowed):
        cp.download(PRESIGNED, dest)
    assert not dest.exists()
    assert len(calls) == 1, f"the redirect hop must never be dereferenced, got calls={calls}"


def test_upload_refuses_a_file_url_when_this_process_is_marked_a_rented_pod(tmp_path, monkeypatch):
    """`upload()` called no guard at all and honoured `file://` outright — on a REAL pod (marked so by its
    own entrypoint, `cp.mark_rented_pod()`, never by ambient env) that is an arbitrary local file write."""
    monkeypatch.setattr(cp, "_RENTED_POD", True)
    monkeypatch.setattr(cp.shutil, "copyfile",
                        lambda *a, **k: pytest.fail("a file was written for a refused url"))
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    with pytest.raises(cp.UrlNotAllowed):
        cp.upload(src, "file:///etc/cron.d/evil")


def test_upload_refuses_a_bare_address_before_it_puts(tmp_path, monkeypatch):
    """`upload()` must run the SAME allowlist `download()` does — a bare address is not a capability."""
    monkeypatch.setattr(cp._store, "put",
                        lambda *a, **k: pytest.fail("a socket was opened for a refused url"))
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    with pytest.raises(cp.UrlNotAllowed):
        cp.upload(src, "http://169.254.169.254/x")


# ── P1: identity is an EXPLICIT call, not an ambient env sniff ──────────────────────────────────────

def test_mark_rented_pod_sets_the_flag_and_nothing_else_does(monkeypatch):
    monkeypatch.setattr(cp, "_RENTED_POD", False)
    assert cp._is_real_pod() is False
    cp.mark_rented_pod()
    assert cp._is_real_pod() is True


def test_a_stray_job_token_in_the_shell_does_not_mark_a_local_render_a_pod(tmp_path, monkeypatch):
    """THE point of the change: an operator's leftover JOB_TOKEN, inherited by a `--local` launch, must
    NOT turn a legitimate file:// binding into a refusal — only `podagent.main`'s own boot read may."""
    monkeypatch.setenv("JOB_TOKEN", "stale-shell-export-not-a-pod")
    monkeypatch.setattr(cp, "_RENTED_POD", False)
    src = tmp_path / "out.bin"; src.write_bytes(b"rendered")
    target = tmp_path / "master" / "final.bin"
    cp.upload(src, target.as_uri())  # must NOT raise
    assert target.read_bytes() == b"rendered"
