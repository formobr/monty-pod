from __future__ import annotations

from podagent.sanitize import safe_endpoint, safe_error, safe_text


def test_presigned_urls_userinfo_and_bearers_never_survive_diagnostics() -> None:
    secret = "top-secret-signature"
    value = (
        f"GET https://user:pass@store.example/work/a.mp4?X-Amz-Signature={secret} "
        f"Authorization: Bearer {secret}")
    cleaned = safe_text(value)
    assert cleaned == "GET [redacted-url] Authorization: Bearer [REDACTED]"
    assert secret not in cleaned and "user:pass" not in cleaned


def test_exception_args_are_scrubbed_before_they_can_become_result_errors() -> None:
    secret = "private-token"
    cleaned = safe_error(RuntimeError(
        f"failed https://u:p@store.example/o?token={secret} bearer {secret}"))
    assert secret not in cleaned
    assert "[redacted-url]" in cleaned and "Bearer [REDACTED]" in cleaned


def test_socket_endpoint_keeps_attribution_but_drops_credentials() -> None:
    endpoint = safe_endpoint("wss://user:pass@cp.example:8443/pod/stream?token=secret#fragment")
    assert endpoint == "cp.example:8443/pod/stream"
