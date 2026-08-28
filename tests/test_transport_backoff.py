"""codex#25: a broken durable voice must retry bounded, then exit honestly — never spin forever and never
die silently on an ambiguous (DeliveryPending) verdict the durable sender is still retrying in the background."""
from __future__ import annotations

import pytest
from podagent import main as agent_main
from podagent.event_stream import DeliveryPending, TransportUnhealthy


class _AlwaysRaises:
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.calls = 0

    def poll_job(self):
        self.calls += 1
        raise self._error


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_main, "_TRANSPORT_UNHEALTHY_BACKOFF_S", 0.0)
    monkeypatch.setattr(agent_main.time, "sleep", lambda _s: None)


def test_delivery_pending_retries_forever_and_never_exits(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")
    calls_before_bail = 50

    class _StopAfterN(_AlwaysRaises):
        def poll_job(self):
            if self.calls >= calls_before_bail:
                raise SystemExit(0)  # test-only escape hatch; the production loop never returns
            return super().poll_job()

    cp = _StopAfterN(DeliveryPending("startup replay must clear before admitting work"))
    with pytest.raises(SystemExit) as excinfo:
        agent_main._dispatch_loop(cp, None, None, None, lambda _j: None)
    assert excinfo.value.code == 0, "DeliveryPending alone must never trigger the transport-unhealthy exit"
    assert cp.calls == calls_before_bail


def test_transport_unhealthy_backs_off_then_exits_honestly_within_the_cap(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")
    monkeypatch.setattr(agent_main, "_TRANSPORT_UNHEALTHY_MAX_ATTEMPTS_DEFAULT", 3)
    cp = _AlwaysRaises(TransportUnhealthy("durable append failed: disk full"))

    with pytest.raises(SystemExit) as excinfo:
        agent_main._dispatch_loop(cp, None, None, None, lambda _j: None)

    assert excinfo.value.code == 4, "a bounded-exhausted transport failure must exit with the honest code"
    assert cp.calls == 4, "3 retried attempts plus the one that trips the cap"
    assert not (tmp_path / "podagent.alive").exists(), "an honest exit clears the liveness mark like any stop"


def test_transport_unhealthy_attempt_count_resets_after_a_healthy_poll(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")
    monkeypatch.setattr(agent_main, "_TRANSPORT_UNHEALTHY_MAX_ATTEMPTS_DEFAULT", 2)

    class _RecoversOnce:
        def __init__(self) -> None:
            self.calls = 0

        def poll_job(self):
            self.calls += 1
            if self.calls == 3:
                return None  # one healthy poll resets the attempt counter
            if self.calls > 5:
                raise SystemExit(0)
            raise TransportUnhealthy("durable append failed: disk full")

    cp = _RecoversOnce()
    with pytest.raises(SystemExit) as excinfo:
        agent_main._dispatch_loop(cp, None, None, None, lambda _j: None, once=False)
    # Without the reset, attempts 1-2 would already have tripped the cap=2 exit before the healthy poll.
    assert excinfo.value.code == 0
