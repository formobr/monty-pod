"""A presigned PUT's deadline is sized to the object and the uplink it shares, and a 4xx is never re-sent.

Prod 2026-10-01: three concurrent previews spent 680/811/460 s in `put` because every object under 30 MiB
got a flat 30 s read window, timed out on a shared uplink and was re-sent from byte 0 — and an expired
presign (403) was retried with backoff although it can only fail again.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

from podagent import cp

_URL = ("https://r2.example/object?X-Amz-Credential=AKIAEXAMPLE%2F20260829%2Fauto%2Fs3%2Faws4_request"
        "&X-Amz-Signature=bb459aa8161dac7d2e80030516e882519b6b9beccbfc141f9f4123d56f0dc6a6")
_BUSY_POD_TRANSFERS = 64


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error for url: {_URL}", response=self)


class _Link:
    """A slow-but-steady uplink: an attempt whose read window is shorter than size/rate times out, exactly
    like a socket send blocked on a shared uplink. Answers with `statuses` in order once the body is sent."""

    def __init__(self, rate_bytes_per_s: float, statuses: list[int] | None = None) -> None:
        self.rate = rate_bytes_per_s
        self.statuses = list(statuses or [200])
        self.timeouts: list[Any] = []

    def put(self, url: str, data: Any, headers: dict[str, str], timeout: Any) -> _Resp:
        self.timeouts.append(timeout)
        read_s = timeout[1] if isinstance(timeout, tuple) else timeout
        needed_s = len(data) / self.rate
        if read_s < needed_s:
            raise requests.exceptions.ReadTimeout(f"read timed out ({read_s:.0f}s < {needed_s:.0f}s)")
        sent = sum(len(chunk) for chunk in data)
        assert sent == len(data)
        return _Resp(self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0])


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(cp.time, "sleep", calls.append)
    monkeypatch.setattr(cp, "_PUT_FLOOR_BYTES_PER_S", cp._MEASURED_STREAM_BYTES_PER_S / _BUSY_POD_TRANSFERS)
    cp.put_rate.reset()
    cp.put_trace.reset()
    yield calls
    cp.put_rate.reset()


def _obj(tmp_path: Path, size: int) -> Path:
    src = tmp_path / "out.mp4"
    with src.open("wb") as f:
        f.truncate(size)
    return src


def test_put_deadline_scales_and_4xx_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    sleeps: list[float]) -> None:
    # (1) 40 MiB on a busy pod's slice of the uplink (~0.38 MB/s, ~110 s of wire) completes in ONE attempt —
    # the old flat `max(30, size // 1 MiB)` = 40 s window cut it off and re-sent it from byte 0.
    size = 40 << 20
    link = _Link(rate_bytes_per_s=0.38e6)
    monkeypatch.setattr(cp, "_store", link)
    old_window = max(cp._TIMEOUT, size // (1 << 20))
    assert old_window < size / link.rate, "the scenario must be one the flat window could not survive"
    cp.upload(_obj(tmp_path, size), _URL)
    assert len(link.timeouts) == 1 and sleeps == []
    assert [r["outcome"] for r in cp.put_trace.collect()] == ["ok"]

    # The deadline grows with size, and with how many PUTs share the measured uplink right now.
    cp.put_rate.reset()
    assert cp._put_deadline_s(400 << 20) > cp._put_deadline_s(40 << 20) > cp._put_deadline_s(4 << 20)
    cp.put_rate.end(100 << 20, 4.0)  # one PUT alone measured 25 MiB/s
    alone = cp._put_deadline_s(size)
    for _ in range(16):
        cp.put_rate.begin()
    shared = cp._put_deadline_s(size)
    assert shared > alone
    assert shared == pytest.approx(
        cp._PUT_CONNECT_MARGIN_S + size / ((100 << 20) / 4.0 / 16 / cp._PUT_RATE_SLACK))
    cp.put_rate.reset()

    # (2) An expired presign (403) fails after ONE attempt, no sleep, status named, no url leaked.
    cp.put_trace.reset()
    link = _Link(rate_bytes_per_s=100e6, statuses=[403])
    monkeypatch.setattr(cp, "_store", link)
    with pytest.raises(cp.PutRejected, match="HTTP 403") as exc:
        cp.upload(_obj(tmp_path, 1 << 20), _URL)
    assert isinstance(exc.value, requests.HTTPError)
    assert "Signature" not in str(exc.value)
    assert len(link.timeouts) == 1 and sleeps == []
    assert [r["outcome"] for r in cp.put_trace.collect()] == ["error"]

    # 408/429 are the 4xx answers that DO mean "try again".
    for status in (408, 429):
        sleeps.clear()
        link = _Link(rate_bytes_per_s=100e6, statuses=[status, 200])
        monkeypatch.setattr(cp, "_store", link)
        cp.upload(_obj(tmp_path, 1 << 20), _URL)
        assert len(link.timeouts) == 2 and sleeps == [1]

    # (3) A 503 keeps the bounded retry: every attempt, 2**attempt backoff, then the error.
    sleeps.clear()
    cp.put_trace.reset()
    link = _Link(rate_bytes_per_s=100e6, statuses=[503])
    monkeypatch.setattr(cp, "_store", link)
    with pytest.raises(requests.HTTPError) as exc:
        cp.upload(_obj(tmp_path, 1 << 20), _URL)
    assert not isinstance(exc.value, cp.PutRejected)
    assert len(link.timeouts) == cp._XFER_ATTEMPTS
    assert sleeps == [2**a for a in range(cp._XFER_ATTEMPTS - 1)]
    assert [r["outcome"] for r in cp.put_trace.collect()] == ["retry", "retry", "error"]


def test_a_timeout_keeps_the_bounded_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           sleeps: list[float]) -> None:
    link = _Link(rate_bytes_per_s=1.0)  # nothing fits any deadline
    monkeypatch.setattr(cp, "_store", link)
    with pytest.raises(requests.exceptions.ReadTimeout):
        cp.upload(_obj(tmp_path, 1 << 20), _URL)
    assert len(link.timeouts) == cp._XFER_ATTEMPTS
    assert sleeps == [1, 2]
    assert cp.put_rate.per_put() == 0.0  # failed attempts never pose as a measured rate


def test_floor_is_the_measured_stream_split_across_the_transfer_width() -> None:
    assert cp._MEASURED_STREAM_BYTES_PER_S == 25e6
    assert cp._PUT_FLOOR_BYTES_PER_S == pytest.approx(cp._MEASURED_STREAM_BYTES_PER_S / cp._store_pool())
