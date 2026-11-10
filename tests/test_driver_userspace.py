"""MISC-998: the last Vulkan rank brings the host driver's own graphics userspace. No network, no real
installer: a fake transport, a fake extractor and a fake clock stand in, so each named refusal and the
allowlist-only result are pinned deterministically."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from podagent import driver_userspace as du

V = "580.178.04"
RUN = b"fake-installer-bytes" * 50


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeResponse:
    def __init__(self, status: int, body: bytes, clock: Clock, tick: float, parts: int) -> None:
        self.status, self.length = status, len(body)
        self._body, self._clock, self._tick, self._parts = body, clock, tick, parts
        self.consumed = False
        self.closed = False

    def chunks(self):
        n = max(1, len(self._body) // self._parts)
        for i in range(0, len(self._body), n):
            self._clock.t += self._tick
            yield self._body[i:i + n]
        self.consumed = True

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, clock: Clock, *, files: dict[str, bytes], tick: float = 0.0, parts: int = 4) -> None:
        self.files, self.clock, self.tick, self.parts = files, clock, tick, parts
        self.urls: list[str] = []
        self.responses: list[FakeResponse] = []

    def __call__(self, url: str, timeout: float, headers=None):
        self.urls.append(url)
        name = url.rsplit("/", 1)[-1]
        resp = FakeResponse(200 if name in self.files else 404, self.files.get(name, b""),
                            self.clock, self.tick if name.endswith(".run") else 0.0, self.parts)
        self.responses.append(resp)
        return resp


def _files(variant: str = f"NVIDIA-Linux-x86_64-{V}-no-compat32.run", body: bytes = RUN, sha: str | None = None):
    sha = sha or hashlib.sha256(body).hexdigest()
    return {variant: body, f"{variant}.sha256sum": f"{sha}  {variant}\n".encode()}


class FakeExtractor:
    def __init__(self, *, rc: int = 0, err: str = "", skip: str | None = None, raises: Exception | None = None,
                 transport: FakeTransport | None = None) -> None:
        self.rc, self.err, self.skip, self.raises, self.transport = rc, err, skip, raises, transport
        self.calls: list[tuple[str, str, float]] = []
        self.target_was_empty = False
        self.archive_fully_read = False

    def __call__(self, run_file: str, target: str, timeout: float):
        self.calls.append((run_file, target, timeout))
        self.target_was_absent = not Path(target).exists()
        self.archive_fully_read = bool(self.transport) and self.transport.responses[-1].consumed
        if not self.target_was_absent:
            return 1, f"The directory '{target}' already exists. Please either move the existing directory out of the way, or specify a different directory with the '--target' option."
        if self.raises:
            raise self.raises
        if self.rc == 0:
            t = Path(target)
            t.mkdir()
            for name in du.allowlist(V):
                if name != self.skip:
                    (t / name).write_bytes(b"ELF" + name.encode())
            for decoy in (f"libGLX_nvidia.so.{V}", f"libnvidia-glcore.so.{V}", f"libnvidia-rtcore.so.{V}",
                          f"libnvidia-tls.so.{V}", "nvidia-installer"):
                (t / decoy).write_bytes(b"decoy")
            (t / "kernel").mkdir()
            (t / "nvidia_icd.json").write_text(
                '{"file_format_version":"1.0.1","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.4.312"}}')
        return self.rc, self.err


def _disk(free: int = 10**12):
    return lambda _p: SimpleNamespace(free=free)


def _fetch(tmp_path, transport, extractor, clock, **kw):
    return du.fetch_driver_userspace(V, tmp_path / "state", transport=transport, extractor=extractor,
                                     clock=clock, disk_usage=kw.pop("disk_usage", _disk()), **kw)


def _left_behind(tmp_path) -> list[str]:
    state = tmp_path / "state"
    return sorted(p.name for p in state.iterdir()) if state.exists() else []


def test_success_keeps_the_allowlist_alias_and_a_manifest_and_nothing_else(tmp_path):
    clock = Clock()
    transport = FakeTransport(clock, files=_files())
    extractor = FakeExtractor(transport=transport)
    res = _fetch(tmp_path, transport, extractor, clock)

    assert res.ok and res.reason == ""
    final = tmp_path / "state" / f"nvidia-{V}"
    assert res.lib_dir == str(final) and res.manifest == str(final / "nvidia_icd.json")
    # the graphics libraries, spelled out (the receipt's EGL path), not read back from the code under test
    assert {p.name for p in final.iterdir() if not p.name.startswith(("libnvidia-encode", "libnvcuvid"))} == {
        f"libEGL_nvidia.so.{V}", f"libnvidia-eglcore.so.{V}", f"libnvidia-glsi.so.{V}",
        f"libnvidia-glvkspirv.so.{V}", f"libnvidia-gpucomp.so.{V}", "libEGL_nvidia.so.0", "nvidia_icd.json"}
    assert os.readlink(final / "libEGL_nvidia.so.0") == f"libEGL_nvidia.so.{V}"
    assert not (final / f"libGLX_nvidia.so.{V}").exists()
    icd = json.loads((final / "nvidia_icd.json").read_text())["ICD"]
    assert icd == {"library_path": str(final / "libEGL_nvidia.so.0"), "api_version": "1.4.312"}
    assert os.path.isabs(icd["library_path"])
    assert extractor.target_was_absent and extractor.archive_fully_read
    assert _left_behind(tmp_path) == [f"nvidia-{V}"], "the .run and the extract dir are gone"


def test_a_complete_directory_is_reused_without_any_download(tmp_path):
    clock = Clock()
    first = _fetch(tmp_path, FakeTransport(clock, files=_files()), FakeExtractor(), clock)
    assert first.ok

    def no_network(url, timeout):
        raise AssertionError("a verified directory must not be fetched again")

    again = _fetch(tmp_path, no_network, FakeExtractor(rc=9), clock)
    assert again.ok and again.lib_dir == first.lib_dir


def test_an_incomplete_directory_is_replaced_not_trusted(tmp_path):
    clock = Clock()
    stale = tmp_path / "state" / f"nvidia-{V}"
    stale.mkdir(parents=True)
    (stale / f"libEGL_nvidia.so.{V}").write_bytes(b"partial")
    res = _fetch(tmp_path, FakeTransport(clock, files=_files()), FakeExtractor(), clock)
    assert res.ok
    assert {p.name for p in stale.iterdir()} == {
        *du.allowlist(V), "libEGL_nvidia.so.0", "libnvidia-encode.so.1", "libnvcuvid.so.1", "nvidia_icd.json"}


def test_success_keeps_the_video_libs_and_their_so1_aliases(tmp_path):
    """The loader asks for libnvidia-encode.so.1; a versioned file alone does not satisfy it."""
    clock = Clock()
    transport = FakeTransport(clock, files=_files())
    res = _fetch(tmp_path, transport, FakeExtractor(transport=transport), clock)
    assert res.ok
    final = tmp_path / "state" / f"nvidia-{V}"
    for stem in ("libnvidia-encode", "libnvcuvid"):
        real = final / f"{stem}.so.{V}"
        assert real.is_file() and not real.is_symlink() and real.stat().st_size > 0
        assert os.readlink(final / f"{stem}.so.1") == f"{stem}.so.{V}"
    assert not list(final.glob("libnvidia-opticalflow*")), "nothing opens opticalflow: it stays out"


def test_a_wrong_video_alias_fails_verification(tmp_path):
    clock = Clock()
    first = _fetch(tmp_path, FakeTransport(clock, files=_files()), FakeExtractor(), clock)
    assert first.ok
    final = tmp_path / "state" / f"nvidia-{V}"
    assert du._verified(final, V)
    (final / "libnvcuvid.so.1").unlink()
    os.symlink("libnvcuvid.so.0.0.0", final / "libnvcuvid.so.1")
    assert not du._verified(final, V)
    # a stale dir without the video aliases is replaced, not reused
    again_clock = Clock()
    transport = FakeTransport(again_clock, files=_files())
    again = _fetch(tmp_path, transport, FakeExtractor(), again_clock)
    assert again.ok and transport.urls, "a directory failing verification must be fetched again"
    assert os.readlink(final / "libnvcuvid.so.1") == f"libnvcuvid.so.{V}"


def test_a_missing_no_compat32_package_falls_back_to_the_full_run(tmp_path):
    clock = Clock()
    full = f"NVIDIA-Linux-x86_64-{V}.run"
    transport = FakeTransport(clock, files=_files(full))
    assert _fetch(tmp_path, transport, FakeExtractor(), clock).ok
    assert transport.urls[0].endswith(f"{V}-no-compat32.run.sha256sum")
    assert transport.urls[-1] == f"{du.BASE_URL}/{V}/{full}"


def test_both_packages_missing_is_package_unavailable(tmp_path):
    clock = Clock()
    res = _fetch(tmp_path, FakeTransport(clock, files={}), FakeExtractor(), clock)
    assert (res.ok, res.reason) == (False, du.DRIVER_PACKAGE_UNAVAILABLE)
    assert _left_behind(tmp_path) == []


def test_a_checksum_mismatch_refuses_before_the_installer_script_runs(tmp_path):
    clock = Clock()
    transport = FakeTransport(clock, files=_files(sha="0" * 64))
    extractor = FakeExtractor()
    res = _fetch(tmp_path, transport, extractor, clock)
    assert (res.ok, res.reason) == (False, du.CHECKSUM_MISMATCH)
    assert extractor.calls == [], "an unverified file is never executed"
    assert _left_behind(tmp_path) == []


def test_a_download_past_the_wall_is_a_download_timeout(tmp_path):
    clock = Clock()
    transport = FakeTransport(clock, files=_files(), tick=60.0)     # 2nd chunk lands past the download budget
    extractor = FakeExtractor()
    res = _fetch(tmp_path, transport, extractor, clock)
    assert (res.ok, res.reason) == (False, du.DOWNLOAD_TIMEOUT)
    assert extractor.calls == [] and _left_behind(tmp_path) == []
    assert res.cap == du.RANK_WALL_S and res.elapsed > 0


def test_a_download_below_the_stall_floor_is_a_download_timeout(tmp_path):
    clock = Clock()
    transport = FakeTransport(clock, files=_files(), tick=du.STALL_WINDOW_S + 1, parts=40)   # bytes/s ~ 0
    res = _fetch(tmp_path, transport, FakeExtractor(), clock, cap=10_000.0)
    assert (res.ok, res.reason) == (False, du.DOWNLOAD_TIMEOUT)
    assert "stalled" in res.detail and _left_behind(tmp_path) == []


def test_a_read_timeout_from_the_transport_is_a_download_timeout(tmp_path):
    clock = Clock()

    def hangs(url, timeout):
        raise TimeoutError("read timed out")

    res = _fetch(tmp_path, hangs, FakeExtractor(), clock)
    assert (res.ok, res.reason) == (False, du.DOWNLOAD_TIMEOUT)


def test_short_disk_refuses_before_a_single_byte_is_written(tmp_path):
    clock = Clock()
    transport = FakeTransport(clock, files=_files())
    extractor = FakeExtractor()
    res = _fetch(tmp_path, transport, extractor, clock, disk_usage=_disk(free=1000))
    assert (res.ok, res.reason) == (False, du.INSUFFICIENT_DISK)
    assert not transport.responses[-1].consumed and extractor.calls == []
    assert _left_behind(tmp_path) == []


def test_a_failing_extractor_is_extraction_failed_and_leaves_nothing(tmp_path):
    clock = Clock()
    res = _fetch(tmp_path, FakeTransport(clock, files=_files()), FakeExtractor(rc=1, err="boom"), clock)
    assert (res.ok, res.reason) == (False, du.EXTRACTION_FAILED)
    assert _left_behind(tmp_path) == []


def test_the_extract_target_does_not_exist_when_the_extractor_runs(tmp_path):
    clock = Clock()
    extractor = FakeExtractor()
    res = _fetch(tmp_path, FakeTransport(clock, files=_files()), extractor, clock)
    assert res.ok, "a fresh target is accepted; a pre-created one is refused rc 1 'already exists'"
    assert extractor.target_was_absent and len(extractor.calls) == 1
    rc, err = FakeExtractor()(str(tmp_path), str(tmp_path), 1.0)
    assert rc == 1 and "already exists" in err


def test_an_extractor_failure_keeps_its_stdout_tail_in_the_detail(tmp_path):
    script = tmp_path / "installer.run"
    script.write_text("echo \"The directory '$3' already exists.\"\nexit 1\n")
    rc, tail = du.default_extractor(str(script), str(tmp_path / "t"), 30.0)
    assert rc == 1 and "already exists" in tail


def test_an_extractor_timeout_is_extraction_failed_and_still_cleans_up(tmp_path):
    clock = Clock()
    extractor = FakeExtractor(raises=TimeoutError("extract exceeded 60s"))
    res = _fetch(tmp_path, FakeTransport(clock, files=_files()), extractor, clock)
    assert (res.ok, res.reason) == (False, du.EXTRACTION_FAILED)
    assert _left_behind(tmp_path) == []


def test_a_missing_unpack_tool_is_dependency_missing(tmp_path):
    clock = Clock()
    for extractor in (FakeExtractor(rc=127, err="xz: command not found"),
                      FakeExtractor(raises=FileNotFoundError(2, "no such file", "sh"))):
        res = _fetch(tmp_path, FakeTransport(clock, files=_files()), extractor, clock)
        assert (res.ok, res.reason) == (False, du.DEPENDENCY_MISSING)
    assert _left_behind(tmp_path) == []


def test_an_extracted_tree_missing_an_allowlist_file_keeps_nothing(tmp_path):
    clock = Clock()
    extractor = FakeExtractor(skip=f"libnvidia-glvkspirv.so.{V}")
    res = _fetch(tmp_path, FakeTransport(clock, files=_files()), extractor, clock)
    assert (res.ok, res.reason) == (False, du.EXTRACTION_FAILED)
    assert "libnvidia-glvkspirv" in res.detail
    assert _left_behind(tmp_path) == []


def test_a_version_that_is_not_a_version_never_builds_a_url(tmp_path):
    def no_network(url, timeout):
        raise AssertionError("no URL for an unvalidated version")

    for bad in ("../../etc", "580", "580.178.04; rm", ""):
        res = du.fetch_driver_userspace(bad, tmp_path / "state", transport=no_network, extractor=FakeExtractor())
        assert (res.ok, res.reason) == (False, du.DRIVER_VERSION_UNAVAILABLE)


# ---- MISC-1032: the run is fetched as parallel byte ranges --------------------------------------------------

@pytest.fixture(autouse=True)
def _fast_supervisor(monkeypatch):
    monkeypatch.setattr(du, "POLL_S", 0.01)       # real wake-up cadence only; every verdict uses the fake clock


class RangeResponse:
    """A ranged answer. Its virtual finish time is t0 + chunks*tick, so parallel streams overlap on the fake clock."""

    def __init__(self, status, body, clock, tick, chunk, content_range, length=None):
        self.status, self.body, self.clock, self.tick, self.chunk = status, body, clock, tick, chunk
        self.length = len(body) if length is None else length
        self.content_range, self.content_encoding = content_range, None
        self.t0 = 0.0                     # every stream starts together on the fake clock, whatever the thread order
        self.ignore_close = False
        self.gate = None                 # set -> chunks() blocks until the gate opens (close(), unless ignore_close)
        self.consumed = self.closed = self.exited = False

    def chunks(self):
        try:
            if self.gate is not None:
                self.gate.wait()          # no real-time bound: the gate opens on close() or the test's teardown
                return
            for k, i in enumerate(range(0, len(self.body), self.chunk), 1):
                self.clock.t = max(self.clock.t, self.t0 + k * self.tick)
                yield self.body[i:i + self.chunk]
            self.consumed = True
        finally:
            self.exited = True

    def close(self) -> None:
        self.closed = True
        if self.gate is not None and not self.ignore_close:
            self.gate.set()


class RangedTransport(FakeTransport):
    def __init__(self, clock, *, files, tick=0.0, chunk=50, ranges=True, tamper=None, parts=4):
        super().__init__(clock, files=files, tick=tick, parts=parts)
        self.chunk, self.ranges, self.tamper = chunk, ranges, tamper
        self.calls: list[dict | None] = []          # headers of every .run request, in order
        self.range_responses: list[RangeResponse] = []
        self._lock = threading.Lock()

    def __call__(self, url, timeout, headers=None):
        if not url.endswith(".run"):
            return super().__call__(url, timeout, headers)
        with self._lock:
            self.calls.append(headers)
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", (headers or {}).get("Range", ""))
        if not (m and self.ranges):
            return super().__call__(url, timeout, headers)
        s, e = int(m.group(1)), int(m.group(2))
        body = self.files[url.rsplit("/", 1)[-1]]
        resp = RangeResponse(206, body[s:e + 1], self.clock, self.tick, self.chunk, f"bytes {s}-{e}/{len(body)}")
        if (s, e) != (0, 0):
            with self._lock:
                self.range_responses.append(resp)
            if self.tamper:
                self.tamper(s, e, resp)
        return resp

    def ranged_calls(self):
        return [h["Range"] for h in self.calls if h and "Range" in h]

    def single_calls(self):
        return [h for h in self.calls if not h or "Range" not in h]


def _ranged_fetch(tmp_path, clock, **kw):
    fetch_kw = {k: kw.pop(k) for k in ("cap", "disk_usage") if k in kw}
    transport = RangedTransport(clock, files=kw.pop("files", _files()), **kw)
    extractor = FakeExtractor(transport=None)
    return transport, extractor, _fetch(tmp_path, transport, extractor, clock, **fetch_kw)


def test_ranges_finish_inside_the_wall_where_one_stream_would_not(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "MIN_BYTES_PER_S", 0)           # this test guards the wall; the stall has its own test
    parallel_clock = Clock()
    transport, extractor, res = _ranged_fetch(tmp_path, parallel_clock, tick=15.0)
    assert res.ok, res.detail
    assert res.transfer == f"bytes={len(RUN)}/{len(RUN)} streams=4"
    # probe, then four contiguous ranges covering [0, L) with no gap and no overlap
    assert transport.calls[0] == {"Range": "bytes=0-0", "Accept-Encoding": "identity"}
    spans = sorted(tuple(map(int, r[6:].split("-"))) for r in transport.ranged_calls()[1:])
    assert len(spans) == du.RANGE_STREAMS and spans[0][0] == 0 and spans[-1][1] == len(RUN) - 1
    assert all(b[0] == a[1] + 1 for a, b in zip(spans, spans[1:]))
    assert all(h["Accept-Encoding"] == "identity" for h in transport.calls if h)
    assert transport.single_calls() == [] and all(r.closed for r in transport.range_responses)
    assert parallel_clock.t > 0 and extractor.calls and _left_behind(tmp_path) == [f"nvidia-{V}"]

    # the same host and speed on ONE stream blows the 110 s download wall
    one_clock = Clock()
    _, _, single = _ranged_fetch(tmp_path / "one", one_clock, tick=15.0, ranges=False, parts=20)
    assert (single.ok, single.reason) == (False, du.DOWNLOAD_TIMEOUT) and "wall hit" in single.detail


@pytest.mark.parametrize("kind", ["status-200", "wrong-range", "encoded", "wrong-length", "short", "long", "reset"])
def test_a_bad_range_fails_then_one_single_stream_into_a_fresh_file_finishes(tmp_path, kind):
    bad_resp = []

    def tamper(s, e, resp):
        if s != len(RUN) // 4:                                  # exactly one range misbehaves
            return
        bad_resp.append(resp)
        if kind == "status-200":
            resp.status, resp.content_range = 200, None
        elif kind == "wrong-range":
            resp.content_range = f"bytes {s}-{e}/{len(RUN) + 1}"
        elif kind == "encoded":
            resp.content_encoding = "gzip"
        elif kind == "wrong-length":
            resp.length += 1
        elif kind == "short":
            resp.body = resp.body[:-10]
        elif kind == "long":
            resp.body = resp.body + b"x" * 10
        elif kind == "reset":
            raise OSError("connection reset")

    clock = Clock()
    transport = RangedTransport(clock, files=_files(), tamper=tamper)
    extractor = FakeExtractor()
    res = _fetch(tmp_path, transport, extractor, clock)
    assert res.ok, res.detail
    assert len(transport.single_calls()) == 1, "exactly ONE single-stream fallback"
    assert extractor.calls[0][0].endswith(".single"), "the fallback writes a fresh file name"
    assert res.transfer == f"bytes={len(RUN)}/{len(RUN)} streams=1"
    assert _left_behind(tmp_path) == [f"nvidia-{V}"]
    if kind in ("status-200", "wrong-range", "encoded", "wrong-length"):
        assert bad_resp and not bad_resp[0].exited, "refused before its body was read"
        assert bad_resp[0].closed


def test_a_global_stall_refuses_and_does_not_restart_as_one_stream(tmp_path):
    clock = Clock()
    transport, extractor, res = _ranged_fetch(tmp_path, clock, tick=3.0, chunk=10, cap=10_000.0)
    assert (res.ok, res.reason) == (False, du.DOWNLOAD_TIMEOUT) and "stalled" in res.detail
    assert transport.single_calls() == [] and extractor.calls == [] and _left_behind(tmp_path) == []
    assert all(r.closed for r in transport.range_responses)


def test_the_wall_refuses_without_a_fallback_and_the_trailer_says_what_arrived(tmp_path):
    clock = Clock()
    transport, extractor, res = _ranged_fetch(tmp_path, clock, tick=200.0)
    assert (res.ok, res.reason) == (False, du.DOWNLOAD_TIMEOUT) and "wall hit" in res.detail
    assert transport.single_calls() == [] and _left_behind(tmp_path) == []
    assert re.fullmatch(rf"bytes=\d+/{len(RUN)} streams=4", res.transfer)
    assert res.detail.endswith(res.transfer)


def test_a_blocked_worker_is_released_and_joined_before_the_refusal_returns(tmp_path):
    gated = {}

    def tamper(s, e, resp):
        if s == len(RUN) // 4:
            resp.gate = threading.Event()               # a read that never returns until the response is closed
            gated["resp"] = resp
        elif s == len(RUN) // 2:
            resp.tick = 500.0                           # this stream pushes the clock past the wall

    clock = Clock()
    transport, _, res = _ranged_fetch(tmp_path, clock, tamper=tamper)
    assert (res.ok, res.reason) == (False, du.DOWNLOAD_TIMEOUT) and "wall hit" in res.detail
    assert gated["resp"].closed and gated["resp"].exited, "the worker was woken and finished before we returned"
    assert not [t for t in threading.enumerate() if t.name.startswith("driver-range-")]
    assert transport.single_calls() == [] and _left_behind(tmp_path) == []


def test_a_worker_that_ignores_close_is_given_up_on_after_the_join_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "JOIN_DEADLINE_S", 0.05)
    gated = {}

    def tamper(s, e, resp):
        if s == len(RUN) // 4:
            resp.gate, resp.ignore_close = threading.Event(), True
            gated["resp"] = resp
        elif s == len(RUN) // 2:
            resp.tick = 500.0

    clock = Clock()
    try:
        transport, _, res = _ranged_fetch(tmp_path, clock, tamper=tamper)
    finally:
        gated["resp"].gate.set()                        # let the stuck worker finish after the assertions' subject
    assert (res.ok, res.reason) == (False, du.DOWNLOAD_TIMEOUT) and "did not stop in time" in res.detail
    assert gated["resp"].closed and transport.single_calls() == []


def test_a_host_that_ignores_ranges_gets_the_single_stream_as_before(tmp_path):
    clock = Clock()
    transport, extractor, res = _ranged_fetch(tmp_path, clock, ranges=False)
    assert res.ok
    assert transport.ranged_calls() == ["bytes=0-0"] and len(transport.single_calls()) == 1
    assert not extractor.calls[0][0].endswith(".single")
    assert res.transfer == f"bytes={len(RUN)}/{len(RUN)} streams=1"


def test_a_ranged_download_with_the_wrong_checksum_still_refuses_before_extraction(tmp_path):
    clock = Clock()
    transport, extractor, res = _ranged_fetch(tmp_path, clock, files=_files(sha="0" * 64))
    assert (res.ok, res.reason) == (False, du.CHECKSUM_MISMATCH)
    assert extractor.calls == [] and transport.single_calls() == [] and _left_behind(tmp_path) == []


def test_short_disk_refuses_on_the_probed_length_before_any_range_is_requested(tmp_path):
    clock = Clock()
    transport, extractor, res = _ranged_fetch(tmp_path, clock, disk_usage=_disk(free=1000))
    assert (res.ok, res.reason) == (False, du.INSUFFICIENT_DISK)
    assert transport.ranged_calls() == ["bytes=0-0"] and transport.range_responses == []
    assert _left_behind(tmp_path) == []


def test_a_full_disk_while_writing_is_insufficient_disk_not_a_fallback(tmp_path, monkeypatch):
    def full(fd, data, offset):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "pwrite", full)
    clock = Clock()
    transport, _, res = _ranged_fetch(tmp_path, clock)
    assert (res.ok, res.reason) == (False, du.INSUFFICIENT_DISK)
    assert transport.single_calls() == [] and _left_behind(tmp_path) == []


def test_a_short_positional_write_is_continued_at_the_right_offset(tmp_path, monkeypatch):
    real = os.pwrite
    monkeypatch.setattr(os, "pwrite", lambda fd, data, offset: real(fd, bytes(data)[:7], offset))
    clock = Clock()
    _, extractor, res = _ranged_fetch(tmp_path, clock)
    assert res.ok, res.detail
    assert extractor.calls and res.transfer == f"bytes={len(RUN)}/{len(RUN)} streams=4"


def test_both_refusal_trailers_carry_bytes_and_streams(monkeypatch):
    from podagent import main as pod_main

    refused = du.FetchResult(ok=False, reason=du.DOWNLOAD_TIMEOUT, detail="x", cap=150.0,
                             transfer="bytes=100/316359348 streams=4")
    monkeypatch.setattr(pod_main, "_fetch_driver_userspace", lambda *a, **k: refused)
    ok, note = pod_main._driver_userspace_rank("595.112.07")
    assert not ok and note.endswith("bytes=100/316359348 streams=4") and "DOWNLOAD_TIMEOUT" in note
    video = pod_main._provide_video_libs(None, "595.112.07", ["true"])
    assert video.endswith("bytes=100/316359348 streams=4") and "DOWNLOAD_TIMEOUT" in video


# ── MISC-1051: the userspace comes from our mirror first ─────────────────────────────────────────────────
def _mirror_tar(tmp_path, *, skip: str | None = None, link: str | None = None,
                icd_path: str = "/not/the/final/dir/libEGL.so") -> Path:
    import tarfile
    src = tmp_path / "mirror-src"
    src.mkdir()
    names = list(du.allowlist(V))
    for name in names:
        if name == link:                       # a link member inside the tar, pointing at a sibling
            os.symlink(next(n for n in names if n != link), src / name)
        elif name != skip:
            (src / name).write_bytes(b"ELF" + name.encode())
    (src / "nvidia_icd.json").write_text(json.dumps(
        {"file_format_version": "1.0.1", "ICD": {"library_path": icd_path, "api_version": "1.4.312"}}))
    out = tmp_path / "userspace.tar.xz"
    with tarfile.open(out, "w:xz") as tf:
        for p in sorted(src.iterdir()):
            tf.add(p, arcname=p.name)
    return out


def _ref(tar: Path, sha: str | None = None):
    data = tar.read_bytes()
    return SimpleNamespace(url=tar.as_uri(), sha256=sha or hashlib.sha256(data).hexdigest(), size=len(data))


def test_a_mirrored_version_is_provisioned_from_our_store_without_touching_nvidia(tmp_path):
    clock = Clock()
    transport = FakeTransport(clock, files=_files())
    extractor = FakeExtractor()
    ref = _ref(_mirror_tar(tmp_path))
    res = _fetch(tmp_path, transport, extractor, clock, mirror=lambda v: ref if v == V else None)

    assert res.ok and res.source == "mirror" and res.mirror == ""
    assert transport.urls == [] and extractor.calls == []
    final = tmp_path / "state" / f"nvidia-{V}"
    icd = json.loads((final / "nvidia_icd.json").read_text())["ICD"]
    assert icd == {"library_path": str(final / "libEGL_nvidia.so.0"), "api_version": "1.4.312"}
    assert os.readlink(final / "libEGL_nvidia.so.0") == f"libEGL_nvidia.so.{V}"
    assert _left_behind(tmp_path) == [f"nvidia-{V}"]


def test_a_mirror_miss_falls_back_to_nvidia_and_names_why(tmp_path):
    clock = Clock()
    transport = FakeTransport(clock, files=_files())
    res = _fetch(tmp_path, transport, FakeExtractor(), clock, mirror=lambda v: None)
    assert res.ok and res.source == "nvidia" and res.mirror == "not_mirrored"
    assert transport.urls


def _broken_ref(tmp_path, broken):
    if broken == "sha":
        return _ref(_mirror_tar(tmp_path), sha="0" * 64)
    if broken == "missing_lib":
        return _ref(_mirror_tar(tmp_path, skip=du.allowlist(V)[0]))
    if broken == "symlink_member":
        return _ref(_mirror_tar(tmp_path, link=du.allowlist(V)[0]))
    if broken == "corrupt_tar":
        bad = tmp_path / "userspace.tar.xz"
        bad.write_bytes(b"not a tar at all")
        return _ref(bad)
    if broken == "no_file":
        return SimpleNamespace(url=(tmp_path / "absent.tar.xz").as_uri(), sha256="0" * 64, size=1)
    return None


_BROKEN = ["sha", "missing_lib", "symlink_member", "corrupt_tar", "no_file", "raises"]


@pytest.mark.parametrize("broken", _BROKEN)
def test_a_broken_mirror_falls_back_to_nvidia(tmp_path, broken):
    clock = Clock()
    transport = FakeTransport(clock, files=_files())
    ref = _broken_ref(tmp_path, broken)

    def mirror(_v):
        if broken == "raises":
            raise RuntimeError("cp down")
        return ref

    res = _fetch(tmp_path, transport, FakeExtractor(), clock, mirror=mirror)
    assert res.ok and res.source == "nvidia" and res.mirror.startswith("error:")
    assert transport.urls
    assert _left_behind(tmp_path) == [f"nvidia-{V}"]


@pytest.mark.parametrize("broken", _BROKEN)
def test_a_mirror_that_burns_the_wall_leaves_nvidia_only_what_is_left(tmp_path, broken):
    """No double wall: the mirror attempt consumes 120 of the 150 s; the fallback must see 30 s left (below the
    40 s post-download reserve) and refuse without a single NVIDIA request."""
    clock = Clock()
    transport = FakeTransport(clock, files=_files())
    ref = _broken_ref(tmp_path, broken)

    def mirror(_v):
        clock.t = 120.0
        if broken == "raises":
            raise RuntimeError("cp down")
        return ref

    res = _fetch(tmp_path, transport, FakeExtractor(), clock, mirror=mirror)
    assert not res.ok and res.reason == du.DOWNLOAD_TIMEOUT
    assert res.mirror.startswith("error:") and transport.urls == []
    assert res.cap == du.RANK_WALL_S and res.elapsed == 120.0


def test_a_mirror_download_that_outlasts_the_wall_is_abandoned_not_waited_for(tmp_path, monkeypatch):
    """The mirror fetch itself is held to the wall: a download still running at t=120 (30 s left, under the
    40 s reserve) is abandoned as error:wall and the rank ends at that moment, not when the fetch returns."""
    clock = Clock()
    release = threading.Event()

    def hung_fetch(ref, dst, *a, **k):
        clock.t = 120.0
        release.wait(30)               # only a safety net: the supervisor must not wait for this
        raise OSError("late")

    monkeypatch.setattr(du.artifact, "fetch_verified", hung_fetch)
    transport = FakeTransport(clock, files=_files())
    try:
        res = _fetch(tmp_path, transport, FakeExtractor(), clock,
                     mirror=lambda v: SimpleNamespace(url="https://x/y", sha256="0" * 64, size=1))
    finally:
        release.set()
    assert not res.ok and res.reason == du.DOWNLOAD_TIMEOUT and res.mirror == "error:TimeoutError"
    assert transport.urls == [] and res.elapsed == 120.0 <= res.cap


def test_a_mirror_manifest_without_a_valid_api_version_falls_back_to_nvidia(tmp_path):
    import tarfile
    clock = Clock()
    src = tmp_path / "src"
    src.mkdir()
    for name in du.allowlist(V):
        (src / name).write_bytes(b"ELF")
    (src / "nvidia_icd.json").write_text(json.dumps({"ICD": {"library_path": "x"}}))
    tar = tmp_path / "userspace.tar.xz"
    with tarfile.open(tar, "w:xz") as tf:
        for p in sorted(src.iterdir()):
            tf.add(p, arcname=p.name)
    transport = FakeTransport(clock, files=_files())
    res = _fetch(tmp_path, transport, FakeExtractor(), clock, mirror=lambda v: _ref(tar))
    assert res.ok and res.source == "nvidia" and res.mirror.startswith("error:")


def test_a_failed_mirror_then_failed_nvidia_still_reports_the_mirror_token(tmp_path):
    clock = Clock()
    res = _fetch(tmp_path, FakeTransport(clock, files={}), FakeExtractor(), clock, mirror=lambda v: None)
    assert not res.ok and res.mirror == "not_mirrored"


def test_the_trailer_names_the_mirror_failure_only_when_a_mirror_was_asked(monkeypatch):
    from podagent import main as pod_main
    refused = du.FetchResult(ok=False, reason=du.DOWNLOAD_TIMEOUT, mirror="not_mirrored")
    monkeypatch.setattr(pod_main, "_fetch_driver_userspace", lambda *a, **k: refused)
    ok, note = pod_main._driver_userspace_rank("595.112.07")
    assert not ok and re.search(r"cap=\d+s mirror=not_mirrored$", note)
    plain = du.FetchResult(ok=False, reason=du.DOWNLOAD_TIMEOUT)
    monkeypatch.setattr(pod_main, "_fetch_driver_userspace", lambda *a, **k: plain)
    assert "mirror=" not in pod_main._driver_userspace_rank("595.112.07")[1]


@pytest.mark.parametrize("source", ["mirror", "nvidia"])
def test_the_ok_trailer_names_the_source(monkeypatch, tmp_path, source):
    from podagent import main as pod_main
    fetched = du.FetchResult(ok=True, lib_dir=str(tmp_path), manifest=str(tmp_path / "m.json"), source=source)
    monkeypatch.setattr(pod_main, "_fetch_driver_userspace", lambda *a, **k: fetched)
    monkeypatch.setattr(pod_main, "_run_vulkan_probe", lambda timeout=120: (True, "ok", b""))
    monkeypatch.setattr(pod_main, "_log_vulkan_rank", lambda *a, **k: None)
    monkeypatch.setattr(pod_main, "_prepend_ld_library_path", lambda _d: None)
    for key in pod_main._VULKAN_ENV_KEYS:          # the rank writes these on green; monkeypatch restores them
        monkeypatch.setenv(key, "x")
    ok, note = pod_main._driver_userspace_rank("595.112.07")
    assert ok and note.startswith(f"driver_userspace=ok source={source} elapsed=")
