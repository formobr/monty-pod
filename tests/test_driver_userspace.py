"""MISC-998: the last Vulkan rank brings the host driver's own graphics userspace. No network, no real
installer: a fake transport, a fake extractor and a fake clock stand in, so each named refusal and the
allowlist-only result are pinned deterministically."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

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

    def __call__(self, url: str, timeout: float):
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
    # the brief's five libraries, spelled out (the receipt's EGL path), not read back from the code under test
    assert {p.name for p in final.iterdir()} == {
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
    assert {p.name for p in stale.iterdir()} == {*du.allowlist(V), "libEGL_nvidia.so.0", "nvidia_icd.json"}


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
