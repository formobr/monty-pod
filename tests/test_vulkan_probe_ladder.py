"""_vulkan_preflight's 3-rank ladder: a synthesized ICD pinned to ONE driver used to clobber every other
driver's real ICD and denylist a healthy host (the-gpu-verdict-carries-its-evidence-or-it-is-not-a-verdict)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from podagent import driver_userspace as du
from podagent import main as agent_main
from test_nvenc_preflight import _CP, _run  # noqa: E402

_ICD_NAME = Path(agent_main._VULKAN_EGL_ICD_PATH).name   # taken from the module, captured before any redirect


def _red_fetch(version, base_dir, **_k):
    return du.FetchResult(ok=False, reason=du.DRIVER_PACKAGE_UNAVAILABLE, detail="no-network stub")


@pytest.fixture(autouse=True)
def _isolated_icd_env(monkeypatch, tmp_path):
    for key in agent_main._VULKAN_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # the last rank downloads from the vendor: every test starts with a no-network red stub, and only the
    # tests about that rank swap in their own
    monkeypatch.setattr(agent_main, "_fetch_driver_userspace", _red_fetch)
    # the refusal path calls _mark_stopped(), which unlinks the real liveness mark
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")
    monkeypatch.setattr(agent_main, "_VULKAN_EGL_ICD_PATH", str(tmp_path / _ICD_NAME))


def _ldconfig_ok(cmd, *_a, **_k):
    return subprocess.CompletedProcess(
        cmd, 0, stdout="libEGL_nvidia.so.0 (x86-64) => /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0\n",
        stderr="")


def _refused(cp=None) -> str:
    """Run the preflight to its refusal (owner 2026-10-27: an exhausted ladder exits 3) and return the
    boot error event's step."""
    cp = cp or _CP()
    with pytest.raises(SystemExit) as e:
        agent_main._vulkan_preflight(cp)
    assert e.value.code == 3
    errs = [ev for ev in cp.events if ev.get("status") == "error"]
    assert len(errs) == 1 and errs[0]["phase"] == "work_finished"
    return errs[0]["step"]


def _preflight_with_stubbed_gates(monkeypatch, cp, capacity=None):
    monkeypatch.setattr(agent_main, "_report_boot", lambda _cp: None)
    monkeypatch.setattr(agent_main, "_nvenc_or_refuse", lambda _cp: None)
    monkeypatch.setattr(agent_main, "_free_vram_or_refuse", lambda _cp: None)
    monkeypatch.setattr(agent_main, "_nvdec_or_refuse", lambda _cp: None)
    monkeypatch.delenv("MONTY_OPS_CONTOUR_DRY", raising=False)
    agent_main._capability_preflight(cp, capacity=capacity)


def test_prod_2026_10_27_pod_2359813_with_every_rank_red_refuses_before_ready(monkeypatch):
    """PROD 2026-10-27 rented pod 2359813: Vulkan would not open yet the pod sent ready and was routed work."""
    monkeypatch.setattr(subprocess, "run", _all_ranks_lose)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    cp = _CP()
    capacity: dict = {}
    with pytest.raises(SystemExit) as e:
        _preflight_with_stubbed_gates(monkeypatch, cp, capacity)
    assert e.value.code == 3
    errs = [ev for ev in cp.events if ev.get("status") == "error"]
    assert len(errs) == 1 and errs[0]["stage"] == "boot" and errs[0]["phase"] == "work_finished"
    assert errs[0]["step"].startswith("REFUSING work: vulkan_false — ")
    assert "ranks_tried=default,egl-1.3.0,egl-1.2.0" in errs[0]["step"]
    assert "driver_version=570.86.10" in errs[0]["step"]
    assert not [ev for ev in cp.events if ev.get("phase") == "ready"], "a refusing pod must never say ready"
    assert cp.announce_calls == 0
    assert "vulkan" not in capacity


def test_default_green_and_egl_1_3_green_still_send_ready_with_vulkan_true(monkeypatch):
    for first_rank_green in (True, False):
        calls = {"ffmpeg": 0}

        def fake_run(cmd, *a, **k):
            if cmd[0] == "ldconfig":
                return _ldconfig_ok(cmd)
            calls["ffmpeg"] += 1
            if first_rank_green or calls["ffmpeg"] > 1:
                return _run(0)
            return _run(1, b"VK_ERROR_INCOMPATIBLE_DRIVER\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        cp = _CP()
        capacity: dict = {}
        _preflight_with_stubbed_gates(monkeypatch, cp, capacity)
        assert capacity["vulkan"] is True
        assert [ev["phase"] for ev in cp.events if ev.get("phase") == "ready"] == ["ready"]
        assert not [ev for ev in cp.events if ev.get("status") == "error"]


def test_default_discovery_green_never_touches_the_icd_env(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(0))
    assert agent_main._vulkan_preflight(_CP()) is True
    assert "VK_ICD_FILENAMES" not in __import__("os").environ


def test_default_red_egl_1_3_green_writes_the_1_3_manifest_and_stops_there(monkeypatch):
    calls = {"ffmpeg": 0}

    def fake_run(cmd, *a, **k):
        if cmd[0] == "ldconfig":
            return _ldconfig_ok(cmd)
        assert cmd[0] == "ffmpeg"
        calls["ffmpeg"] += 1
        if calls["ffmpeg"] == 1:
            return _run(1, b"exit-me: VK_ERROR_INCOMPATIBLE_DRIVER\n")
        return _run(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    cp = _CP()
    assert agent_main._vulkan_preflight(cp) is True
    assert calls["ffmpeg"] == 2, "must stop retrying once a rank goes green"
    import os
    assert os.environ["VK_ICD_FILENAMES"] == agent_main._VULKAN_EGL_ICD_PATH
    manifest = open(agent_main._VULKAN_EGL_ICD_PATH).read()
    assert '"api_version":"1.3.0"' in manifest
    assert cp.events == [], "a green verdict carries no failure evidence"


_SHIPPED_ICD_PATH = agent_main._VULKAN_EGL_ICD_PATH   # captured before the autouse fixture redirects it


def _under(path: str, root: str) -> bool:
    import os
    return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)


def test_the_icd_manifest_is_written_outside_tmp():
    """MISC-998 prod 2026-11-03 a46f02d7: Vulkan green at boot, red later — the ICD file VK_ICD_FILENAMES
    names vanished. The shipped path must not live where scratch sweepers work."""
    import tempfile
    assert _SHIPPED_ICD_PATH.startswith("/")
    for scratch in ("/tmp", "/var/tmp", tempfile.gettempdir()):
        assert not _under(_SHIPPED_ICD_PATH, scratch), f"{_SHIPPED_ICD_PATH} is under {scratch}"


def test_a_tmp_sweep_after_the_preflight_leaves_the_manifest_in_place(monkeypatch, tmp_path):
    import os
    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    (fake_tmp / "scratch.bin").write_text("x")
    state = tmp_path / "state" / "monty-pod"      # not created: the writer must create it with parents
    monkeypatch.setattr(agent_main, "_VULKAN_EGL_ICD_PATH", str(state / _ICD_NAME))
    calls = {"ffmpeg": 0}

    def fake_run(cmd, *a, **k):
        if cmd[0] == "ldconfig":
            return _ldconfig_ok(cmd)
        calls["ffmpeg"] += 1
        return _run(1, b"VK_ERROR_INCOMPATIBLE_DRIVER\n") if calls["ffmpeg"] == 1 else _run(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert agent_main._vulkan_preflight(_CP()) is True
    icd = os.environ["VK_ICD_FILENAMES"]

    for p in sorted(fake_tmp.rglob("*"), reverse=True):   # the sweep
        p.unlink() if p.is_file() else p.rmdir()

    assert os.path.exists(icd)
    assert '"library_path"' in open(icd).read()


def test_an_uncreatable_state_dir_fails_the_rank_by_name_and_never_falls_back_to_tmp(monkeypatch, tmp_path):
    import os
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    monkeypatch.setattr(agent_main, "_VULKAN_EGL_ICD_PATH", str(blocker / "sub" / _ICD_NAME))
    monkeypatch.setattr(subprocess, "run", _all_ranks_lose)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    detail = _refused()
    assert "cannot write the ICD manifest" in detail
    assert "ranks_tried=default,egl-1.3.0" in detail and "egl-1.2.0" not in detail
    assert "VK_ICD_FILENAMES" not in os.environ


def test_every_rank_red_refuses_with_ranks_and_driver_evidence(monkeypatch):
    def fake_run(cmd, *a, **k):
        if cmd[0] == "ldconfig":
            return _ldconfig_ok(cmd)
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout=b"570.86.10\n", stderr=b"")
        if str(cmd[0]).endswith("vulkaninfo"):
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"")
        assert cmd[0] == "ffmpeg"
        return _run(1, b"VK_ERROR_INCOMPATIBLE_DRIVER on every rank\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    detail = _refused()
    assert "ranks_tried=default,egl-1.3.0,egl-1.2.0" in detail
    assert "driver_version=570.86.10" in detail
    assert "VK_ERROR_INCOMPATIBLE_DRIVER" in detail
    assert len(detail) < 1800


def test_a_refusal_names_each_graphics_library_and_the_icd_library_path(monkeypatch, tmp_path):
    """MISC-998: the truncated nvidia list and a bare icd.d listing could not settle which graphics userspace
    the runtime injected nor where the loader's ICD points."""
    def fake_run(cmd, *a, **k):
        if cmd[0] == "ldconfig":
            return subprocess.CompletedProcess(
                cmd, 0,
                stdout=("\tlibGLX_nvidia.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0\n"
                        "\tlibnvidia-encode.so.1 (libc6,x86-64) => /usr/lib/libnvidia-encode.so.1\n"),
                stderr="")
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout=b"580.178.04\n", stderr=b"")
        assert cmd[0] == "ffmpeg"
        return _run(1, b"VK_ERROR_INCOMPATIBLE_DRIVER\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    icd = tmp_path / "icd.d"
    icd.mkdir()
    (icd / "nvidia_icd.json").write_text(
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}')
    monkeypatch.setattr(agent_main, "_VULKAN_ICD_DIRS", (str(icd),))

    detail = _refused()

    assert "graphics_libs=" in detail and "libGLX_nvidia.so.0=yes" in detail
    assert "libnvidia-glcore=no" in detail and "libnvidia-encode.so.1=yes" in detail
    assert "libnvcuvid.so.1=no" in detail
    assert f"icd={icd / 'nvidia_icd.json'}->libGLX_nvidia.so.0" in detail
    assert detail.index("graphics_libs=") < detail.index("icd=") < detail.index("nvidia libs:")


def test_egl_1_3_failing_for_a_non_driver_reason_never_reaches_1_2(monkeypatch):
    calls = {"ffmpeg": 0}

    def fake_run(cmd, *a, **k):
        if cmd[0] == "ldconfig":
            return _ldconfig_ok(cmd)
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout=b"580.76.05\n", stderr=b"")
        assert cmd[0] == "ffmpeg"
        calls["ffmpeg"] += 1
        return _run(1, b"VK_ERROR_EXTENSION_NOT_PRESENT: some other libplacebo failure\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    detail = _refused()
    # a lower api_version claim cannot fix a non-driver-version failure
    assert calls["ffmpeg"] == 2, "rank 1 (default) + rank 2 (egl-1.3.0) only — 1.2.0 must not be tried"
    assert "ranks_tried=default,egl-1.3.0" in detail


def _all_ranks_lose(cmd, *_a, **_k):
    if cmd[0] == "ldconfig":
        return _ldconfig_ok(cmd)
    if cmd[0] == "nvidia-smi":
        return subprocess.CompletedProcess(cmd, 0, stdout=b"570.86.10\n", stderr=b"")
    assert cmd[0] == "ffmpeg"
    return _run(1, b"VK_ERROR_INCOMPATIBLE_DRIVER on every rank\n")


def test_an_exhausted_ladder_leaves_no_synthesized_icd_wired_in(monkeypatch):
    import os
    monkeypatch.setattr(subprocess, "run", _all_ranks_lose)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    _refused()
    assert "VK_ICD_FILENAMES" not in os.environ, (
        "a LOSING synthesized manifest must never poison every ffmpeg child that follows boot")


def test_an_exhausted_ladder_restores_an_operators_preset_icd_untouched(monkeypatch):
    import os
    monkeypatch.setenv("VK_ICD_FILENAMES", "/opt/custom/operator_icd.json")
    monkeypatch.setattr(subprocess, "run", _all_ranks_lose)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    _refused()
    assert os.environ["VK_ICD_FILENAMES"] == "/opt/custom/operator_icd.json", (
        "an operator's own preset ICD must survive a losing ladder, not be clobbered forever")


def test_a_non_driver_break_also_restores_the_preset_icd(monkeypatch):
    """The early-break path (rank 2 fails for a non-driver reason, rank 3 never attempted) is a second exit
    from the loop besides exhaustion — it must restore just as faithfully."""
    import os
    monkeypatch.setenv("VK_ICD_FILENAMES", "/opt/custom/operator_icd.json")

    def fake_run(cmd, *a, **k):
        if cmd[0] == "ldconfig":
            return _ldconfig_ok(cmd)
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout=b"580.76.05\n", stderr=b"")
        assert cmd[0] == "ffmpeg"
        return _run(1, b"VK_ERROR_EXTENSION_NOT_PRESENT: not a driver-version problem\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    _refused()
    assert os.environ["VK_ICD_FILENAMES"] == "/opt/custom/operator_icd.json"


# ── MISC-101: ranks_tried=default must arrive with the WHY, not just the symptom ───────────────────────
def _ldconfig_without_egl(cmd, *_a, **_k):
    return subprocess.CompletedProcess(
        cmd, 0,
        stdout=("\tlibnvidia-ml.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1\n"
                "\tlibcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1\n"
                "\tlibnvidia-ptxjitcompiler.so.1 (libc6,x86-64) => /usr/lib/libnvidia-ptxjitcompiler.so.1\n"),
        stderr="")


def _host_without_graphics_userspace(cmd, *_a, **_k):
    if cmd[0] == "ldconfig":
        return _ldconfig_without_egl(cmd)
    if cmd[0] == "nvidia-smi":
        return subprocess.CompletedProcess(cmd, 0, stdout=b"580.126.18\n", stderr=b"")
    assert cmd[0] == "ffmpeg"
    return _run(1, b"Failed to create Vulkan instance: VK_ERROR_INCOMPATIBLE_DRIVER\n")


def test_default_rank_evidence_names_the_missing_egl_the_icd_dirs_and_the_caps(monkeypatch, tmp_path):
    """exit 187 with ranks_tried=default said only that the loader found nothing — it could not tell an
    image fault from a host that exposes the GPU with no GL/EGL userspace at all."""
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "all")
    monkeypatch.setattr(subprocess, "run", _host_without_graphics_userspace)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    icd = tmp_path / "icd.d"
    icd.mkdir()
    (icd / "nvidia_icd.json").write_text("{}")
    monkeypatch.setattr(agent_main, "_VULKAN_ICD_DIRS", (str(icd), str(tmp_path / "gone")))

    detail = _refused()

    assert "ranks_tried=default" in detail and "egl-" not in detail
    assert "no libEGL_nvidia.so.0 in ldconfig" in detail
    assert f"{icd}: nvidia_icd.json" in detail
    assert f"{tmp_path / 'gone'}: absent" in detail
    assert "caps=all" in detail
    assert "libnvidia-ml.so.1" in detail and "libnvidia-ptxjitcompiler.so.1" in detail
    assert "libcuda.so.1" not in detail, "the nvidia grep is the filter, exactly as an operator would run it"
    assert "/usr/lib/x86_64-linux-gnu" not in detail, "library NAMES only, never the resolved paths"
    assert "driver_version=580.126.18" in detail


def test_an_unset_capability_variable_is_a_named_absence_not_a_blank(monkeypatch, tmp_path):
    monkeypatch.delenv("NVIDIA_DRIVER_CAPABILITIES", raising=False)
    monkeypatch.setattr(subprocess, "run", _host_without_graphics_userspace)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    monkeypatch.setattr(agent_main, "_VULKAN_ICD_DIRS", (str(tmp_path),))

    detail = _refused()
    assert "caps=UNSET" in detail
    assert f"{tmp_path}: empty" in detail


def test_a_huge_ffmpeg_dump_is_trimmed_and_the_host_evidence_is_not(monkeypatch, tmp_path):
    """The ffmpeg stderr repeats itself and the box can be re-run; the host evidence dies with the pod, so
    it is the half that must never be the one cut to fit the payload budget."""
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
    monkeypatch.setattr(agent_main, "_VULKAN_ICD_DIRS", (str(tmp_path),))
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    noise = b"x" * 40000

    def fake_run(cmd, *a, **k):
        if cmd[0] == "ldconfig":
            return _ldconfig_without_egl(cmd)
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout=b"580.126.18\n", stderr=b"")
        assert cmd[0] == "ffmpeg"
        return _run(1, noise)

    monkeypatch.setattr(subprocess, "run", fake_run)
    detail = _refused()

    evidence = agent_main._vulkan_host_evidence(_ldconfig_without_egl(["ldconfig"]).stdout, None)
    assert evidence in detail, "the evidence arrived whole"
    assert "chars omitted" in detail, "the dump, not the evidence, paid for the budget"
    assert len(detail) < agent_main._VULKAN_DETAIL_BUDGET + len(evidence) + 400


# ── MISC-998: a host that injects no graphics userspace at all — the last rank brings the driver's own ──────
_V = "580.178.04"


def _fetched(tmp_path):
    lib_dir = tmp_path / f"nvidia-{_V}"
    return du.FetchResult(ok=True, lib_dir=str(lib_dir), manifest=str(lib_dir / "nvidia_icd.json"),
                          elapsed=12.0)


def _host_whose_vulkan_opens_only_on_the_manifest(manifest: str | None):
    """The probe is a fresh process: it sees whatever env is set when it starts."""
    import os

    def fake_run(cmd, *a, **k):
        if cmd[0] == "ldconfig":
            return _ldconfig_without_egl(cmd)
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{_V}\n".encode(), stderr=b"")
        assert cmd[0] == "ffmpeg"
        if manifest is not None and os.environ.get("VK_ICD_FILENAMES") == manifest:
            return _run(0)
        return _run(1, b"Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER\n")

    return fake_run


def test_a_host_without_graphics_userspace_boots_green_on_the_fetched_driver_libs(monkeypatch, tmp_path, capsys):
    import os
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/preset/lib")
    fetched = _fetched(tmp_path)
    calls: list[tuple] = []

    def fake_fetch(version, base_dir, **_k):
        calls.append((version, Path(base_dir)))
        return fetched

    monkeypatch.setattr(agent_main, "_fetch_driver_userspace", fake_fetch)
    monkeypatch.setattr(subprocess, "run", _host_whose_vulkan_opens_only_on_the_manifest(fetched.manifest))
    cp = _CP()
    capacity: dict = {}
    _preflight_with_stubbed_gates(monkeypatch, cp, capacity)

    assert capacity["vulkan"] is True
    assert [ev["phase"] for ev in cp.events if ev.get("phase") == "ready"] == ["ready"]
    assert not [ev for ev in cp.events if ev.get("status") == "error"]
    assert calls == [(_V, Path(agent_main._VULKAN_EGL_ICD_PATH).parent)], "validated version, state dir"
    assert os.environ["VK_ICD_FILENAMES"] == os.environ["VK_DRIVER_FILES"] == fetched.manifest
    assert os.environ["LD_LIBRARY_PATH"] == f"{fetched.lib_dir}:/opt/preset/lib"
    assert f"driver-userspace-{_V}" in capsys.readouterr().err


def test_an_unset_ld_library_path_gets_no_empty_entry(monkeypatch, tmp_path):
    import os
    fetched = _fetched(tmp_path)
    monkeypatch.setattr(agent_main, "_fetch_driver_userspace", lambda *a, **k: fetched)
    monkeypatch.setattr(subprocess, "run", _host_whose_vulkan_opens_only_on_the_manifest(fetched.manifest))
    assert agent_main._vulkan_preflight(_CP()) is True
    assert os.environ["LD_LIBRARY_PATH"] == fetched.lib_dir


_PRESET = {"VK_ICD_FILENAMES": "/opt/op/icd.json", "VK_DRIVER_FILES": "/opt/op/drv.json",
           "LD_LIBRARY_PATH": "/opt/op/lib"}


def test_a_failed_driver_fetch_refuses_by_name_and_restores_the_env(monkeypatch):
    import os
    for key, value in _PRESET.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(agent_main, "_fetch_driver_userspace", lambda *a, **k: du.FetchResult(
        ok=False, reason=du.CHECKSUM_MISMATCH, detail="x", elapsed=30.0))
    monkeypatch.setattr(subprocess, "run", _host_whose_vulkan_opens_only_on_the_manifest(None))
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)

    detail = _refused()

    assert "ranks_tried=default,driver-userspace-580.178.04" in detail
    assert "driver_userspace=CHECKSUM_MISMATCH" in detail and "cap=150s" in detail and "elapsed=" in detail
    assert detail.index("ranks_tried=") < detail.index("driver_userspace=")
    assert "VK_ERROR_INCOMPATIBLE_DRIVER" in detail
    assert {k: os.environ.get(k) for k in _PRESET} == _PRESET


def test_a_fetch_that_leaves_vulkan_red_is_a_named_reprobe_failure_and_unsets_the_env(monkeypatch, tmp_path):
    import os
    fetched = _fetched(tmp_path)
    monkeypatch.setattr(agent_main, "_fetch_driver_userspace", lambda *a, **k: fetched)
    monkeypatch.setattr(subprocess, "run", _host_whose_vulkan_opens_only_on_the_manifest(None))
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)

    detail = _refused()

    assert "driver_userspace=VULKAN_REPROBE_FAILED" in detail
    assert not any(k in os.environ for k in agent_main._VULKAN_ENV_KEYS)


def test_a_red_reprobe_puts_an_operators_three_preset_vars_back_exactly(monkeypatch, tmp_path):
    """The env really IS rewritten here (the fetch succeeded), so restoring it is observable: the ICD and
    driver-files vars return to the operator's values and LD_LIBRARY_PATH loses our prepended directory."""
    import os
    for key, value in _PRESET.items():
        monkeypatch.setenv(key, value)
    fetched = _fetched(tmp_path)
    seen_during_probe: list[str] = []
    inner = _host_whose_vulkan_opens_only_on_the_manifest(None)

    def fake_run(cmd, *a, **k):
        if cmd[0] == "ffmpeg":
            seen_during_probe.append(os.environ.get("LD_LIBRARY_PATH", ""))
        return inner(cmd, *a, **k)

    monkeypatch.setattr(agent_main, "_fetch_driver_userspace", lambda *a, **k: fetched)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)

    detail = _refused()

    assert "driver_userspace=VULKAN_REPROBE_FAILED" in detail
    assert seen_during_probe[-1] == f"{fetched.lib_dir}:/opt/op/lib", "the rank did rewrite the env"
    assert {k: os.environ.get(k) for k in _PRESET} == _PRESET


def test_no_userspace_is_fetched_when_an_earlier_rank_is_green(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("the vendor download must be the LAST rank")

    monkeypatch.setattr(agent_main, "_fetch_driver_userspace", forbidden)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(0))
    assert agent_main._vulkan_preflight(_CP()) is True

    calls = {"ffmpeg": 0}

    def egl_green(cmd, *a, **k):
        if cmd[0] == "ldconfig":
            return _ldconfig_ok(cmd)
        calls["ffmpeg"] += 1
        return _run(1, b"VK_ERROR_INCOMPATIBLE_DRIVER\n") if calls["ffmpeg"] == 1 else _run(0)

    monkeypatch.setattr(subprocess, "run", egl_green)
    assert agent_main._vulkan_preflight(_CP()) is True


def test_a_version_the_host_does_not_settle_skips_the_fetch_and_says_so(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("no fetch without one validated version")

    monkeypatch.setattr(agent_main, "_fetch_driver_userspace", forbidden)

    def fake_run(cmd, *a, **k):
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout=b"580.178.04\n595.71.05\n", stderr=b"")
        if cmd[0] == "ldconfig":
            return _ldconfig_without_egl(cmd)
        return _run(1, b"VK_ERROR_INCOMPATIBLE_DRIVER\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    detail = _refused()
    assert "driver_userspace=DRIVER_VERSION_UNAVAILABLE" in detail and "driver_version=unknown" in detail
    assert "ranks_tried=default ·" in detail


def _smi(monkeypatch, stdout: bytes, rc: int = 0):
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=b""))


def test_the_driver_version_helper_dedupes_identical_gpu_lines(monkeypatch):
    _smi(monkeypatch, b"580.178.04\n 580.178.04 \n\n580.178.04\n")
    assert agent_main._nvidia_driver_version() == "580.178.04"
    _smi(monkeypatch, b"595.71\n")
    assert agent_main._nvidia_driver_version() == "595.71"


def test_the_driver_version_helper_refuses_unknown_malformed_or_mixed_versions(monkeypatch):
    for out in (
            b"",                          # zero GPU lines → DRIVER_VERSION_UNAVAILABLE (brief)
            b"\n  \n",                    # blank lines only: same, after strip (brief)
            b"580.178.04\n595.71.05\n",   # >1 distinct version: no single userspace fits (brief; MISC-998)
            b"[N/A]\n",                   # nvidia-smi's own placeholder is not a version (brief regex)
            b"580\n",                     # fewer than major.minor (brief regex ^\d+\.\d+(\.\d+)?$)
            b"580.178.04-beta\n",         # suffix would build a URL for a package that does not exist (regex)
            b"v580.178.04\n",             # prefix (regex)
            b"580.178.04.1\n",            # four components (regex)
            b"../../etc\n",               # path-injection guard: the version becomes a URL and a directory name
    ):
        _smi(monkeypatch, out)
        assert agent_main._nvidia_driver_version() is None, out
    _smi(monkeypatch, b"580.178.04\n", rc=9)
    assert agent_main._nvidia_driver_version() is None

    def broken(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", broken)
    assert agent_main._nvidia_driver_version() is None
