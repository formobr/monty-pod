"""_vulkan_preflight's 3-rank ladder: a synthesized ICD pinned to ONE driver used to clobber every other
driver's real ICD and denylist a healthy host (the-gpu-verdict-carries-its-evidence-or-it-is-not-a-verdict)."""
from __future__ import annotations

import subprocess

import pytest

from podagent import main as agent_main
from test_nvenc_preflight import _CP, _run  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_icd_env(monkeypatch, tmp_path):
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    monkeypatch.setattr(agent_main, "_VULKAN_EGL_ICD_PATH", str(tmp_path / "nvidia_egl_icd.json"))


def _ldconfig_ok(cmd, *_a, **_k):
    return subprocess.CompletedProcess(
        cmd, 0, stdout="libEGL_nvidia.so.0 (x86-64) => /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0\n",
        stderr="")


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
    capacity: dict = {}
    assert agent_main._vulkan_preflight(_CP(), capacity=capacity) is True
    assert calls["ffmpeg"] == 2, "must stop retrying once a rank goes green"
    import os
    assert os.environ["VK_ICD_FILENAMES"] == agent_main._VULKAN_EGL_ICD_PATH
    manifest = open(agent_main._VULKAN_EGL_ICD_PATH).read()
    assert '"api_version":"1.3.0"' in manifest
    assert "vulkan_detail" not in capacity, "a green verdict carries no failure evidence"


def test_every_rank_red_returns_false_with_ranks_and_driver_evidence(monkeypatch):
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
    capacity: dict = {}
    assert agent_main._vulkan_preflight(_CP(), capacity=capacity) is False
    detail = capacity["vulkan_detail"]
    assert "ranks_tried=default,egl-1.3.0,egl-1.2.0" in detail
    assert "driver_version=570.86.10" in detail
    assert "VK_ERROR_INCOMPATIBLE_DRIVER" in detail
    assert len(detail) < 1800


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
    capacity: dict = {}
    assert agent_main._vulkan_preflight(_CP(), capacity=capacity) is False
    # a lower api_version claim cannot fix a non-driver-version failure
    assert calls["ffmpeg"] == 2, "rank 1 (default) + rank 2 (egl-1.3.0) only — 1.2.0 must not be tried"
    assert "ranks_tried=default,egl-1.3.0" in capacity["vulkan_detail"]


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
    assert agent_main._vulkan_preflight(_CP()) is False
    assert "VK_ICD_FILENAMES" not in os.environ, (
        "a LOSING synthesized manifest must never poison every ffmpeg child that follows boot")


def test_an_exhausted_ladder_restores_an_operators_preset_icd_untouched(monkeypatch):
    import os
    monkeypatch.setenv("VK_ICD_FILENAMES", "/opt/custom/operator_icd.json")
    monkeypatch.setattr(subprocess, "run", _all_ranks_lose)
    monkeypatch.setattr(agent_main.shutil, "which", lambda _name: None)
    assert agent_main._vulkan_preflight(_CP()) is False
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
    assert agent_main._vulkan_preflight(_CP()) is False
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

    capacity: dict = {}
    assert agent_main._vulkan_preflight(_CP(), capacity=capacity) is False
    detail = capacity["vulkan_detail"]

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

    capacity: dict = {}
    agent_main._vulkan_preflight(_CP(), capacity=capacity)
    assert "caps=UNSET" in capacity["vulkan_detail"]
    assert f"{tmp_path}: empty" in capacity["vulkan_detail"]


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
    capacity: dict = {}
    agent_main._vulkan_preflight(_CP(), capacity=capacity)
    detail = capacity["vulkan_detail"]

    evidence = agent_main._vulkan_host_evidence(_ldconfig_without_egl(["ldconfig"]).stdout, None)
    assert evidence in detail, "the evidence arrived whole"
    assert "chars omitted" in detail, "the dump, not the evidence, paid for the budget"
    assert len(detail) < agent_main._VULKAN_DETAIL_BUDGET + len(evidence) + 400
