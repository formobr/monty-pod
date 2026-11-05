"""Headless NVIDIA Vulkan needs libEGL in the IMAGE, not just the loader.

The container toolkit injects the NVIDIA Vulkan ICD (libGLX_nvidia + its ICD manifest), but that ICD resolves
libEGL.so.1 internally and silently fails vkCreateInstance (ERROR_INCOMPATIBLE_DRIVER) when the image has
none — so libplacebo cannot create a Vulkan device and every camera.apply pass fails on a pod. Static check
of the Dockerfile; no docker build here. See docs/research/pod-image-headless-vulkan-libegl.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
EGL_MANIFEST = ROOT / "docker" / "10_nvidia.json"
EGL_MANIFEST_IN_IMAGE = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"


def _apt_installs() -> list[tuple[list[str], set[str]]]:
    """(flags, package names) of every `apt-get install` — continuations joined, comments dropped."""
    text = "\n".join(ln for ln in DOCKERFILE.read_text().splitlines() if not ln.strip().startswith("#"))
    text = re.sub(r"\\\n", " ", text)
    installs = []
    for m in re.finditer(r"apt-get install\s+(.*?)(?=&&|;|\n|$)", text):
        words = m.group(1).split()
        installs.append(([w for w in words if w.startswith("-")], {w for w in words if not w.startswith("-")}))
    return installs


def _caps_env_by_stage() -> list[tuple[int, set[str]]]:
    """(stage index, capability set) of every effective NVIDIA_DRIVER_CAPABILITIES ENV — comments dropped,
    continuations joined like _apt_installs."""
    text = "\n".join(ln for ln in DOCKERFILE.read_text().splitlines() if not ln.strip().startswith("#"))
    text = re.sub(r"\\\n", " ", text)
    stage = -1
    found: list[tuple[int, set[str]]] = []
    for ln in text.splitlines():
        if re.match(r"\s*FROM\s", ln, re.I):
            stage += 1
        elif re.match(r"\s*ENV\s", ln, re.I):
            m = re.search(r"NVIDIA_DRIVER_CAPABILITIES(?:=|\s+)(\"[^\"]*\"|\S+)", ln)
            if m:
                found.append((stage, set(m.group(1).strip('"').split(","))))
    return found


def test_the_image_declares_compute_utility_video_graphics_and_nothing_else():
    """MISC-998: the CUDA base image's compute,utility left the runtime no reason to inject the graphics and
    video userspace. A create-time -e overrides this ENV, so it governs only injection paths that consume no
    create-time value; it is a hypothesis test, not a proven fix."""
    declared = _caps_env_by_stage()
    assert declared == [(0, {"compute", "utility", "video", "graphics"})], (
        "exactly one effective declaration, in the base (first) stage, with exactly these capabilities "
        "(no display, no all) and no later stage re-setting it")


def test_the_image_ships_libegl_next_to_the_vulkan_loader():
    with_loader = [(flags, pkgs) for flags, pkgs in _apt_installs() if "libvulkan1" in pkgs]
    assert len(with_loader) == 1, "expected exactly one apt-get install that ships libvulkan1"
    flags, pkgs = with_loader[0]
    assert "libegl1" in pkgs, "libegl1 must be installed in the same apt line as libvulkan1"
    assert "--no-install-recommends" in flags


def test_the_repo_manifest_is_exactly_the_soname_only_nvidia_egl_vendor_file():
    assert json.loads(EGL_MANIFEST.read_text()) == {
        "file_format_version": "1.0.0",
        "ICD": {"library_path": "libEGL_nvidia.so.0"},
    }


def test_the_dockerfile_installs_the_manifest_into_the_glvnd_vendor_dir():
    copies = [
        ln.split()[1:] for ln in DOCKERFILE.read_text().splitlines() if ln.startswith("COPY ") and "--from" not in ln
    ]
    assert ["docker/10_nvidia.json", EGL_MANIFEST_IN_IMAGE] in copies
