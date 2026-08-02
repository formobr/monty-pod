"""The pull is the cold start, and layer SHAPE is what the pull can parallelise.

This lane has no persistent volume: every rent pulls the whole image from scratch, and docker fetches
layers concurrently (3 by default). A measured build put 88% of the image — 4.16 GB compressed — into a
single `pip install` layer, so the pull collapsed to ONE TCP stream: 230 s of a 282 s stage.

These assertions are about SHAPE, not size, because size is the thing a Dockerfile edit changes by
accident. A future edit that folds the python stack back into one RUN, or that copies a bucket instead of
moving it (shipping the same 900 MB library twice), lands RED here instead of on a rented box.
"""
from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"

# Three concurrent downloads is docker's default and we do not control the rented host's daemon, so the
# stack has to arrive in enough near-equal pieces to keep all three busy for more than one round.
MIN_BUCKETS = 6

# The wheels that make the image big. A `pip install` of these in the SHIPPED stage is the single fat
# layer coming back.
HEAVY = ("torch", "transformers", "opencv-python-headless")


def _stages() -> list[tuple[str, list[str]]]:
    """(stage name, its instruction lines) — line continuations joined, comments dropped."""
    # comment lines go BEFORE continuations are joined — that is docker's own order, and a comment sitting
    # inside a multi-line RUN is legal there.
    text = "\n".join(ln for ln in DOCKERFILE.read_text().splitlines() if not ln.strip().startswith("#"))
    text = re.sub(r"\\\n", " ", text)
    stages: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("FROM "):
            name = line.split(" AS ")[-1].strip() if " AS " in line.upper() else "<final>"
            stages.append((name, []))
            continue
        if stages:
            stages[-1][1].append(line)
    return stages


def _final_stage() -> list[str]:
    return _stages()[-1][1]


def test_the_shipped_stage_installs_no_heavy_wheel():
    """The fat layer is a `pip install` of the ML stack in the stage that ships. Resolving it in a builder
    and copying the result back is what makes the split possible at all."""
    offenders = [ln for ln in _final_stage()
                 if ln.startswith("RUN") and "pip install" in ln and any(h in ln for h in HEAVY)]
    assert not offenders, (
        "the python stack is installed in the SHIPPED stage — that is one 4 GB layer and one TCP stream "
        f"on every rent: {offenders}")


def test_the_python_stack_arrives_in_several_layers():
    """Each COPY is a layer is a stream. One COPY of the whole site-packages would be the fat layer again,
    just spelled differently."""
    copies = [ln for ln in _final_stage() if ln.startswith("COPY --from=") and "/stage/" in ln]
    assert len(copies) >= MIN_BUCKETS, (
        f"only {len(copies)} python-stack layers; docker pulls 3 at a time, so fewer than {MIN_BUCKETS} "
        "near-equal pieces cannot keep the streams busy")


def test_buckets_are_moved_not_copied():
    """A bucket that COPIES leaves the original behind, and the file ships twice — a duplicated cudnn is
    ~950 MB of cold start for nothing, and nothing else in the build would notice."""
    builder = dict(_stages()).get("pydeps")
    assert builder is not None, "no `pydeps` builder stage — the buckets have nowhere to come from"
    staging = [ln for ln in builder if "/stage/" in ln]
    assert staging, "the builder never stages the tree into buckets"
    assert not any(re.search(r"\bcp\b|\bcp -", ln) for ln in staging), (
        "a bucket is COPIED out of site-packages; it must be MOVED, or the image carries it twice")


def test_the_last_bucket_is_a_sweep():
    """Buckets named by hand cannot cover a dependency nobody listed. The final bucket takes whatever is
    left, so a new wheel ships in a layer instead of not shipping at all."""
    builder = dict(_stages())["pydeps"]
    assert any("find" in ln and "-maxdepth 1" in ln and "mv" in ln for ln in builder), (
        "no sweep bucket — a dependency added later would silently not be copied into the image")


def test_ffmpeg_is_pinned_and_not_the_rolling_master():
    """NEGATIVE: point this back at `master-latest` and it reddens. BtbN's master is built against whatever
    nvenc SDK is current — the 2026-08-02 rebuild took SDK 13.1, which refuses h264_nvenc under driver
    <610.00, so the image could not encode a single frame on any host our providers rent."""
    text = "\n".join(ln for ln in DOCKERFILE.read_text().splitlines() if not ln.strip().startswith("#"))
    assert "master-latest" not in text, (
        "the image pulls BtbN's rolling master build — an unpinned encoder is a supply chain, and this one "
        "shipped an ffmpeg needing a driver newer than the fleet has")
    assert re.search(r"ARG FFMPEG_BUILD=autobuild-\d{4}-\d{2}-\d{2}", text), (
        "no dated FFMPEG_BUILD tag — without one the URL follows whatever BtbN published last")
    asset = re.search(r"ARG FFMPEG_ASSET=(\S+)", text)
    assert asset, "no FFMPEG_ASSET pin"
    assert re.search(r"-gpl-\d+\.\d+\.tar\.xz$", asset.group(1)), (
        f"FFMPEG_ASSET {asset.group(1)!r} is not a release-branch build (…-gpl-<n.n>.tar.xz) — only the "
        "release branches were verified to open h264_nvenc on the fleet's driver")
