# THIN image: only what EVERY job needs. Model weights are NOT baked — they arrive per-job as a
# presigned tar in InferRequest.weights (contracts v4) and are cached on local disk, so an align pod
# never pays for SigLIP and a render pod pays for neither. Runtime still needs exactly two env vars —
# CP_URL and JOB_TOKEN — see README.
#
# Layers ordered cheap-to-expensive, code copied LAST (it churns the most,
# everything above it is cache-stable across normal commits).
#
# LAYER SHAPE IS COLD-START TIME. This lane has no persistent volume, so every rent pulls the whole image
# from scratch, and docker pulls layers CONCURRENTLY (3 by default). One 4.16 GB blob is therefore one TCP
# stream at an ordinary single-connection rate — the python stack below is split into buckets so the pull
# runs several streams instead of one. Nothing is added or removed by the split; it is the same bytes in
# more pieces. Keep every bucket under ~1 GB and keep them roughly EQUAL: three streams finish together
# only if no single layer is a straggler.
# CUDA 12.8 + torch cu128 = ONE universal image: sm_75..sm_120 (Ada 4090 AND Blackwell RTX 50xx). cu124
# had no sm_120 kernels, so align crashed on 50xx hosts (NVENC is ffmpeg, arch-independent, so it kept working).
# `-base`, NOT `-runtime`: -runtime adds 2.06 GB of cuda-libraries (cublas/cufft/cusolver/cusparse/nccl)
# that NOTHING here links — torch ships its own copies in site-packages/nvidia/*, and the ffmpeg GPU path is
# Vulkan/libplacebo + NVENC, which come from the driver the container runtime injects.
FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# The base declares cuda>=12.8, which nvidia-container-runtime enforces against the HOST driver — and that
# refusal costs a rent: the box boots, bills, never starts the container. Nothing here needs 12.8 (torch
# carries its own cu128 runtime, minor-version-compatible back to the R525/12.0 driver family), so declare
# what we actually need. A host too old for the kernels we load fails loudly in the boot beacon instead.
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.0"

# --- system: python3.11 (via deadsnakes, ubuntu22.04 ships 3.10) + bundled-browser runtime deps
# (standard Remotion list). fontconfig stays: libass resolves caption fonts through it. librsvg2-bin is
# the SVG rasteriser media.still needs: without one on the box, every vector artwork fails to materialise
# and the job ships a black card instead — a failure nothing errors on. ~2 MB on top of the cairo/pango
# libraries the bundled browser uses. ------------------------------------------------------------------
# libvulkan1 is the Vulkan loader ffmpeg libplacebo dlopen's; vulkan-tools supplies vulkaninfo diagnostics.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl xz-utils ca-certificates gnupg \
        fonts-dejavu-core fontconfig librsvg2-bin libvulkan1 vulkan-tools \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils \
        libnss3 libatk-bridge2.0-0 libgtk-3-0 libasound2 libxss1 libgbm1 \
        libxshmfence1 libxcomposite1 libxdamage1 libxrandr2 libxi6 \
        libpango-1.0-0 libcairo2 libxkbcommon0 libx11-xcb1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ── builder: resolve the python stack ONCE, then park it in per-bucket staging roots ──────────────────
# pip resolves the versions (no hand-pinned nvidia wheel list to drift from torch's own requirements), and
# the bucketing MOVES the trees apart, never copies them: a bucket that left a file behind would ship it
# twice, and a duplicated 900 MB library is exactly the defect this stage exists to remove.
#
# Install and bucket in ONE layer on purpose: a second RUN would write the whole 7.25 GB tree a second time
# into the builder's snapshot, and a CI runner's disk is not free. Nothing wants them cached apart anyway —
# a dependency edit invalidates both.
#
# Buckets sized off the MEASURED tree (uncompressed / gzip MB): cudnn 951/658 · cublas 830/594 ·
# cusparselt+cusolver 818/554 · nccl+cusparse 754/584 · rest of nvidia 947/535 · libtorch_cuda.so 871/586 ·
# rest of torch 761/243 · triton 641/188 · everything else 682/190. Nine pieces, a multiple of the three
# concurrent downloads, and the biggest is 658 MB where the single blob was 4163.
# The last bucket is a SWEEP, not a list: a new dependency lands there instead of silently not shipping.
FROM base AS pydeps
RUN set -eux; \
    python3 -m pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128; \
    # soundfile: torchaudio 2.x has no bundled decoder — wav I/O needs a backend
    # transformers PINNED here (not just pyproject): the app is installed `--no-deps` below, so the
    # pyproject `transformers==4.57.6` pin never applied — this line is the EFFECTIVE pin. An unpinned
    # transformers ships a get_image_features that returns a BaseModelOutputWithPooling → clip_rank crash.
    # `websockets` is the only control-plane lane (podagent.event_stream): typed jobs, events and results.
    # It belongs on THIS line and not only in pyproject for the reason stated above — the app is
    # installed --no-deps, so a pyproject dependency never reaches the image. Missing it makes the image
    # unusable: there is deliberately no HTTP fallback.
    # requests/urllib3 FLOORED here for the same --no-deps reason as transformers above: pyproject's deps
    # never reach the image. urllib3 1.x has enforce_content_length but defaults it OFF; 2.x flips the
    # default to ON, so the floor is what makes cp.py's short-body detection load-bearing, not optional.
    python3 -m pip install --no-cache-dir transformers==4.57.6 opencv-python-headless numpy 'requests>=2.32,<3' 'urllib3>=2,<3' pydantic huggingface_hub soundfile Pillow jsonschema websockets; \
    SP=/usr/local/lib/python3.11/dist-packages; \
    bucket() { n="$1"; shift; for p in "$@"; do d="/stage/$n/$(dirname "$p")"; mkdir -p "$d"; mv "$SP/$p" "$d/"; done; }; \
    bucket 01 nvidia/cudnn; \
    bucket 02 nvidia/cublas; \
    bucket 03 nvidia/cusparselt nvidia/cusolver; \
    bucket 04 nvidia/nccl nvidia/cusparse; \
    bucket 05 nvidia; \
    bucket 06 torch/lib/libtorch_cuda.so; \
    bucket 07 torch; \
    bucket 08 triton; \
    mkdir -p /stage/09; find "$SP" -mindepth 1 -maxdepth 1 -exec mv -t /stage/09/ {} +

# ── the shipped image ────────────────────────────────────────────────────────────────────────────────
FROM base

# --- node 20: the Remotion runtime; its pinned headless shell arrives inside each content-addressed bundle.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
ENV REMOTION_BUNDLE_CACHE=/var/cache/monty/remotion

# --- ffmpeg: BtbN static build (NVENC + libplacebo, not in ubuntu22.04 apt) -
# n8.x libplacebo expression graphs hang forever (measured, 3-keyframe repro).
# PINNED TO THE n6.1 LINE, NOT NEWEST: n7.1.5-16-g9a4bb2c579 (the prior pin) silently DISCARDS most of the
# first operand in `acrossfade` when that input is much longer than the second — exactly the shape
# cut_audio.py's left-folded join produces every time, and exactly why every production montage for months
# lost its closing ~24s while every local repro (host ffmpeg 6.1.1-3ubuntu5) passed. Isolated, reproduced 3x
# each: host 6.1.1 and this BtbN n6.1.3 build both hold the full acrossfade join at 17.770000s three times;
# n7.1.5-16 collapses it to 0.78-1.38s. Matching the HOST's 6.1.x line (not chasing the newest BtbN branch)
# is the point — `same-tool-same-version-everywhere` (memory) — and `scripts/broker/pod_ffmpeg_pin.py` +
# `fleet_doctor` gate this exact identity so a future repin that reintroduces the regression is a refusal,
# not a silent loss. Re-verify NVENC against a real rented host before ever moving this pin again
# (docs/POD_RUNBOOK.md `v0.9.0` incident) — this fix only proves the acrossfade/CPU filter path.
ARG FFMPEG_BUILD=autobuild-2025-08-31-13-00
ARG FFMPEG_ASSET=ffmpeg-n6.1.3-linux64-gpl-6.1.tar.xz
# `/releases/download/<tag>/`, NOT `/releases/latest/download/`: the second means "the newest release",
# which on any day BtbN cuts a dated autobuild carries only version-stamped asset names — the URL 404s, curl
# without --fail writes the error page, and tar dies with "File format not recognized" three lines later.
RUN curl -fL -o /tmp/ffmpeg.tar.xz \
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_BUILD}/${FFMPEG_ASSET}" \
    && mkdir -p /tmp/ffmpeg && tar -xf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
    && install -m 0755 /tmp/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg \
    && install -m 0755 /tmp/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

# --- python deps, one layer per bucket. Same tree the builder resolved, reassembled in place: the
# destination is one directory, so the union of the buckets IS site-packages and no bucket overlaps another.
COPY --from=pydeps /stage/01/ /usr/local/lib/python3.11/dist-packages/
COPY --from=pydeps /stage/02/ /usr/local/lib/python3.11/dist-packages/
COPY --from=pydeps /stage/03/ /usr/local/lib/python3.11/dist-packages/
COPY --from=pydeps /stage/04/ /usr/local/lib/python3.11/dist-packages/
COPY --from=pydeps /stage/05/ /usr/local/lib/python3.11/dist-packages/
COPY --from=pydeps /stage/06/ /usr/local/lib/python3.11/dist-packages/
COPY --from=pydeps /stage/07/ /usr/local/lib/python3.11/dist-packages/
COPY --from=pydeps /stage/08/ /usr/local/lib/python3.11/dist-packages/
COPY --from=pydeps /stage/09/ /usr/local/lib/python3.11/dist-packages/
# console scripts the wheels installed (torchrun, transformers-cli): nothing here calls them, but leaving
# an installed distribution half-present is the kind of difference that surfaces as a mystery on a paid box.
COPY --from=pydeps /usr/local/bin/ /usr/local/bin/

# Weights are NOT baked and the pod holds no HF credential — it never dials HF. Every heavy checkpoint
# arrives as a presigned tar the CP hands it (podagent/weights.py), cached under WEIGHTS_CACHE by content
# hash. Keeping HF_HUB_OFFLINE=1 makes an accidental hub call fail LOUDLY instead of hanging on a rented
# box whose egress may be blocked.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WEIGHTS_CACHE=/var/cache/monty/weights \
    POD_STREAM_OUTBOX=/var/cache/monty/pod-stream/outbox.json

# The image runs as root (no later USER directive), but create the durable transport directory explicitly:
# an unwritable outbox is a boot failure, never a reason to downgrade a terminal to memory.
RUN mkdir -p /var/cache/monty/pod-stream

# YuNet (face_probe) STAYS baked: 227 KB, every probe job needs it, and a fetch round-trip would cost
# more than the bytes. The rule is "big models are inputs", not "nothing is baked".
RUN mkdir -p /opt/models \
    && curl -L -o /opt/models/yunet.onnx \
        https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
ENV MODEL_YUNET=/opt/models/yunet.onnx

# --- Montserrat (cover/caption rendering) -----------------------------------
RUN mkdir -p /usr/share/fonts/truetype/montserrat \
    && curl -L -o /usr/share/fonts/truetype/montserrat/Montserrat.ttf \
        "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf" \
    && fc-cache -f

# --- code: churns most, copied last so nothing above rebuilds on a commit --
WORKDIR /app
COPY pyproject.toml ./
COPY podagent/ ./podagent/
COPY contracts/ ./contracts/
RUN python3 -m pip install --no-cache-dir --no-deps .

# mograph is now COMPLETE on the pod: node above is the runtime, and the Remotion bundle
# (node_modules + src + render_batch.mjs) is delivered per job as `motion_plan.bundle` — a presigned tar
# cached under REMOTION_BUNDLE_CACHE by content hash, same shape as weights (podagent/bundle.py). The
# contract refuses sections without a bundle, so a mis-deployed pod fails loud instead of quietly
# publishing a video with no motion graphics in it. See docs/POD_RUNBOOK.md §2a.

# WHICH BUILD IS THIS. A pod that refuses an op ("this image does not carry media.audio") is unactionable
# unless it can also say which image it is — the control plane pins a tag, but only the running box knows
# what it actually booted. Set from the CI tag; "unknown" on a hand-built image, which is itself the answer.
# Last layer: it changes on every tag and must invalidate nothing above it.
ARG IMAGE_TAG=dev
ARG IMAGE_REVISION=unknown
ENV POD_IMAGE_TAG=${IMAGE_TAG}
LABEL org.opencontainers.image.revision=${IMAGE_REVISION} \
      org.opencontainers.image.version=${IMAGE_TAG} \
      org.opencontainers.image.source="https://github.com/formobr/monty-pod"

ENTRYPOINT ["python3", "-m", "podagent.main"]
