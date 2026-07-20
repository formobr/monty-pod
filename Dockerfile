# THIN image: only what EVERY job needs. Model weights are NOT baked — they arrive per-job as a
# presigned tar in InferRequest.weights (contracts v4) and are cached on local disk, so an align pod
# never pays for SigLIP and a render pod pays for neither. Runtime still needs exactly two env vars —
# CP_URL and JOB_TOKEN — see README.
#
# Layers ordered cheap-to-expensive, code copied LAST (it churns the most,
# everything above it is cache-stable across normal commits).
# CUDA 12.8 + torch cu128 = ONE universal image: sm_75..sm_120 (Ada 4090 AND Blackwell RTX 50xx). cu124
# had no sm_120 kernels, so align crashed on 50xx hosts (NVENC is ffmpeg, arch-independent, so it kept working).
# `-base`, NOT `-runtime`: -runtime adds 2.06 GB of cuda-libraries (cublas/cufft/cusolver/cusparse/nccl)
# that NOTHING here links — torch ships its own copies in site-packages/nvidia/*, and the ffmpeg GPU path is
# Vulkan/libplacebo + NVENC, which come from the driver the container runtime injects.
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --- system: python3.11 (via deadsnakes, ubuntu22.04 ships 3.10) + headless-chrome's runtime deps
# (standard puppeteer list). fontconfig stays: libass resolves caption fonts through it. ------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl xz-utils ca-certificates gnupg \
        fonts-dejavu-core fontconfig \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils \
        libnss3 libatk-bridge2.0-0 libgtk-3-0 libasound2 libxss1 libgbm1 \
        libxshmfence1 libxcomposite1 libxdamage1 libxrandr2 libxi6 \
        libpango-1.0-0 libcairo2 libxkbcommon0 libx11-xcb1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# --- node 20 + chrome stable: the RUNTIME for mograph. ~243 MB, and unlike the bundle it is genuinely
# image-shaped — it is a binary toolchain, not content, so it neither churns with our code nor differs
# per brand. The bundle it executes (node_modules + src, 506 MB) is NOT here: that arrives per job as a
# presigned tar cached by content hash, so a pod that renders no mograph pays for none of it. -------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && curl -L -o /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y --no-install-recommends /tmp/chrome.deb \
    && rm -rf /tmp/chrome.deb /var/lib/apt/lists/*
# Remotion must use the chrome we installed, never download its own at render time: a rented pod may have
# blocked egress, and a silent per-job browser fetch is exactly the cold-start cost this lane exists to avoid.
ENV REMOTION_CHROME_EXECUTABLE=/usr/bin/google-chrome-stable \
    REMOTION_BUNDLE_CACHE=/var/cache/monty/remotion

# --- ffmpeg: BtbN static build (NVENC + libplacebo, not in ubuntu22.04 apt) -
RUN curl -L -o /tmp/ffmpeg.tar.xz \
        https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linux64-gpl.tar.xz \
    && mkdir -p /tmp/ffmpeg && tar -xf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
    && install -m 0755 /tmp/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg \
    && install -m 0755 /tmp/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

# --- python deps: torch/torchaudio (cu128 wheel, the huge one) first, then
# the rest. This is the ONE unavoidably heavy layer: it is the runtime, not an input. ------
RUN python3 -m pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    # soundfile: torchaudio 2.x has no bundled decoder — wav I/O needs a backend
    && python3 -m pip install --no-cache-dir transformers opencv-python-headless numpy requests pydantic huggingface_hub soundfile Pillow

# Weights are NOT baked and the pod holds no HF credential — it never dials HF. Every heavy checkpoint
# arrives as a presigned tar the CP hands it (podagent/weights.py), cached under WEIGHTS_CACHE by content
# hash. Keeping HF_HUB_OFFLINE=1 makes an accidental hub call fail LOUDLY instead of hanging on a rented
# box whose egress may be blocked.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WEIGHTS_CACHE=/var/cache/monty/weights

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

# mograph is now COMPLETE on the pod: node+chrome above are the runtime, and the Remotion bundle
# (node_modules + src + render_batch.mjs) is delivered per job as `motion_plan.bundle` — a presigned tar
# cached under REMOTION_BUNDLE_CACHE by content hash, same shape as weights (podagent/bundle.py). The
# contract refuses sections without a bundle, so a mis-deployed pod fails loud instead of quietly
# publishing a video with no motion graphics in it. See docs/POD_RUNBOOK.md §2a.

ENTRYPOINT ["python3", "-m", "podagent.main"]
