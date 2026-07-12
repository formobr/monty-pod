# One fat image: everything baked at build time, nothing dials in. Runtime
# needs exactly two env vars — CP_URL and JOB_TOKEN — see README.
#
# Layers ordered cheap-to-expensive, code copied LAST (it churns the most,
# everything above it is cache-stable across normal commits).
# CUDA 12.8 base + torch cu128 = ONE universal image: sm_75..sm_120 (Ada 4090 AND Blackwell RTX 50xx). cu124
# had no sm_120 kernels, so align crashed on 50xx hosts (NVENC is ffmpeg, arch-independent, so it kept working).
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --- system: python3.11 (via deadsnakes, ubuntu22.04 ships 3.10) + chromium's
# runtime deps (standard puppeteer/headless-chrome list) ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl xz-utils ca-certificates gnupg \
        fonts-dejavu-core \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils \
        libnss3 libatk-bridge2.0-0 libgtk-3-0 libasound2 libxss1 libgbm1 \
        libxshmfence1 libxcomposite1 libxdamage1 libxrandr2 libxi6 \
        libpango-1.0-0 libcairo2 libxkbcommon0 libx11-xcb1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# --- node 20 + chrome stable: baked now for the future motion-graphics
# render pass, so the image never has to change shape for it -----
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && curl -L -o /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y --no-install-recommends /tmp/chrome.deb \
    && rm -rf /tmp/chrome.deb /var/lib/apt/lists/*

# --- ffmpeg: BtbN static build (NVENC + libplacebo, not in ubuntu22.04 apt) -
RUN curl -L -o /tmp/ffmpeg.tar.xz \
        https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linux64-gpl.tar.xz \
    && mkdir -p /tmp/ffmpeg && tar -xf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
    && install -m 0755 /tmp/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg \
    && install -m 0755 /tmp/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

# --- python deps: torch/torchaudio (cu124 wheel, the huge one) first, then
# the rest ---------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    # soundfile: torchaudio 2.x has no bundled decoder — wav I/O needs a backend
    && python3 -m pip install --no-cache-dir transformers opencv-python-headless numpy requests pydantic huggingface_hub soundfile Pillow

# --- bake model weights so the pod boots ready, no cold-start download -----
ENV HF_HOME=/opt/hf
# gated repo: token comes in as a BuildKit secret, never a layer
RUN --mount=type=secret,id=hf_token \
    python3 -c "import pathlib; \
        from huggingface_hub import snapshot_download; \
        tok = pathlib.Path('/run/secrets/hf_token').read_text().strip(); \
        snapshot_download('voidful/wav2vec2-xlsr-multilingual-56', token=tok)"
# weights are baked — runtime never dials HF (rented boxes may block egress anyway)
ENV HF_HUB_OFFLINE=1

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

ENTRYPOINT ["python3", "-m", "podagent.main"]
