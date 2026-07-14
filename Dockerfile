# syntax=docker/dockerfile:1.7
# ---------- Stage 1: builder ----------
# Pin to bookworm (Debian 12). The bare `python:3.11-slim` tag moved to trixie
# (Debian 13), where `libgl1` no longer ships libGL.so.1 and `import cv2` fails
# at boot. Bookworm is the environment the known-good images were built on.
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build tools only needed in the builder stage. Keeps the runtime image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only torch / torchvision FIRST so the requirements.txt install
# below sees them already satisfied and doesn't pull the CUDA wheels.
RUN pip install --upgrade pip setuptools wheel \
 && pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 \
        torchvision==0.23.0

COPY requirements.txt .
RUN pip install -r requirements.txt \
 && pip install gunicorn==23.0.0

# ---------- Stage 2: runtime ----------
# Pinned to bookworm — see the builder stage note. libgl1 provides libGL.so.1
# here (opencv needs it at import); trixie does not.
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Hugging Face / Surya cache lives on a mounted volume so models aren't
    # re-downloaded on every container restart.
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    # ONNX & BLAS threading: one thread per worker, gunicorn fans out via
    # multiple workers instead. Prevents thread storms on 2 vCPU.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Runtime system libs only.
# - libgomp1: required by onnxruntime + torch (OpenMP).
# - libglib2.0-0, libxcb1, libgl1: opencv transitive deps on slim. libgl1 is
#   needed because torchvision/surya pull in cv2 paths that still link
#   against libGL even when opencv-python-headless is the installed wheel.
# - libsm6, libxext6, libxrender1: extra cv2 transitive deps that surface
#   when surya/torchvision touch the imaging stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libxcb1 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd --create-home --shell /bin/bash --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app api.py config.py kyc_pipeline.py layout_detector.py \
                     ocr_engines.py output_schema.py preprocessing.py \
                     gunicorn_conf.py /app/

RUN mkdir -p /cache/huggingface /cache/torch /app/sample-docs \
 && chown -R app:app /cache /app/sample-docs

USER app

EXPOSE 8000

# tini as PID 1 so SIGTERM / zombie reaping behave correctly.
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["gunicorn", "api:app", "-c", "/app/gunicorn_conf.py"]
