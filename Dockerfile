# syntax=docker/dockerfile:1.7
# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

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
FROM python:3.11-slim AS runtime

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
# - libglib2.0-0, libxcb1: opencv-python-headless transitive deps on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libxcb1 \
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
