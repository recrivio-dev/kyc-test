"""Gunicorn production config for the KYC OCR FastAPI service.

Tuned for a 2 vCPU / 4 GB host (e.g. AWS t3.medium):
  - 2 uvicorn workers, 1 thread each (ONNX releases the GIL, so async
    inside the worker already overlaps requests).
  - OMP_NUM_THREADS=1 in the container env so workers don't fight for cores.
  - 120 s timeout covers cold-start ONNX model load + slow PDFs.

If KYC_ENABLE_SURYA=true, each worker grows by ~1.5-2 GB after the first
low-confidence crop. On a 4 GB host you MUST drop to WEB_CONCURRENCY=1.

The liveness endpoints add a MediaPipe graph + InsightFace ONNX sessions per
worker on top of the OCR models, once that worker serves its first liveness
request. See docker-compose.yml for the measured cost and the concurrency
trade-off.
"""
from __future__ import annotations

import multiprocessing
import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

workers = _int_env(
    "WEB_CONCURRENCY",
    # Default: 2 for t3.medium. Cap at cpu_count to avoid OOM on tiny hosts.
    min(2, multiprocessing.cpu_count()),
)
worker_class = "uvicorn.workers.UvicornWorker"
threads = 1

# ONNX models load on first request per worker; don't preload via fork
# (onnxruntime sessions don't survive fork cleanly on Linux). This now covers
# the liveness InsightFace sessions too — they are onnxruntime sessions with
# the same fork-safety problem, so preload_app must stay False.
preload_app = False

timeout = _int_env("GUNICORN_TIMEOUT", 120)
graceful_timeout = 30
keepalive = 5

# Recycle workers periodically to defend against any slow RAM growth in
# long-lived processes (image buffers, cv2 caches, etc.).
max_requests = _int_env("GUNICORN_MAX_REQUESTS", 500)
max_requests_jitter = 50

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
