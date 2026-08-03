"""Central configuration for the liveness / face-match layers.

Same tunables as live-mini's ``backend/config.py``, but expressed in the house
style of this repo's root ``config.py`` (a ``@dataclass`` + ``os.getenv``
helpers) instead of pydantic-settings + a ``.env`` file — the OCR side has no
pydantic-settings dependency and this package must not add one.

Every field name, default and threshold below is carried over verbatim; the
comments explaining *why* each operating point is what it is came with them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root. ``liveness/config.py`` sits one level below it, so parent.parent
# resolves to the same directory ``api.py`` lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Bumped whenever scoring changes — new stage, reweighting, retuned threshold.
# Stamped onto every response and persisted by the caller alongside the decision,
# so an audit can tell "this was scored under v1, that under v2" instead of
# assuming today's rules produced a decision made months ago. Distinct from
# ``app_version`` below, which tracks the HTTP surface, not the maths.
PIPELINE_VERSION = "v1.1.0"


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class LivenessSettings:
    """Strongly-typed liveness settings."""

    # ---- Server ----
    app_name: str = "Enterprise Liveness Verification"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # ---- Models ----
    # Weights are ~190 MB after the module trim and are fetched at runtime, so
    # they must land on a mounted cache volume rather than the container layer.
    # For local dev outside Docker: LIVENESS_MODELS_DIR=./liveness_models
    models_dir: str = field(
        default_factory=lambda: _env_str("LIVENESS_MODELS_DIR", "/cache/liveness")
    )
    insightface_model: str = field(
        default_factory=lambda: _env_str("LIVENESS_INSIGHTFACE_MODEL", "buffalo_l")
    )
    ort_providers: str = field(
        default_factory=lambda: _env_str("LIVENESS_ORT_PROVIDERS", "CPUExecutionProvider")
    )
    det_size: int = field(default_factory=lambda: _env_int("LIVENESS_DET_SIZE", 640))

    # ---- Decision bands (0-100) ----
    threshold_verified: float = 95
    threshold_high: float = 90
    threshold_medium: float = 80
    threshold_review: float = 70

    # ---- Identity ----
    # ArcFace (w600k_r50) cosine operating points tuned for KYC with aging
    # tolerance. Impostors typically score < 0.25, so these keep a wide margin.
    similarity_threshold_high: float = 0.40
    similarity_threshold_low: float = 0.30

    # ---- Periocular (age-stable eye-region) corroboration ----
    # Large age gaps mature the jaw/mouth and drag whole-face cosine below the
    # threshold even for genuine pairs; the eye/brow region stays stable. A
    # strict periocular gate adds corroborating evidence WITHOUT lowering the
    # global similarity threshold (so impostor rejection is preserved).
    periocular_crop_fraction: float = 0.55
    periocular_match_threshold: float = 0.42
    periocular_margin: float = 0.08

    # ---- Final weights ----
    weight_capture: float = 0.20
    weight_identity: float = 0.35
    weight_liveness: float = 0.45

    # ---- JSON frame-burst limits ----
    # Mirrors the multipart ``read_frames`` ceiling so both transports behave
    # identically; the byte cap stops a base64 body from OOM-ing the worker
    # (base64 inflates ~33%, and 90 frames of 720p JPEG is order-10 MB).
    max_json_frames: int = field(
        default_factory=lambda: _env_int("LIVENESS_MAX_JSON_FRAMES", 90)
    )
    max_json_bytes: int = field(
        default_factory=lambda: _env_int("LIVENESS_MAX_JSON_BYTES", 24 * 1024 * 1024)
    )

    @property
    def models_path(self) -> Path:
        p = (PROJECT_ROOT / self.models_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def providers(self) -> list[str]:
        return [p.strip() for p in self.ort_providers.split(",") if p.strip()]


settings = LivenessSettings()


def get_settings() -> LivenessSettings:
    """Return the process-wide settings singleton."""
    return settings


__all__ = ["PROJECT_ROOT", "LivenessSettings", "settings", "get_settings"]
