"""Central configuration for the fast KYC pipeline.

Keeping every tunable in one place makes the latency/accuracy trade-offs
explicit and lets the Streamlit layer or tests override them per run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class OCRSettings:
    primary_lang: str = "en"
    # Crops whose average recognition confidence falls below this trigger a
    # Surya re-OCR — on that crop only, never the full page.
    fallback_threshold: float = 0.75
    min_text_len: int = 2
    # Surya fallback is OFF by default: surya-ocr 0.17 needs a matching
    # `transformers` version and otherwise fails at inference. Flip this on
    # once the dependency versions are aligned. Overridable in production via
    # the KYC_ENABLE_SURYA env var so the container doesn't need rebuilding.
    enable_surya_fallback: bool = field(
        default_factory=lambda: _env_bool("KYC_ENABLE_SURYA", False)
    )
    # ONNX Runtime intra-op threads per session. 0 = ORT default (one per
    # physical core). Set to the vCPUs available to ONE worker.
    threads: int = field(default_factory=lambda: _env_int("KYC_ORT_THREADS", 0))
    # Longest side the text *detector* sees. Recognition still crops from the
    # full-resolution work image, so this trades only small-text recall for
    # speed. 0 = RapidOCR default (detector runs at up to work_max_side).
    det_max_side: int = field(
        default_factory=lambda: _env_int("KYC_DET_MAX_SIDE", 0))
    # Recognition model override. Empty = RapidOCR's bundled ch_PP-OCRv4 rec.
    rec_model_path: str = field(
        default_factory=lambda: os.getenv("KYC_REC_MODEL", ""))
    rec_keys_path: str = field(
        default_factory=lambda: os.getenv("KYC_REC_KEYS", ""))


@dataclass
class LayoutSettings:
    # "rapidocr_det" — use the ONNX text-detection model to locate text
    #                  regions. Works out of the box, no extra model file.
    # "yolo"         — use a generic document-layout YOLO ONNX model.
    backend: str = "rapidocr_det"
    yolo_model_path: str = "models/layout.onnx"
    yolo_classes: tuple = ("text", "title", "list", "table", "figure")
    score_threshold: float = 0.30
    # Stand the page upright before the main read. A detection + angle-class
    # probe decides clear cases; ambiguous ones fall back to the 4-angle
    # full-OCR probe.
    detect_orientation: bool = True
    # Longest side of the downscaled copy the orientation probe runs on.
    orientation_probe_side: int = 960


@dataclass
class MaskSettings:
    # Padding added around a sensitive box, as a fraction of the box height.
    pad_ratio: float = 0.18
    # For a single-line Aadhaar number, keep the right-most fraction of the
    # box visible so the last 4 digits stay readable.
    aadhaar_visible_ratio: float = 0.34
    # Horizontal gap (px) under which two sensitive boxes are merged.
    merge_gap: int = 22


@dataclass
class Settings:
    ocr: OCRSettings = field(default_factory=OCRSettings)
    layout: LayoutSettings = field(default_factory=LayoutSettings)
    mask: MaskSettings = field(default_factory=MaskSettings)
    output_dir: str = "sample-docs"
    # Write the masked render of every OCR request to `output_dir`. Off by
    # default: the API never serves the file, and it leaves ID images on disk.
    # The Streamlit/CLI tools turn it on.
    save_masked_output: bool = field(
        default_factory=lambda: _env_bool("KYC_SAVE_MASKED", False))
    # Working resolution for the located/cropped OCR stage.
    work_max_side: int = 2000


SETTINGS = Settings()
