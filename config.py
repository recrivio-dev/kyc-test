"""Central configuration for the fast KYC pipeline.

Keeping every tunable in one place makes the latency/accuracy trade-offs
explicit and lets the Streamlit layer or tests override them per run.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OCRSettings:
    primary_lang: str = "en"
    # Crops whose average recognition confidence falls below this trigger a
    # Surya re-OCR — on that crop only, never the full page.
    fallback_threshold: float = 0.75
    min_text_len: int = 2
    # Surya fallback is OFF by default: surya-ocr 0.17 needs a matching
    # `transformers` version and otherwise fails at inference. Flip this on
    # once the dependency versions are aligned.
    enable_surya_fallback: bool = False


@dataclass
class LayoutSettings:
    # "rapidocr_det" — use the ONNX text-detection model to locate text
    #                  regions. Works out of the box, no extra model file.
    # "yolo"         — use a generic document-layout YOLO ONNX model.
    backend: str = "rapidocr_det"
    yolo_model_path: str = "models/layout.onnx"
    yolo_classes: tuple = ("text", "title", "list", "table", "figure")
    score_threshold: float = 0.30
    # Brute-force 4-angle orientation probe on a downscaled image.
    # Cheap (detection-only), and far cheaper than the old 4 full OCR passes.
    detect_orientation: bool = True


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
    # Working resolution for the located/cropped OCR stage.
    work_max_side: int = 2000


SETTINGS = Settings()
