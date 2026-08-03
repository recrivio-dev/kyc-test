"""Image-quality assessment utilities.

These power the "Quality Compensation System": instead of hard-rejecting a
genuine user under poor conditions, we measure capture quality and translate
deficits into confidence reductions and human-readable explanations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from liveness.schemas.responses import Reason
from liveness.services import reason_codes as rc
from liveness.utils.image import to_gray
from liveness.utils.metrics import clamp, linear_map


@dataclass
class QualityReport:
    """Per-frame (or aggregated) image quality assessment."""

    blur_score: float          # 0-100, higher = sharper
    brightness_score: float    # 0-100, higher = better exposed
    contrast_score: float      # 0-100, higher = better dynamic range
    face_size_score: float     # 0-100, higher = larger/closer face
    resolution_score: float    # 0-100, higher = more pixels
    overall: float = 0.0       # 0-100 capture quality index
    issues: list[Reason] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "blur": round(self.blur_score, 1),
            "brightness": round(self.brightness_score, 1),
            "contrast": round(self.contrast_score, 1),
            "face_size": round(self.face_size_score, 1),
            "resolution": round(self.resolution_score, 1),
            "overall": round(self.overall, 1),
            "issues": [i.model_dump() for i in self.issues],
        }


def blur_metric(img: np.ndarray) -> float:
    """Variance of the Laplacian — a classic focus/sharpness measure."""
    gray = to_gray(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_metric(img: np.ndarray) -> float:
    """Mean luminance in [0, 255]."""
    return float(np.mean(to_gray(img)))


def contrast_metric(img: np.ndarray) -> float:
    """Std of luminance — a proxy for dynamic range."""
    return float(np.std(to_gray(img)))


def assess_quality(
    img: np.ndarray,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> QualityReport:
    """Produce a :class:`QualityReport` for a single frame.

    ``face_bbox`` (x1, y1, x2, y2) lets us reward an adequately sized face.
    """
    h, w = img.shape[:2]
    issues: list[Reason] = []

    # --- Sharpness ---
    blur_raw = blur_metric(img)
    blur_score = linear_map(blur_raw, 20.0, 200.0, 0.0, 100.0)
    if blur_raw < 60:
        issues.append(rc.BLUR("Image appears blurry; hold the camera steady."))

    # --- Brightness ---
    bright_raw = brightness_metric(img)
    if bright_raw < 60:
        brightness_score = linear_map(bright_raw, 10.0, 60.0, 0.0, 70.0)
        issues.append(rc.LOW_LIGHT("Lighting is too dark; increase ambient light."))
    elif bright_raw > 200:
        brightness_score = linear_map(bright_raw, 255.0, 200.0, 0.0, 70.0)
        issues.append(rc.OVEREXPOSED("Image is overexposed; reduce direct light."))
    else:
        brightness_score = linear_map(abs(bright_raw - 128.0), 70.0, 0.0, 60.0, 100.0)

    # --- Contrast ---
    contrast_raw = contrast_metric(img)
    contrast_score = linear_map(contrast_raw, 20.0, 70.0, 0.0, 100.0)
    if contrast_raw < 25:
        issues.append(rc.LOW_CONTRAST("Low contrast; avoid flat or backlit scenes."))

    # --- Resolution ---
    resolution_score = linear_map(min(h, w), 240.0, 720.0, 30.0, 100.0)
    if min(h, w) < 240:
        issues.append(rc.LOW_RESOLUTION("Low resolution capture; move to a better camera."))

    # --- Face size ---
    if face_bbox is not None:
        fx = face_bbox[2] - face_bbox[0]
        fy = face_bbox[3] - face_bbox[1]
        face_frac = (fx * fy) / float(w * h)
        face_size_score = linear_map(face_frac, 0.02, 0.25, 0.0, 100.0)
        if face_frac < 0.05:
            issues.append(rc.FACE_TOO_SMALL("Face is too small; move closer to the camera."))
    else:
        face_size_score = 50.0  # neutral when no face context available

    report = QualityReport(
        blur_score=clamp(blur_score),
        brightness_score=clamp(brightness_score),
        contrast_score=clamp(contrast_score),
        face_size_score=clamp(face_size_score),
        resolution_score=clamp(resolution_score),
        issues=issues,
    )
    report.overall = capture_quality_index(report)
    return report


def capture_quality_index(q: QualityReport) -> float:
    """Weighted aggregate capture-quality index (0-100)."""
    return clamp(
        0.30 * q.blur_score
        + 0.25 * q.brightness_score
        + 0.15 * q.contrast_score
        + 0.20 * q.face_size_score
        + 0.10 * q.resolution_score
    )


def confidence_penalty(quality_index: float) -> float:
    """Map a capture quality index to a fractional confidence penalty [0,1].

    A perfect capture (100) yields 0 penalty; very poor captures approach a
    capped 0.30 penalty. Used to explain (not silently cause) lower scores.
    """
    if quality_index >= 85:
        return 0.0
    return clamp(linear_map(quality_index, 85.0, 30.0, 0.0, 0.30), 0.0, 0.30)
