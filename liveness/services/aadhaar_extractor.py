"""Aadhaar / ID portrait extraction (Upgrade 3).

Real users upload the *whole* card (or a phone photo of it), not a pre-cropped
face. This module isolates the portrait region so downstream alignment +
embedding operate on a clean, high-resolution face.

Design choice: extraction is **face-driven**, not layout-driven. Running the
detector across orientations and cropping around the strongest face is far more
robust to old/new Aadhaar layouts, scans, phone photos and perspective skew than
brittle template/contour heuristics — and it inherently validates that a face is
present. For small faces (a tiny portrait inside a big card photo) we upscale +
restore the portrait region and re-detect to recover detail before embedding.

No OCR / field extraction is performed (out of scope by request).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from liveness.utils.face_engine import detect_faces, detect_oriented, primary_face
from liveness.utils.image import crop_bbox, enhance_for_recognition
from liveness.utils.logger import logger

# Below this face-area fraction the face is considered a small portrait inside a
# larger document/scene, worth cropping + upscaling before embedding.
SMALL_FACE_FRACTION = 0.12
PORTRAIT_PAD = 0.6          # padding around the face when cropping the portrait
UPSCALE_TARGET = 320        # upscale small portraits to ~this on the long side


@dataclass
class PortraitResult:
    image: np.ndarray            # prepared portrait (oriented, possibly enhanced)
    face: object                 # InsightFace face within ``image``
    bbox: tuple[int, int, int, int]
    angle: int                   # rotation applied to the source (degrees)
    confidence: float            # detection score of the portrait face
    enhanced: bool               # whether restoration was applied
    cropped: bool                # whether a portrait sub-region was cropped


def detect_aadhaar(image: np.ndarray):
    """Orientation-correct the card and locate its strongest face.

    Returns ``(oriented_image, face, angle, score)`` (face may be ``None``).
    """
    return detect_oriented(image)


def _face_area_fraction(face, image: np.ndarray) -> float:
    x1, y1, x2, y2 = face.bbox
    fa = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    h, w = image.shape[:2]
    return fa / float(w * h + 1e-6)


def validate_portrait(portrait: np.ndarray):
    """Confirm a face exists in ``portrait``; return the subject face or ``None``."""
    return primary_face(detect_faces(portrait))


def extract_portrait(image: np.ndarray) -> PortraitResult | None:
    """Extract a clean portrait face from a full card / photo.

    Returns a :class:`PortraitResult`, or ``None`` if no face can be found in
    any orientation. Large faces (already a face photo) are returned as-is;
    small faces are cropped, restored and upscaled, then re-validated.
    """
    oriented, face, angle, score = detect_aadhaar(image)
    if face is None:
        logger.debug("Aadhaar extractor: no face found in any orientation.")
        return None

    frac = _face_area_fraction(face, oriented)
    if frac >= SMALL_FACE_FRACTION:
        # Already a sufficiently large face — no portrait surgery needed.
        bbox = tuple(int(v) for v in face.bbox)
        return PortraitResult(oriented, face, bbox, angle, score, False, False)

    # Small portrait inside a bigger image: crop around it, restore + upscale,
    # then re-detect to recover a higher-quality face.
    bbox = tuple(int(v) for v in face.bbox)
    portrait = crop_bbox(oriented, bbox, pad=PORTRAIT_PAD)
    portrait = enhance_for_recognition(portrait, upscale_to=UPSCALE_TARGET)
    refined = validate_portrait(portrait)
    if refined is None:
        # Enhancement lost the face; fall back to the raw oriented detection.
        return PortraitResult(oriented, face, bbox, angle, score, False, False)
    rbbox = tuple(int(v) for v in refined.bbox)
    logger.debug("Aadhaar extractor: cropped+restored small portrait (frac={:.3f}).", frac)
    return PortraitResult(portrait, refined, rbbox, angle,
                          float(refined.det_score), True, True)


__all__ = ["PortraitResult", "detect_aadhaar", "extract_portrait", "validate_portrait"]
