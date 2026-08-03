"""Face alignment to a canonical pose before embedding (Upgrade 1).

Aligning every face to the standard ArcFace 112x112 template (via a similarity
transform on the 5 facial keypoints) removes pose / tilt / scale variation so
the recogniser sees faces in a consistent frame. This noticeably improves
genuine-pair similarity — especially across selfies vs. ID photos at different
angles — and is cheap (a single warpAffine).

We reuse InsightFace's well-tested ``norm_crop`` (the exact transform ArcFace
was trained with) rather than re-deriving the template, which keeps embeddings
in-distribution for the model.
"""
from __future__ import annotations

import cv2
import numpy as np
from insightface.utils import face_align


def align_face(image: np.ndarray, landmarks, size: int = 112) -> np.ndarray:
    """Align a face to a canonical ``size`` x ``size`` crop using 5 keypoints.

    ``landmarks`` is the 5-point array (left-eye, right-eye, nose, left-mouth,
    right-mouth) in pixel coordinates of ``image``.
    """
    kps = np.asarray(landmarks, dtype=np.float32)
    if kps.shape != (5, 2):
        raise ValueError(f"Expected 5x2 landmarks, got {kps.shape}")
    return face_align.norm_crop(image, landmark=kps, image_size=size)


def get_aligned_face(image: np.ndarray, face, size: int = 112) -> np.ndarray:
    """Align using an InsightFace ``face`` object's keypoints (``face.kps``).

    Falls back to a plain bbox crop resized to ``size`` if keypoints are
    unavailable (rare), so callers always get a usable crop.
    """
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 5:
        return align_face(image, kps[:5], size)
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)


__all__ = ["align_face", "get_aligned_face"]
