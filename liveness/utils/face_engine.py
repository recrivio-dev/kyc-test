"""Lazy-loaded face analysis engines (MediaPipe + InsightFace).

Two singletons are exposed:

* :func:`get_landmarker` - a MediaPipe ``FaceLandmarker`` (478 landmarks,
  blendshapes, head-pose matrix) used by Layer 3 liveness.
* :func:`get_arcface`    - an InsightFace ``FaceAnalysis`` app (detection +
  ArcFace embeddings) used by Layer 2 identity.

Both are loaded on first use and cached, keeping import time low and the API
process responsive. Everything runs on CPU.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from liveness.config import settings
from liveness.utils.image import to_rgb
from liveness.utils.logger import logger
from liveness.utils.model_manager import (
    ALLOWED_MODULES,
    ensure_insightface_model,
    ensure_mediapipe_models,
)

_LOCK = threading.Lock()
# MediaPipe FaceLandmarker (IMAGE mode) and the InsightFace app are NOT safe for
# concurrent inference on a single instance. FastAPI dispatches requests on a
# threadpool, so we serialise the actual model calls with a dedicated lock.
_INFER_LOCK = threading.Lock()
_LANDMARKER = None
_ARCFACE = None


# --------------------------------------------------------------------------- #
# MediaPipe FaceLandmarker
# --------------------------------------------------------------------------- #
@dataclass
class LandmarkResult:
    """Landmarks + derived geometry for a single frame."""

    landmarks: np.ndarray            # (N, 3) pixel coords (x, y, z*w)
    normalized: np.ndarray           # (N, 3) normalized coords in [0,1]
    blendshapes: dict[str, float]    # name -> score
    pose: dict[str, float]           # yaw, pitch, roll in degrees
    image_size: tuple[int, int]      # (w, h)

    @property
    def found(self) -> bool:
        return self.landmarks.size > 0


def get_landmarker():
    """Return a cached MediaPipe FaceLandmarker (IMAGE mode)."""
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER
    with _LOCK:
        if _LANDMARKER is not None:
            return _LANDMARKER
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        paths = ensure_mediapipe_models()
        model_path = str(paths["face_landmarker"])
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(options)
        logger.success("MediaPipe FaceLandmarker loaded.")
        return _LANDMARKER


def _matrix_to_euler(mat: np.ndarray) -> dict[str, float]:
    """Extract yaw/pitch/roll (degrees) from a 4x4 transform matrix."""
    r = mat[:3, :3]
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
        yaw = np.degrees(np.arctan2(-r[2, 0], sy))
        roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-r[1, 2], r[1, 1]))
        yaw = np.degrees(np.arctan2(-r[2, 0], sy))
        roll = 0.0
    return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}


def detect_landmarks(img_bgr: np.ndarray) -> LandmarkResult:
    """Run the landmarker on a single BGR frame."""
    import mediapipe as mp

    h, w = img_bgr.shape[:2]
    landmarker = get_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=to_rgb(img_bgr))
    with _INFER_LOCK:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return LandmarkResult(
            landmarks=np.empty((0, 3)),
            normalized=np.empty((0, 3)),
            blendshapes={},
            pose={"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            image_size=(w, h),
        )

    lms = result.face_landmarks[0]
    norm = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
    pix = norm.copy()
    pix[:, 0] *= w
    pix[:, 1] *= h
    pix[:, 2] *= w  # approximate depth scaling

    blend: dict[str, float] = {}
    if result.face_blendshapes:
        for cat in result.face_blendshapes[0]:
            blend[cat.category_name] = float(cat.score)

    pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    if result.facial_transformation_matrixes:
        mat = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
        pose = _matrix_to_euler(mat)

    return LandmarkResult(
        landmarks=pix, normalized=norm, blendshapes=blend, pose=pose, image_size=(w, h)
    )


# --------------------------------------------------------------------------- #
# InsightFace ArcFace
# --------------------------------------------------------------------------- #
def get_arcface():
    """Return a cached InsightFace FaceAnalysis app (CPU)."""
    global _ARCFACE
    if _ARCFACE is not None:
        return _ARCFACE
    with _LOCK:
        if _ARCFACE is not None:
            return _ARCFACE
        from insightface.app import FaceAnalysis

        ensure_insightface_model()
        root = str(settings.models_path / "insightface")
        app = FaceAnalysis(
            name=settings.insightface_model,
            root=root,
            providers=settings.providers,
            # Only detection + recognition + the 2D landmarks face_align needs.
            # The default pack also loads 1k3d68 (137 MB) and genderage, neither
            # of which any code path here calls — and this process shares a 3 GB
            # container with the OCR pipeline. See ALLOWED_MODULES.
            allowed_modules=ALLOWED_MODULES,
        )
        app.prepare(ctx_id=-1, det_size=(settings.det_size, settings.det_size))
        _ARCFACE = app
        logger.success("InsightFace ArcFace app loaded (CPU).")
        return _ARCFACE


def detect_faces(img_bgr: np.ndarray):
    """Return InsightFace face objects (sorted by detection score desc)."""
    app = get_arcface()
    with _INFER_LOCK:
        faces = app.get(img_bgr)
    return sorted(faces, key=lambda f: float(f.det_score), reverse=True)


# In-plane orientations tried when a face isn't found upright. ID-card photos
# are commonly shot sideways / upside-down.
_ROTATIONS = (
    (0, None),
    (90, cv2.ROTATE_90_CLOCKWISE),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    (180, cv2.ROTATE_180),
)


def _face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def primary_face(faces, min_score: float = 0.35):
    """Select the *subject* face: the largest face (closest to camera / the ID
    portrait), not merely the highest-scoring one.

    A selfie often contains small background faces that can out-score the main
    subject; an ID card has one portrait. Ranking by area (gated by a minimum
    detection score) reliably picks the intended person.
    """
    if not faces:
        return None
    eligible = [f for f in faces if float(f.det_score) >= min_score] or list(faces)
    return max(eligible, key=_face_area)


def detect_oriented(img_bgr: np.ndarray):
    """Detect the *subject* face across 0/90/270/180 rotations.

    Returns ``(oriented_img, subject_face, angle, score)`` for the orientation
    whose primary (largest) face has the highest detection score. Short-circuits
    on an upright, confident detection. Shared by identity verification and the
    Aadhaar portrait extractor so the rotation logic lives in one place.
    """
    best = (img_bgr, None, 0, -1.0)
    for angle, code in _ROTATIONS:
        rot = img_bgr if code is None else cv2.rotate(img_bgr, code)
        face = primary_face(detect_faces(rot))
        if face is None:
            continue
        score = float(face.det_score)
        if score > best[3]:
            best = (rot, face, angle, score)
        if angle == 0 and score >= 0.6:
            break
    return best


def get_recognizer():
    """Return the InsightFace ArcFace recognition sub-model."""
    app = get_arcface()
    rec = app.models.get("recognition")
    if rec is None:
        raise RuntimeError("InsightFace recognition model not available.")
    return rec


def embed_aligned(aligned):
    """Embed one or a list of pre-aligned 112x112 BGR faces (ArcFace).

    Accepts a single image or a list (batched for efficiency) and returns a
    ``(512,)`` vector or ``(N, 512)`` array respectively. Embeddings are raw
    (not normalised); callers L2-normalise as needed.
    """
    rec = get_recognizer()
    single = not isinstance(aligned, list)
    imgs = [aligned] if single else aligned
    with _INFER_LOCK:
        feats = rec.get_feat(imgs)
    feats = np.asarray(feats).reshape(len(imgs), -1)
    return feats[0] if single else feats


__all__ = [
    "LandmarkResult",
    "detect_landmarks",
    "detect_faces",
    "detect_oriented",
    "primary_face",
    "get_landmarker",
    "get_arcface",
    "get_recognizer",
    "embed_aligned",
]
