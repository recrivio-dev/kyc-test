"""Layer 2 - Identity Verification.

Confirms the live user matches a reference ID photo (Aadhaar / Passport /
License). Designed to be robust to large age gaps, beards, hairstyle and
weight changes via:

* **Multi-crop embeddings** - full / upper-face / lower-face crops are embedded
  and averaged, reducing the impact of localised appearance changes.
* **Quality-adaptive thresholding** - old / low-quality references use a more
  forgiving similarity threshold.
* **Landmark geometry** - interocular and nose geometry stay stable across
  years and provide secondary evidence.

Final identity score = 70% ArcFace similarity + 20% landmark geometry +
10% image-quality confidence.
"""
from __future__ import annotations

import cv2
import numpy as np

from liveness.config import settings
from liveness.schemas.responses import IdentityResult, Reason
from liveness.services import reason_codes as rc
from liveness.services.aadhaar_extractor import extract_portrait
from liveness.utils.face_alignment import get_aligned_face
from liveness.utils.face_engine import detect_faces, embed_aligned
from liveness.utils.image import enhance_for_recognition
from liveness.utils.logger import logger
from liveness.utils.metrics import clamp, cosine_similarity, l2_normalize, linear_map
from liveness.utils.quality import assess_quality


def _prepare_face(img: np.ndarray):
    """Locate and prepare a face for matching (orientation + portrait + restore).

    Returns ``(prepared_image, face, angle, enhanced, cropped)`` or ``None`` if
    no face is found. Handles sideways cards, full-card uploads (portrait
    extraction) and faded/low-contrast photos in one place, shared by reference
    and probe.
    """
    res = extract_portrait(img)
    if res is None:
        return None
    return res.image, res.face, res.angle, res.enhanced, res.cropped


def _robust_embedding(img: np.ndarray, face) -> np.ndarray:
    """Aging-/degradation-robust embedding from a single detected face.

    The face is aligned to the canonical ArcFace 112x112 template, then embedded
    from four views — aligned, horizontally-flipped (TTA), restored, and
    restored+flipped — in one batched recognition call. Averaging the
    L2-normalised embeddings suppresses pose, lighting and degradation effects,
    raising genuine-pair similarity. Alignment + direct recognition is also
    faster than re-detecting each view.
    """
    aligned = get_aligned_face(img, face, size=112)
    restored = enhance_for_recognition(aligned, upscale_to=112)
    views = [aligned, cv2.flip(aligned, 1), restored, cv2.flip(restored, 1)]
    feats = embed_aligned(views)  # (4, 512), batched
    embs = [l2_normalize(f) for f in np.asarray(feats)]
    mean = np.mean(np.stack(embs, axis=0), axis=0)
    return l2_normalize(mean)


def _periocular_embedding(img: np.ndarray, face) -> np.ndarray:
    """Age-stable embedding from the periocular (eye/brow) region.

    With a large age gap the lower face (jaw/chin/mouth) matures and drags the
    whole-face ArcFace cosine below threshold, even for the same person. The
    periocular region changes far less, so it provides a second corroborating
    channel. We crop the top ``periocular_crop_fraction`` rows of the canonical
    112x112 aligned face, resize back to 112x112, and embed it with horizontal
    -flip TTA (2 views), then L2-normalise and average.
    """
    aligned = get_aligned_face(img, face, size=112)
    rows = max(1, min(112, int(round(112 * settings.periocular_crop_fraction))))
    crop = aligned[:rows, :, :]
    crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_CUBIC)
    views = [crop, cv2.flip(crop, 1)]
    feats = embed_aligned(views)  # (2, 512), batched
    embs = [l2_normalize(f) for f in np.asarray(feats)]
    mean = np.mean(np.stack(embs, axis=0), axis=0)
    return l2_normalize(mean)


def _geometry_features(face) -> np.ndarray | None:
    """Stable geometric ratios from the 5 landmark keypoints.

    kps order: left-eye, right-eye, nose, left-mouth, right-mouth.
    Ratios are scale/translation invariant so they survive aging & distance.
    """
    if getattr(face, "kps", None) is None or len(face.kps) < 5:
        return None
    kp = np.asarray(face.kps, dtype=np.float64)
    le, re, nose, lm, rm = kp[0], kp[1], kp[2], kp[3], kp[4]
    interocular = float(np.linalg.norm(re - le)) + 1e-6
    eye_mid = (le + re) / 2.0
    mouth_mid = (lm + rm) / 2.0
    return np.array([
        np.linalg.norm(nose - eye_mid) / interocular,    # eye->nose vertical span
        np.linalg.norm(mouth_mid - eye_mid) / interocular,  # eye->mouth span
        np.linalg.norm(rm - lm) / interocular,           # mouth width ratio
        np.linalg.norm(nose - mouth_mid) / interocular,  # nose->mouth span
    ], dtype=np.float64)


def _landmark_stability(ref_face, probe_face) -> float:
    a = _geometry_features(ref_face)
    b = _geometry_features(probe_face)
    if a is None or b is None:
        return 60.0  # neutral when geometry unavailable
    rel_err = np.abs(a - b) / (np.abs(a) + 1e-6)
    mean_err = float(np.mean(rel_err))
    # 0% error -> 100, 25%+ error -> 0.
    return clamp(linear_map(mean_err, 0.0, 0.25, 100.0, 0.0))


def _adaptive_threshold(ref_quality: float, probe_quality: float) -> float:
    """Lower the similarity threshold for poor-quality references.

    Very degraded references (old, faded ID prints) get a notably more forgiving
    floor; crisp captures keep the strict threshold. The floor sits below the
    configured ``similarity_threshold_low`` so genuine matches on heavily
    degraded photos remain reachable after enhancement + TTA.
    """
    q = min(ref_quality, probe_quality)
    hi, lo = settings.similarity_threshold_high, settings.similarity_threshold_low
    floor = max(0.26, lo - 0.04)  # extra slack for very degraded references
    # High quality (>=78) -> hi threshold; very low quality (<=30) -> floor.
    return float(round(linear_map(q, 30.0, 78.0, floor, hi), 3))


def verify_identity(reference_img: np.ndarray, probe_img: np.ndarray) -> IdentityResult:
    """Compare a reference ID image with a live probe image."""
    reasons: list[Reason] = []

    # More than one face in the LIVE capture is a fraud signal worth surfacing:
    # a second person coaching or standing in for the candidate. Checked on the
    # raw probe before portrait extraction narrows it to a single face. The
    # reference is deliberately not checked — ID cards legitimately carry a
    # ghost portrait alongside the main one.
    multiple_faces = len(detect_faces(probe_img)) > 1
    if multiple_faces:
        reasons.append(rc.MULTIPLE_FACES(
            "More than one face is visible in the live capture."))

    # Prepare both faces: orientation correction + Aadhaar portrait extraction
    # (full-card uploads) + restoration of small/faded portraits, all shared.
    ref_prep = _prepare_face(reference_img)
    probe_prep = _prepare_face(probe_img)

    if ref_prep is None:
        reasons.append(rc.NO_FACE_IN_REFERENCE(
            "No face detected in the reference image (tried all rotations)."))
        return IdentityResult(
            identity_score=0.0, similarity=0.0, threshold=settings.similarity_threshold_high,
            match=False, landmark_score=0.0, quality_score=0.0,
            embedding_distance=1.0, multiple_faces=multiple_faces, reasons=reasons,
        )
    if probe_prep is None:
        reasons.append(rc.NO_FACE_IN_PROBE(
            "No face detected in the live capture (tried all rotations)."))
        return IdentityResult(
            identity_score=0.0, similarity=0.0, threshold=settings.similarity_threshold_high,
            match=False, landmark_score=0.0, quality_score=0.0,
            embedding_distance=1.0, multiple_faces=multiple_faces, reasons=reasons,
        )

    reference_img, ref_face, ref_angle, ref_enh, ref_crop = ref_prep
    probe_img, probe_face, probe_angle, _, _ = probe_prep

    if ref_angle:
        reasons.append(rc.REFERENCE_AUTO_ROTATED(
            f"Reference image auto-rotated {ref_angle}° to locate the face."))
    if probe_angle:
        reasons.append(rc.PROBE_AUTO_ROTATED(
            f"Live capture auto-rotated {probe_angle}° to locate the face."))
    if ref_crop:
        reasons.append(rc.PORTRAIT_EXTRACTED(
            "Portrait region extracted from the document and restored before matching."))
    elif ref_enh:
        reasons.append(rc.REFERENCE_RESTORED(
            "Reference image was restored (white-balance/contrast) before matching."))

    ref_emb = _robust_embedding(reference_img, ref_face)
    probe_emb = _robust_embedding(probe_img, probe_face)
    similarity = cosine_similarity(ref_emb, probe_emb)

    # Age-stable periocular (eye-region) channel — corroborating evidence for
    # large age-gap pairs whose lower face has matured.
    ref_peri = _periocular_embedding(reference_img, ref_face)
    probe_peri = _periocular_embedding(probe_img, probe_face)
    periocular_sim = cosine_similarity(ref_peri, probe_peri)

    ref_bbox = tuple(int(v) for v in ref_face.bbox)
    probe_bbox = tuple(int(v) for v in probe_face.bbox)
    ref_q = assess_quality(reference_img, ref_bbox).overall
    probe_q = assess_quality(probe_img, probe_bbox).overall
    quality_score = float(min(ref_q, probe_q))

    threshold = _adaptive_threshold(ref_q, probe_q)
    landmark_score = _landmark_stability(ref_face, probe_face)

    # Multi-signal decision: a direct embedding pass, or a borderline embedding
    # corroborated by an age-stable channel — consistent facial geometry, or a
    # strong periocular (eye-region) match. Both gates are strict conjunctions
    # kept above the impostor band (≈0.15), so corroboration stays low-risk and
    # the global threshold is never lowered.
    match = similarity >= threshold
    geom_corroborated = (
        not match
        and similarity >= max(0.33, threshold - 0.06)
        and landmark_score >= 72.0
    )
    # Periocular channel: the lower face matured (full-face cosine dropped) but
    # the age-stable eye region still matches clearly above the full-face score.
    peri_corroborated = (
        not match
        and similarity >= settings.similarity_threshold_low - 0.04
        and periocular_sim >= settings.periocular_match_threshold
        and periocular_sim >= similarity + settings.periocular_margin
    )
    if geom_corroborated or peri_corroborated:
        match = True

    # Threshold-relative score: a genuine match *at* the (quality-adaptive)
    # threshold should already read as a solid pass, scaling up with margin.
    # This stops degraded-but-genuine ID matches from being unfairly low. When
    # the periocular channel carries the decision, credit it so the genuine
    # match doesn't read as a borderline score.
    effective_sim = similarity
    if peri_corroborated:
        effective_sim = max(similarity, 0.6 * similarity + 0.4 * periocular_sim)
    delta = effective_sim - threshold
    if delta >= 0:
        sim_component = clamp(65.0 + linear_map(delta, 0.0, 0.20, 0.0, 35.0))
    else:
        sim_component = clamp(linear_map(delta, -0.20, 0.0, 0.0, 65.0))
    identity_score = clamp(
        0.70 * sim_component + 0.20 * landmark_score + 0.10 * quality_score
    )

    if geom_corroborated:
        reasons.append(rc.FACE_MATCH_GEOMETRY_CORROBORATED(
            f"Borderline embedding similarity {similarity:.2f} confirmed by consistent "
            f"facial geometry (age-stable) — accepted."
        ))
    elif peri_corroborated:
        reasons.append(rc.FACE_MATCH_PERIOCULAR_CORROBORATED(
            f"Borderline full-face similarity {similarity:.2f} confirmed by a strong "
            f"age-stable periocular (eye-region) match {periocular_sim:.2f} — consistent "
            f"with the same person across a large age gap."
        ))
    elif match:
        reasons.append(rc.FACE_MATCH_PASS(
            f"Strong embedding match (similarity {similarity:.2f} >= {threshold:.2f})."))
    else:
        reasons.append(rc.FACE_MATCH_FAIL(
            f"Embedding similarity {similarity:.2f} below adaptive threshold {threshold:.2f}."))
    if threshold < settings.similarity_threshold_high:
        reasons.append(rc.FORGIVING_THRESHOLD(
            "Reference appears low-quality/aged; using a forgiving threshold."))
    if landmark_score >= 75:
        reasons.append(rc.GEOMETRY_CONSISTENT(
            "Facial geometry remained highly consistent (age-stable)."))
    elif landmark_score < 50:
        reasons.append(rc.GEOMETRY_DIVERGENT(
            "Facial geometry differs notably between images."))

    logger.info(
        "Layer2 identity_score={:.1f} sim={:.3f} peri={:.3f} thr={:.3f} match={} geom={:.1f} q={:.1f}",
        identity_score, similarity, periocular_sim, threshold, match, landmark_score, quality_score,
    )

    return IdentityResult(
        identity_score=round(identity_score, 1),
        similarity=round(float(similarity), 3),
        threshold=threshold,
        match=bool(match),
        landmark_score=round(landmark_score, 1),
        quality_score=round(quality_score, 1),
        embedding_distance=round(1.0 - float(similarity), 3),
        multiple_faces=multiple_faces,
        reasons=reasons,
    )
