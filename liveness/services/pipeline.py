"""End-to-end verification orchestrator.

Wires Layer 1 -> Layer 2 -> Layer 3 -> scoring -> reporting and supplies the
quality-compensation context. Used by the ``/verify`` endpoint and reused by
the per-layer endpoints where convenient.
"""
from __future__ import annotations

import time

import numpy as np

from liveness.config import settings
from liveness.schemas.requests import CameraMetadata, ChallengeType
from liveness.schemas.responses import (
    ModelInfo,
    ProcessingInfo,
    Reason,
    VerificationResponse,
)
from liveness.services.layer1_capture import analyze_capture
from liveness.services.layer2_identity import verify_identity
from liveness.services.layer3_liveness import verify_liveness
from liveness.services.reporting import build_report
from liveness.utils.face_engine import detect_faces
from liveness.utils.image import sample_frames
from liveness.utils.logger import logger
from liveness.utils.quality import assess_quality, blur_metric


def _best_probe_frame(frames: list[np.ndarray]) -> np.ndarray:
    """Pick the sharpest frame that actually contains a detectable face.

    Choosing purely the sharpest frame can land on a mid-blink / turned-away /
    motion-blurred frame with no usable face. We rank candidates by sharpness
    and return the first one with a confident face, falling back to the
    sharpest frame overall.
    """
    if len(frames) == 1:
        return frames[0]
    sampled = sample_frames(frames, 12)
    ranked = sorted(sampled, key=blur_metric, reverse=True)
    for frame in ranked[:6]:  # cap detection cost
        faces = detect_faces(frame)
        if faces and float(faces[0].det_score) >= 0.5:
            return frame
    return ranked[0]


def _aggregate_quality(frames: list[np.ndarray]) -> tuple[float, list[Reason]]:
    sampled = sample_frames(frames, 8)
    reports = [assess_quality(f) for f in sampled]
    idx = float(np.mean([r.overall for r in reports]))
    issues: list[Reason] = []
    seen: set[str] = set()
    for r in reports:
        for issue in r.issues:
            # Dedupe on the code, not the prose: the same underlying problem can
            # be worded with different interpolated values across frames.
            if issue.code not in seen:
                seen.add(issue.code)
                issues.append(issue)
    return idx, issues


def _model_info() -> ModelInfo:
    """The models that actually ran, named from live config rather than hardcoded.

    ``insightface_model`` is env-overridable, so a deployment running a different
    pack must not have its responses claim the default one.
    """
    pack = settings.insightface_model
    return ModelInfo(
        face_detector=f"insightface_{pack}_scrfd_det_10g",
        face_embedding=f"insightface_{pack}_arcface_w600k_r50",
        face_landmarks="mediapipe_face_landmarker_v2",
    )


def run_full_pipeline(
    reference_img: np.ndarray,
    frames: list[np.ndarray],
    timestamps_ms: list[float],
    metadata: CameraMetadata,
    challenge: ChallengeType,
    fps: float,
    challenges: list[ChallengeType] | None = None,
    mirrored: bool | None = None,
) -> VerificationResponse:
    """Run all three layers and return the composed, explainable report."""
    logger.info("Running full verification pipeline over {} frames.", len(frames))
    frame_count = len(frames)
    t0 = time.monotonic()

    capture = analyze_capture(frames, timestamps_ms, metadata, frame_count=frame_count)
    t_capture = time.monotonic()

    probe = _best_probe_frame(frames)
    identity = verify_identity(reference_img, probe)
    t_identity = time.monotonic()

    liveness = verify_liveness(
        frames, challenge=challenge, fps=fps, challenges=challenges, mirrored=mirrored,
    )
    t_liveness = time.monotonic()

    quality_index, quality_issues = _aggregate_quality(frames)

    report = build_report(
        capture=capture,
        identity=identity,
        liveness=liveness,
        quality_index=quality_index,
        quality_issues=quality_issues,
    )
    t_end = time.monotonic()

    report.processing = ProcessingInfo(
        total_ms=_ms(t0, t_end),
        capture_ms=_ms(t0, t_capture),
        identity_ms=_ms(t_capture, t_identity),
        liveness_ms=_ms(t_identity, t_liveness),
        # Quality aggregation + fusion + report assembly: everything between the
        # last layer finishing and the decision existing.
        decision_ms=_ms(t_liveness, t_end),
    )
    report.models = _model_info()
    return report


def _ms(start: float, end: float) -> int:
    return int(round((end - start) * 1000.0))
