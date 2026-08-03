"""Final Decision Engine.

Fuses the three layer scores into a single risk-weighted final score and maps
it onto the decision bands. Also derives an overall confidence level which is
informed by the capture-quality compensation system.
"""
from __future__ import annotations

from dataclasses import dataclass

from liveness.config import settings
from liveness.schemas.responses import ConfidenceLevel, DecisionStatus
from liveness.utils.metrics import clamp


@dataclass
class ScoreBreakdown:
    capture_score: float
    identity_score: float
    liveness_score: float
    final_score: float
    status: DecisionStatus
    confidence: ConfidenceLevel


def compute_final_score(capture: float, identity: float, liveness: float) -> float:
    """Weighted fusion: 0.20*capture + 0.35*identity + 0.45*liveness."""
    return clamp(
        settings.weight_capture * capture
        + settings.weight_identity * identity
        + settings.weight_liveness * liveness
    )


def decide_status(final_score: float) -> DecisionStatus:
    """Map the final score onto the decision bands."""
    if final_score >= settings.threshold_verified:
        return DecisionStatus.VERIFIED
    if final_score >= settings.threshold_high:
        return DecisionStatus.VERIFIED_HIGH
    if final_score >= settings.threshold_medium:
        return DecisionStatus.VERIFIED_MEDIUM
    if final_score >= settings.threshold_review:
        return DecisionStatus.MANUAL_REVIEW
    return DecisionStatus.REJECTED


def derive_confidence(
    final_score: float, quality_penalty_pct: float, identity_match: bool
) -> ConfidenceLevel:
    """Derive a confidence level from score, quality penalty and identity match."""
    if not identity_match:
        return ConfidenceLevel.LOW
    if final_score >= settings.threshold_high and quality_penalty_pct < 10:
        return ConfidenceLevel.HIGH
    if final_score >= settings.threshold_medium and quality_penalty_pct < 25:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def score_pipeline(
    capture_score: float,
    identity_score: float,
    liveness_score: float,
    quality_penalty_pct: float,
    identity_match: bool,
) -> ScoreBreakdown:
    final = compute_final_score(capture_score, identity_score, liveness_score)
    status = decide_status(final)
    confidence = derive_confidence(final, quality_penalty_pct, identity_match)
    return ScoreBreakdown(
        capture_score=round(capture_score, 1),
        identity_score=round(identity_score, 1),
        liveness_score=round(liveness_score, 1),
        final_score=round(final, 1),
        status=status,
        confidence=confidence,
    )
