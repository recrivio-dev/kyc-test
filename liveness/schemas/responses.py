"""Response schemas for the verification API.

Every layer returns a sub-result with its score, granular sub-scores and a list
of reasons. The final response composes them with the overall decision,
confidence and warnings.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from liveness.config import PIPELINE_VERSION
from liveness.schemas.requests import ChallengeType

ReasonSeverity = Literal["low", "medium", "high"]


class Reason(BaseModel):
    """One explained contribution to a layer's score.

    Carries a stable ``code`` for machines and the original prose ``message``
    for humans — see ``liveness/services/reason_codes.py`` for why both exist
    and what ``severity`` means.
    """

    code: str = Field(..., description="Stable machine-readable key; safe to branch on")
    message: str = Field(..., description="User-facing prose; safe to show a candidate")
    severity: ReasonSeverity = "low"


class DecisionStatus(str, Enum):
    VERIFIED = "verified"
    VERIFIED_HIGH = "verified_high_confidence"
    VERIFIED_MEDIUM = "verified_medium_confidence"
    MANUAL_REVIEW = "manual_review"
    REJECTED = "rejected"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CameraType(str, Enum):
    """What the Layer 1 metadata check concluded about the capture device."""

    PHYSICAL = "physical"
    VIRTUAL = "virtual"
    UNKNOWN = "unknown"


class CaptureResult(BaseModel):
    capture_score: float = Field(..., ge=0, le=100)
    timing_score: float = Field(..., ge=0, le=100)
    metadata_score: float = Field(..., ge=0, le=100)
    entropy_score: float = Field(..., ge=0, le=100)
    injection_score: float = Field(..., ge=0, le=100, description="Higher = more injection risk")
    median_fps: float = Field(default=0.0, ge=0)
    timing_variance: float = Field(
        default=0.0, ge=0,
        description="Normalised inter-frame jitter (MAD / median delta). ~0 is deterministic.",
    )
    camera_type: CameraType = Field(
        default=CameraType.UNKNOWN,
        description="`unknown` means no metadata was supplied, NOT that the camera is clean.",
    )
    frame_count: int = Field(default=0, ge=0, description="Frames received in the request")
    selected_frames: int = Field(
        default=0, ge=0, description="Frames actually analysed after sampling",
    )
    reasons: list[Reason] = Field(default_factory=list)


class IdentityResult(BaseModel):
    identity_score: float = Field(..., ge=0, le=100)
    similarity: float = Field(..., ge=-1, le=1)
    threshold: float = Field(..., description="Quality-adaptive decision threshold")
    match: bool = False
    landmark_score: float = Field(default=0.0, ge=0, le=100)
    quality_score: float = Field(default=0.0, ge=0, le=100)
    embedding_distance: float = Field(
        default=0.0, description="Cosine distance (1 - similarity); the inverse of `similarity`",
    )
    multiple_faces: bool = Field(
        default=False, description="More than one face found in the live capture",
    )
    reasons: list[Reason] = Field(default_factory=list)


class ChallengeStepResult(BaseModel):
    """One step of an ordered active-liveness challenge sequence."""

    challenge: ChallengeType
    passed: bool = False
    score: float = Field(default=0.0, ge=0, le=100)
    peak_frame_index: int = Field(
        default=-1,
        description="Index (within the analysed frames) where this action peaked; -1 if never found",
    )


class LivenessResult(BaseModel):
    liveness_score: float = Field(..., ge=0, le=100)
    position_score: float = Field(default=0.0, ge=0, le=100)
    lighting_score: float = Field(default=0.0, ge=0, le=100)
    blink_score: float = Field(default=0.0, ge=0, le=100)
    challenge_score: float = Field(default=0.0, ge=0, le=100)
    depth_score: float = Field(default=0.0, ge=0, le=100)
    motion_score: float = Field(default=0.0, ge=0, le=100)
    replay_resistance_score: float = Field(default=0.0, ge=0, le=100)
    blink_detected: bool = False
    challenge_passed: bool = False
    depth_passed: bool = False
    # Per-step detail for an ordered sequence. A single-challenge request still
    # populates this with one entry, so consumers only need one code path.
    challenge_sequence: list[ChallengeStepResult] = Field(default_factory=list)
    challenge_sequence_passed: bool = Field(
        default=False,
        description="Every challenge passed AND they occurred in the requested order",
    )
    rppg_bpm: float | None = Field(default=None, description="Advisory only; never gates")
    reasons: list[Reason] = Field(default_factory=list)


class FraudIndicators(BaseModel):
    """Only checks that are actually backed by a signal in this pipeline.

    Deliberately does NOT carry deepfake / face-swap / document-tampering keys:
    no model here evaluates them, and a hardcoded ``false`` reads to a reviewer
    as "checked and clean". Absent is honest; false is not.
    """

    virtual_camera: bool = False
    replay_attack: bool = False
    multiple_faces: bool = False


class QualityInfo(BaseModel):
    capture_quality_index: float = Field(..., ge=0, le=100)
    confidence_penalty_pct: float = Field(..., ge=0, le=100)
    issues: list[Reason] = Field(default_factory=list)


class ProcessingInfo(BaseModel):
    """Wall-clock cost per stage. Diagnostics only — never affects the decision."""

    total_ms: int = 0
    capture_ms: int = 0
    identity_ms: int = 0
    liveness_ms: int = 0
    decision_ms: int = 0


class ModelInfo(BaseModel):
    """The models that actually ran.

    There is no depth model and no spoof model: depth is derived geometrically
    from MediaPipe landmarks (homography residual under head rotation), and
    nothing here does dedicated spoof classification. Those keys are omitted
    rather than filled with a plausible-looking name.
    """

    face_detector: str
    face_embedding: str
    face_landmarks: str


class VerificationResponse(BaseModel):
    """Top-level response for the full pipeline."""

    status: DecisionStatus
    capture_score: float = Field(..., ge=0, le=100)
    identity_score: float = Field(..., ge=0, le=100)
    liveness_score: float = Field(..., ge=0, le=100)
    final_score: float = Field(..., ge=0, le=100)
    confidence: ConfidenceLevel
    reasons: list[Reason] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Detailed breakdown (the explainable report).
    capture: CaptureResult | None = None
    identity: IdentityResult | None = None
    liveness: LivenessResult | None = None
    quality: QualityInfo | None = None
    notes: list[str] = Field(default_factory=list)

    # Diagnostics. Consumers building a reviewer view can ignore these; they
    # exist so an engineer debugging a bad decision doesn't have to guess which
    # model version produced it or where the latency went.
    fraud_indicators: FraudIndicators = Field(default_factory=FraudIndicators)
    processing: ProcessingInfo | None = None
    models: ModelInfo | None = None
    pipeline_version: str = Field(
        default=PIPELINE_VERSION,
        description="Bumped whenever scoring changes; pin decisions to it when auditing",
    )


class DecisionResponse(BaseModel):
    """Fused decision from three pre-computed layer scores."""

    status: DecisionStatus
    final_score: float = Field(..., ge=0, le=100)
    confidence: ConfidenceLevel
    capture_score: float = Field(..., ge=0, le=100)
    identity_score: float = Field(..., ge=0, le=100)
    liveness_score: float = Field(..., ge=0, le=100)


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    models_ready: bool
    details: dict[str, bool] = Field(default_factory=dict)
