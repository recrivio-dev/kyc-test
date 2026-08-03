"""Request schemas for the verification API.

Most endpoints accept multipart uploads (images / video) so the heavy binary
payloads are not base64-inflated. These models capture the structured side
channel (frame timestamps, camera metadata, challenge selection).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChallengeType(str, Enum):
    """Supported active-liveness challenges.

    The first four are head movements scored from landmark yaw/pitch. ``BLINK``
    is scored from the eye-blink blendshape signal instead, but is expressed as
    a challenge so it can take an ordered position in a sequence — "blink after
    you turn" is a meaningfully stronger proof than "blink at some point".
    """

    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    BLINK = "blink"


class ChallengeSequenceMixin(BaseModel):
    """Accepts either one challenge or an ordered sequence of them.

    ``challenge`` (singular) predates the sequence support and is still honoured,
    so existing callers keep working untouched. When both are supplied
    ``challenges`` wins — it is the more specific instruction.
    """

    challenge: ChallengeType = Field(
        default=ChallengeType.TURN_LEFT,
        description="Single challenge. Ignored when `challenges` is supplied.",
    )
    challenges: list[ChallengeType] | None = Field(
        default=None,
        max_length=6,
        description=(
            "Ordered challenges the user was asked to perform, in the order they "
            "were prompted. Scored over ONE continuous recording — send a single "
            "take, not one clip per challenge, or the ordering check is meaningless."
        ),
    )
    mirrored: bool | None = Field(
        default=None,
        description=(
            "Whether the frames are horizontally mirrored, as selfie previews "
            "usually are. SEND THIS when the sequence contains both turn_left and "
            "turn_right: a mirrored left-then-right is an identical signal to an "
            "unmirrored right-then-left, so left/right ordering cannot be verified "
            "while it is unknown, and the response is flagged "
            "CHALLENGE_SEQUENCE_ORDER_PARTIAL. The browser knows this — it is "
            "true whenever the video element carries a `scaleX(-1)` transform "
            "AND the frames were captured from that transformed element."
        ),
    )

    def resolved_challenges(self) -> list[ChallengeType]:
        """The challenge list to score, however the caller expressed it."""
        if self.challenges:
            return self.challenges
        return [self.challenge]


class CameraMetadata(BaseModel):
    """Client-reported camera metadata for Layer 1 validation."""

    device_name: str | None = Field(default=None, description="Human-readable device label")
    device_id: str | None = Field(default=None, description="Unique device identifier")
    driver: str | None = Field(default=None, description="Driver / backend name")
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, ge=0)


class CaptureCheckRequest(BaseModel):
    """Structured payload accompanying a capture-integrity check.

    ``frame_timestamps_ms`` are client-side capture timestamps (milliseconds)
    used for frame-timing / jitter analysis. The frames themselves are sent as
    multipart files alongside this JSON blob.
    """

    frame_timestamps_ms: list[float] = Field(
        default_factory=list,
        description="Per-frame capture timestamps in milliseconds",
    )
    metadata: CameraMetadata = Field(default_factory=CameraMetadata)


class LivenessCheckRequest(ChallengeSequenceMixin):
    """Structured payload accompanying a liveness check."""

    fps: float = Field(default=15.0, gt=0, description="Capture frame rate")
    frame_timestamps_ms: list[float] = Field(default_factory=list)


class VerifyRequest(ChallengeSequenceMixin):
    """Structured payload for the full pipeline endpoint."""

    fps: float = Field(default=15.0, gt=0)
    frame_timestamps_ms: list[float] = Field(default_factory=list)
    metadata: CameraMetadata = Field(default_factory=CameraMetadata)


class ScoreRequest(BaseModel):
    """Fuse three already-computed layer scores into a final decision.

    Used by the Final Report, which aggregates results produced on the
    individual layer pages instead of re-running the whole pipeline.
    """

    capture_score: float = Field(..., ge=0, le=100)
    identity_score: float = Field(..., ge=0, le=100)
    liveness_score: float = Field(..., ge=0, le=100)
    identity_match: bool = Field(default=True)
    quality_index: float = Field(default=100.0, ge=0, le=100)
