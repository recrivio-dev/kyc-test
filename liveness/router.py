"""HTTP surface for the liveness / face-match layers.

All endpoints hang off one ``APIRouter`` mounted by ``api.py`` under the same
``/api/v1`` prefix the OCR endpoints use, so liveness ships inside the OCR
container rather than as a second service.

Two transports, one implementation
----------------------------------
The *multipart* endpoints are the straight port of live-mini's routes. The two
*JSON* endpoints (``/frames``, ``/verify-json``) are new: the real caller is
``recriauth``, which pulls frames off a LiveKit video track server-side, and
posting a JSON array of base64 JPEGs from Node is far simpler than assembling a
multipart body with N file parts. Both transports are thin adapters over the
same ``verify_liveness()`` / ``run_full_pipeline()`` service calls and share the
same decoder, so a malformed frame produces the same 400 either way.

Response contract — deliberate divergence from OCR
--------------------------------------------------
These endpoints return the pydantic models in ``liveness.schemas.responses``
directly. They are NOT wrapped in ``output_schema.success_envelope`` /
``failure_envelope``: that envelope is the OCR contract, whereas the liveness
contract is the explainable score breakdown (per-stage sub-scores + ``reasons``)
already described by its own schemas. Wrapping one in the other would make both
harder to consume, so the divergence is intentional.
"""
from __future__ import annotations

import base64
import binascii
import json

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import Field

from liveness.config import settings
from liveness.schemas.requests import (
    CameraMetadata,
    CaptureCheckRequest,
    ChallengeSequenceMixin,
    LivenessCheckRequest,
    ScoreRequest,
    VerifyRequest,
)
from liveness.schemas.responses import (
    CaptureResult,
    DecisionResponse,
    HealthResponse,
    IdentityResult,
    LivenessResult,
    VerificationResponse,
)
from liveness.services.layer1_capture import analyze_capture
from liveness.services.layer2_identity import verify_identity
from liveness.services.layer3_liveness import verify_liveness
from liveness.services.pipeline import run_full_pipeline
from liveness.services.scoring import score_pipeline
from liveness.utils.image import decode_image_bytes, decode_video_bytes
from liveness.utils.logger import logger
from liveness.utils.quality import confidence_penalty

router = APIRouter(prefix="/api/v1/liveness", tags=["liveness"])


# --------------------------------------------------------------------------- #
# Upload helpers — inlined from live-mini's routes/deps.py (it was a module of
# shared route helpers; here there is a single router, so they live with it).
# --------------------------------------------------------------------------- #
async def read_frames(
    video: UploadFile | None,
    frames: list[UploadFile] | None,
    *,
    max_frames: int = 90,
) -> list[np.ndarray]:
    """Read frames from either a single video upload or many image uploads."""
    out: list[np.ndarray] = []
    if video is not None:
        data = await video.read()
        try:
            out = decode_video_bytes(data, max_frames=max_frames)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid video: {exc}") from exc
    elif frames:
        for f in frames:
            data = await f.read()
            try:
                out.append(decode_image_bytes(data))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid frame '{f.filename}': {exc}") from exc
    if not out:
        raise HTTPException(
            status_code=400,
            detail="Provide either a 'video' file or one or more 'frames' image files.",
        )
    logger.debug("Parsed {} frames from upload.", len(out))
    return out


async def read_image(file: UploadFile, label: str = "image") -> np.ndarray:
    data = await file.read()
    try:
        return decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label}: {exc}") from exc


def parse_payload(payload: str | None, model):
    """Parse an optional JSON form field into a pydantic model (defaults if None)."""
    if not payload:
        return model()
    try:
        return model(**json.loads(payload))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid payload JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# JSON transport — base64 frame bursts
# --------------------------------------------------------------------------- #
class FramesJSONRequest(ChallengeSequenceMixin):
    """Base64 frame burst for the JSON liveness endpoint."""

    frames: list[str] = Field(
        ...,
        description="Base64-encoded JPEG/PNG frames, optionally data-URI prefixed",
    )
    fps: float = Field(default=15.0, gt=0)
    frame_timestamps_ms: list[float] = Field(default_factory=list)


class VerifyJSONRequest(FramesJSONRequest):
    """Base64 reference photo + frame burst for the JSON full-pipeline endpoint."""

    reference: str = Field(..., description="Base64-encoded reference ID photo")
    metadata: CameraMetadata = Field(default_factory=CameraMetadata)


def _b64_bytes(value: str, label: str) -> bytes:
    """Decode one base64 string, tolerating a ``data:image/...;base64,`` prefix."""
    s = value.strip()
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label}: not valid base64 ({exc})") from exc


def _decode_b64_frames(frames: list[str]) -> list[np.ndarray]:
    """Decode a base64 frame burst, enforcing the same ceiling as ``read_frames``.

    Rejects an empty list and anything past ``max_json_frames`` with 400, and a
    body past ``max_json_bytes`` with 413 rather than letting the worker OOM.
    Decoding goes through ``decode_image_bytes`` — the same function the
    multipart path uses — so a malformed frame yields the same message.
    """
    if not frames:
        raise HTTPException(status_code=400, detail="Provide at least one base64 frame in 'frames'.")
    if len(frames) > settings.max_json_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Too many frames: {len(frames)} (max {settings.max_json_frames}).",
        )

    total = 0
    out: list[np.ndarray] = []
    for i, raw in enumerate(frames):
        data = _b64_bytes(raw, f"frame[{i}]")
        total += len(data)
        if total > settings.max_json_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Decoded frame payload exceeds {settings.max_json_bytes} bytes.",
            )
        try:
            out.append(decode_image_bytes(data))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid frame[{i}]: {exc}") from exc
    logger.debug("Parsed {} frames from JSON body ({} bytes).", len(out), total)
    return out


def _decode_b64_image(value: str, label: str) -> np.ndarray:
    data = _b64_bytes(value, label)
    if len(data) > settings.max_json_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Decoded {label} exceeds {settings.max_json_bytes} bytes.",
        )
    try:
        return decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Layer 1 — capture integrity
# --------------------------------------------------------------------------- #
@router.post("/capture-check", response_model=CaptureResult)
async def capture_check(
    video: UploadFile | None = File(default=None),
    frames: list[UploadFile] | None = File(default=None),
    payload: str | None = Form(default=None),
) -> CaptureResult:
    """Run Layer 1 capture-integrity analysis.

    Send frames as a ``video`` file or multiple ``frames`` image files, and an
    optional ``payload`` JSON form field matching :class:`CaptureCheckRequest`
    (frame timestamps + camera metadata).
    """
    req = parse_payload(payload, CaptureCheckRequest)
    parsed = await read_frames(video, frames)
    return analyze_capture(parsed, req.frame_timestamps_ms, req.metadata)


# --------------------------------------------------------------------------- #
# Layer 2 — identity
# --------------------------------------------------------------------------- #
@router.post("/identity", response_model=IdentityResult)
async def identity_check(
    reference: UploadFile = File(..., description="Reference ID photo"),
    probe: UploadFile = File(..., description="Live capture of the user"),
) -> IdentityResult:
    """Run Layer 2 identity verification between a reference photo and a probe."""
    ref_img = await read_image(reference, "reference image")
    probe_img = await read_image(probe, "probe image")
    return verify_identity(ref_img, probe_img)


# --------------------------------------------------------------------------- #
# Layer 3 — active liveness
# --------------------------------------------------------------------------- #
@router.post("/check", response_model=LivenessResult)
async def liveness_check(
    video: UploadFile | None = File(default=None),
    frames: list[UploadFile] | None = File(default=None),
    payload: str | None = Form(default=None),
) -> LivenessResult:
    """Run Layer 3 active-liveness verification over a frame sequence.

    Send a ``video`` clip or multiple ``frames``. The ``payload`` JSON form
    field matches :class:`LivenessCheckRequest` (challenge/challenges + fps).
    """
    req = parse_payload(payload, LivenessCheckRequest)
    parsed = await read_frames(video, frames)
    return verify_liveness(
        parsed, challenges=req.resolved_challenges(), fps=req.fps, mirrored=req.mirrored,
    )


@router.post("/frames", response_model=LivenessResult)
async def liveness_frames_json(req: FramesJSONRequest) -> LivenessResult:
    """Layer 3 over a JSON burst of base64 frames.

    Same analysis as ``/check`` — this is the transport a server-side caller
    (``recriauth`` pulling frames off a LiveKit track) can produce most cheaply.
    """
    parsed = _decode_b64_frames(req.frames)
    return verify_liveness(
        parsed, challenges=req.resolved_challenges(), fps=req.fps, mirrored=req.mirrored,
    )


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
@router.post("/verify", response_model=VerificationResponse)
async def verify(
    reference: UploadFile = File(..., description="Reference ID photo"),
    video: UploadFile | None = File(default=None),
    frames: list[UploadFile] | None = File(default=None),
    payload: str | None = Form(default=None),
) -> VerificationResponse:
    """Run the complete verification pipeline.

    Requires a ``reference`` ID photo plus a live capture (``video`` clip or
    multiple ``frames``). The ``payload`` JSON form field matches
    :class:`VerifyRequest` (challenge/challenges, fps, frame timestamps, camera
    metadata).
    """
    req = parse_payload(payload, VerifyRequest)
    ref_img = await read_image(reference, "reference image")
    parsed = await read_frames(video, frames)
    return run_full_pipeline(
        reference_img=ref_img,
        frames=parsed,
        timestamps_ms=req.frame_timestamps_ms,
        metadata=req.metadata,
        challenge=req.challenge,
        challenges=req.resolved_challenges(),
        mirrored=req.mirrored,
        fps=req.fps,
    )


@router.post("/verify-json", response_model=VerificationResponse)
async def verify_json(req: VerifyJSONRequest) -> VerificationResponse:
    """Full pipeline over a JSON body — base64 reference + base64 frame burst."""
    ref_img = _decode_b64_image(req.reference, "reference image")
    parsed = _decode_b64_frames(req.frames)
    return run_full_pipeline(
        reference_img=ref_img,
        frames=parsed,
        timestamps_ms=req.frame_timestamps_ms,
        metadata=req.metadata,
        challenge=req.challenge,
        challenges=req.resolved_challenges(),
        mirrored=req.mirrored,
        fps=req.fps,
    )


# --------------------------------------------------------------------------- #
# Decision fusion
# --------------------------------------------------------------------------- #
@router.post("/score", response_model=DecisionResponse)
async def score(req: ScoreRequest) -> DecisionResponse:
    """Fuse capture/identity/liveness scores into the final risk-weighted decision."""
    penalty_pct = confidence_penalty(req.quality_index) * 100.0
    breakdown = score_pipeline(
        capture_score=req.capture_score,
        identity_score=req.identity_score,
        liveness_score=req.liveness_score,
        quality_penalty_pct=penalty_pct,
        identity_match=req.identity_match,
    )
    return DecisionResponse(
        status=breakdown.status,
        final_score=breakdown.final_score,
        confidence=breakdown.confidence,
        capture_score=breakdown.capture_score,
        identity_score=breakdown.identity_score,
        liveness_score=breakdown.liveness_score,
    )


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def models_on_disk() -> dict[str, bool]:
    """Which model artefacts are already present on the cache volume.

    Pure filesystem check — no model is loaded, so this is safe to call from
    ``/healthz`` on every probe. Shared with ``api.py``'s readiness reporting.
    """
    models_path = settings.models_path
    return {
        "face_landmarker": (models_path / "face_landmarker.task").exists(),
        "blaze_face": (models_path / "blaze_face_short_range.tflite").exists(),
        "insightface": any((models_path / "insightface").rglob("*.onnx")),
    }


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report service status and whether models are present on disk."""
    details = models_on_disk()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        models_ready=all(details.values()),
        details=details,
    )


__all__ = ["router", "models_on_disk"]
