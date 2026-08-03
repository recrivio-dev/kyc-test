"""FastAPI service that exposes the KYC OCR pipeline to a frontend.

Run locally:
    uvicorn api:app --reload --port 8000

Generic endpoint (``doc_type`` chosen by the caller):

    curl -F "file=@sample/pan-test.png" -F "doc_type=PAN" \\
         http://127.0.0.1:8000/api/v1/ocr

Per-document-type endpoints (no ``doc_type`` form field needed):

    POST /api/v1/ocr/pan
    POST /api/v1/ocr/aadhaar
    POST /api/v1/ocr/passport
    POST /api/v1/ocr/voter-id
    POST /api/v1/ocr/driving-license

The response body is exactly the contract documented in
``output_schema.py`` — i.e. the same JSON that the Streamlit UI displays.

Face liveness + ID-photo face-match ship in the same process under
``/api/v1/liveness/...`` (see ``liveness/router.py``). They deliberately return
their own schemas rather than the OCR ``success_envelope``; the two subsystems
share a container, not a response contract.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kyc_pipeline import DocumentPipeline
from liveness.router import models_on_disk
from liveness.router import router as liveness_router
from output_schema import failure_envelope, success_envelope

DocType = Literal["PAN", "AADHAAR", "PASSPORT", "VOTER_ID", "DRIVING_LICENSE"]

# The /api/v1/ocr/mask-identity contract uses lower-case, hyphenated document types.
# Underscores are also accepted so the OCR-style VOTER_ID / DRIVING_LICENSE
# forms work too.
_MASK_DOC_MAP = {
    "aadhaar": "AADHAAR",
    "pan": "PAN",
    "voter-id": "VOTER_ID",
    "passport": "PASSPORT",
    "driving-license": "DRIVING_LICENSE",
}

app = FastAPI(title="Recrivio KYC OCR", version="1.0.0")

# Permissive CORS for the frontend — tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(liveness_router)

# One pipeline instance for the process lifetime — the OCR models load once.
_pipeline = DocumentPipeline()

# Deep-readiness state. `/healthz` returns 200 only when the models are loaded
# AND a startup self-test drove a document through the ENTIRE pipeline without
# raising. The old check reported healthy on "models loaded" alone, so a build
# that boots but 500s on every request (the July-2026 estimate_skew_angle
# regression) went green and silently served errors. Now such a build never
# becomes healthy — the container is caught at deploy time, not by users.
_selftest_ok = False
_selftest_error: "str | None" = None

# Liveness readiness is tracked and reported SEPARATELY from the OCR self-test.
# The two subsystems share a process but not a failure mode: an OCR-only deploy
# must not go unhealthy merely because the ~330 MB of liveness weights are still
# downloading onto a cold cache volume. By default `/healthz` therefore gates
# `ok` on OCR alone; set LIVENESS_REQUIRED=true on a deployment that genuinely
# serves liveness traffic and wants the probe to fail until it can.
_liveness_ready = False
_liveness_error: "str | None" = None
_LIVENESS_REQUIRED = os.getenv(
    "LIVENESS_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}


async def _provision_liveness() -> None:
    """Download the liveness weights and prove they infer, in the background.

    Runs OFF the startup path on purpose. The first cold-volume download is
    ~330 MB, which would blow past the 60 s `start_period` in the compose
    healthcheck and flap the container. Until it finishes, `/healthz` reports
    `liveness_ready: false` and the liveness endpoints load lazily anyway (each
    would just pay the download cost on first call).
    """
    global _liveness_ready, _liveness_error

    def _work() -> None:
        from liveness.utils.face_engine import detect_faces, detect_landmarks
        from liveness.utils.model_manager import ensure_all_models

        ensure_all_models()
        # Readiness is not "files exist" — it's "the graphs actually run". One
        # synthetic frame through both engines catches a corrupt download or an
        # ONNX/mediapipe API break that a stat() check would sail past.
        probe = np.full((256, 256, 3), 127, np.uint8)
        detect_landmarks(probe)
        detect_faces(probe)

    try:
        await asyncio.get_running_loop().run_in_executor(None, _work)
        _liveness_ready, _liveness_error = True, None
        logging.getLogger("uvicorn.error").info("Liveness models provisioned and verified.")
    except Exception as exc:  # any failure ⇒ liveness not ready
        _liveness_ready = False
        _liveness_error = f"{type(exc).__name__}: {exc}"
        logging.getLogger("uvicorn.error").error(
            "Liveness provisioning FAILED: %s", _liveness_error)


def _selftest_image() -> "np.ndarray":
    """A tiny synthetic PAN-like card (light background, dark text, a border) —
    enough to drive preprocess → crop → deskew (HoughLinesP) → locate → classify
    without bundling a fixture in the image. We assert only that it runs cleanly,
    not what it extracts."""
    img = np.full((260, 430, 3), 245, np.uint8)
    cv2.rectangle(img, (8, 8), (421, 251), (60, 60, 60), 2)
    for i, line in enumerate((
        "INCOME TAX DEPARTMENT",
        "Permanent Account Number Card",
        "ABCDE1234F",
        "RAHUL GUPTA",
        "01/01/1990",
    )):
        cv2.putText(img, line, (22, 46 + i * 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return img


@app.on_event("startup")
async def _warmup() -> None:
    global _selftest_ok, _selftest_error
    _pipeline._ensure_ocr()
    # Run the full OCR path once on a synthetic doc. Any exception here means the
    # deploy is broken at runtime — report unhealthy instead of serving 500s.
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, _selftest_image())
        await _pipeline.process_and_verify(path, "PAN")
        _selftest_ok, _selftest_error = True, None
    except Exception as exc:  # any failure ⇒ not ready
        _selftest_ok = False
        _selftest_error = f"{type(exc).__name__}: {exc}"
        logging.getLogger("uvicorn.error").error(
            "OCR startup self-test FAILED — reporting unhealthy: %s", _selftest_error)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    # Fire-and-forget: liveness provisions in the background so a cold cache
    # volume never delays the container becoming healthy. See _provision_liveness.
    asyncio.create_task(_provision_liveness())


@app.get("/healthz")
def healthz():
    # 200 only when models are loaded AND the pipeline self-test passed, so a
    # boots-but-crashes build never reports healthy. 503 flips the Docker
    # healthcheck (and any load balancer) to unhealthy.
    #
    # Liveness is reported alongside but does not gate `ok` unless
    # LIVENESS_REQUIRED is set — the subsystems are independently deployable and
    # liveness weights arrive after boot.
    ocr_ok = bool(_pipeline.ocr_available and _selftest_ok)
    ok = ocr_ok and (_liveness_ready or not _LIVENESS_REQUIRED)
    return JSONResponse(
        content={
            "ok": ok,
            "ocr_available": _pipeline.ocr_available,
            "selftest_ok": _selftest_ok,
            "selftest_error": _selftest_error,
            # True once the weights are on disk AND a synthetic frame ran
            # through the MediaPipe + InsightFace graphs without raising.
            "liveness_ready": _liveness_ready,
            "liveness_error": _liveness_error,
            "liveness_required": _LIVENESS_REQUIRED,
            "liveness_models": models_on_disk(),
        },
        status_code=200 if ok else 503,
    )


async def _run_pipeline(file: UploadFile, doc_type: str) -> JSONResponse:
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())

        result = await _pipeline.process_and_verify(path, doc_type.upper())
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    payload = result.get("output_json")
    if payload is None:
        payload = failure_envelope(
            result.get("message") or "processing failed")

    return JSONResponse(
        content=payload,
        status_code=payload.get("status_code", 200),
    )


@app.post("/api/v1/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    doc_type: DocType = Form(...),
):
    """Run the full locate→read→mask pipeline on an uploaded document and
    return the structured JSON contract."""
    return await _run_pipeline(file, doc_type)


@app.post("/api/v1/ocr/pan")
async def ocr_pan(file: UploadFile = File(...)):
    return await _run_pipeline(file, "PAN")


@app.post("/api/v1/ocr/aadhaar")
async def ocr_aadhaar(file: UploadFile = File(...)):
    return await _run_pipeline(file, "AADHAAR")


@app.post("/api/v1/ocr/passport")
async def ocr_passport(file: UploadFile = File(...)):
    return await _run_pipeline(file, "PASSPORT")


@app.post("/api/v1/ocr/voter-id")
async def ocr_voter_id(file: UploadFile = File(...)):
    return await _run_pipeline(file, "VOTER_ID")


@app.post("/api/v1/ocr/driving-license")
async def ocr_driving_license(file: UploadFile = File(...)):
    return await _run_pipeline(file, "DRIVING_LICENSE")


@app.post("/api/v1/ocr/mask-identity")
async def mask_identity_endpoint(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    client_id: Optional[str] = Form(None),
):
    """Return two cropped + rotated (deskewed, upright) renders of the
    document: ``unmasked_image`` (full ID visible — for authorised/internal
    use) and ``masked_image`` (ID number redacted, last 4 kept — safe to
    display). Both are base64 PNGs in the same coordinate space, alongside the
    ``masked_regions`` geometry for client-side overlay.

    ``masked_image`` fails closed: when no ID number can be confidently
    located it is ``null`` (the ``unmasked_image`` is still returned). A 4xx /
    5xx status is only raised when the document itself could not be processed
    (bad type, OCR unavailable, encode failure)."""
    key = (document_type or "").strip().lower().replace("_", "-")
    internal = _MASK_DOC_MAP.get(key)
    if internal is None:
        return JSONResponse(
            content=failure_envelope(
                f"Unsupported document_type: {document_type}", status=400),
            status_code=400,
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
        result = await _pipeline.mask_identity(path, internal)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    # Document couldn't be processed at all ⇒ failure (bad type / OCR down /
    # encode error). 4xx = permanent (don't retry), 5xx = transient.
    if result.get("unmasked_image") is None:
        status = result.get("status", 422)
        return JSONResponse(
            content=failure_envelope(
                result.get("message") or "document processing failed",
                status=status),
            status_code=status,
        )

    masked = result.get("masked_image")
    data = {
        "unmasked_image": base64.b64encode(
            result["unmasked_image"]).decode("ascii"),
        # null when the ID could not be located (fails closed); the message
        # then explains why nothing was redacted.
        "masked_image": (base64.b64encode(masked).decode("ascii")
                         if masked is not None else None),
        "mime_type": "image/png",
        "document_detected": result["document_detected"],
        "masked_regions": result["masked_regions"],
        "message": result.get("message"),
        "client_id": client_id,
    }
    return JSONResponse(content=success_envelope(data), status_code=200)
