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
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import tempfile
from typing import AsyncIterator, Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import _env_int
from kyc_pipeline import DocumentLoadError, DocumentPipeline
from output_schema import failure_envelope, success_envelope

log = logging.getLogger("uvicorn.error")

# Uploads larger than this are rejected with 413 before any decoding.
MAX_UPLOAD_BYTES = _env_int("KYC_MAX_UPLOAD_MB", 25) * 1024 * 1024

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

@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await _warmup()
    yield


app = FastAPI(title="Recrivio KYC OCR", version="1.0.0", lifespan=_lifespan)

# Permissive CORS for the frontend — tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# One pipeline instance for the process lifetime — the OCR models load once.
_pipeline = DocumentPipeline()

# Masking runs in the background for the caller, but it is as CPU-heavy as an
# extraction and shares the same cores. Cap how many run at once per worker so
# a burst of mask jobs can't queue user-facing extractions behind them.
_mask_slots = asyncio.Semaphore(max(1, _env_int("KYC_MASK_CONCURRENCY", 1)))

# Deep-readiness state. `/healthz` returns 200 only when the models are loaded
# AND a startup self-test drove a document through the ENTIRE pipeline without
# raising. The old check reported healthy on "models loaded" alone, so a build
# that boots but 500s on every request (the July-2026 estimate_skew_angle
# regression) went green and silently served errors. Now such a build never
# becomes healthy — the container is caught at deploy time, not by users.
_selftest_ok = False
_selftest_error: "str | None" = None


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
        log.error("OCR startup self-test FAILED — reporting unhealthy: %s",
                  _selftest_error)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/healthz")
def healthz():
    # 200 only when models are loaded AND the pipeline self-test passed, so a
    # boots-but-crashes build never reports healthy. 503 flips the Docker
    # healthcheck (and any load balancer) to unhealthy.
    ok = bool(_pipeline.ocr_available and _selftest_ok)
    return JSONResponse(
        content={
            "ok": ok,
            "ocr_available": _pipeline.ocr_available,
            "selftest_ok": _selftest_ok,
            "selftest_error": _selftest_error,
        },
        status_code=200 if ok else 503,
    )


class _UploadError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


@contextlib.asynccontextmanager
async def _upload_to_tempfile(file: UploadFile) -> AsyncIterator[str]:
    """Spool the upload to a temp file (the pipeline reads by path so cv2 can
    honour EXIF orientation / pdfium can open PDFs) and always delete it."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise _UploadError("Empty file", 400)
    if len(data) > MAX_UPLOAD_BYTES:
        raise _UploadError(
            f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB", 413)
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse(content=failure_envelope(message, status=status),
                        status_code=status)


async def _guarded(coro_fn, file: UploadFile, *args):
    """Run a pipeline entry point on the uploaded file, mapping bad input to
    4xx and anything unexpected to a logged 500 in the standard envelope
    (instead of FastAPI's bare-text 500)."""
    try:
        async with _upload_to_tempfile(file) as path:
            return await coro_fn(path, *args)
    except _UploadError as e:
        return _error(str(e), e.status)
    except DocumentLoadError as e:        # undecodable image / PDF
        return _error(str(e), 422)
    except Exception:
        log.exception("pipeline failed")
        return _error("Internal error while processing the document", 500)


async def _run_pipeline(file: UploadFile, doc_type: str) -> JSONResponse:
    result = await _guarded(_pipeline.process_and_verify, file,
                            doc_type.upper())
    if isinstance(result, JSONResponse):
        return result

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

    async with _mask_slots:
        result = await _guarded(_pipeline.mask_identity, file, internal)
    if isinstance(result, JSONResponse):
        return result

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
