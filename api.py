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

import base64
import os
import tempfile
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kyc_pipeline import DocumentPipeline
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

# One pipeline instance for the process lifetime — the OCR models load once.
_pipeline = DocumentPipeline()


@app.on_event("startup")
def _warmup() -> None:
    _pipeline._ensure_ocr()


@app.get("/healthz")
def healthz():
    return {"ok": _pipeline.ocr_available}


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
