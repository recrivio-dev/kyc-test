"""FastAPI service that exposes the KYC OCR pipeline to a frontend.

Run locally:
    uvicorn api:app --reload --port 8000

The single business endpoint is ``POST /api/v1/ocr``:

    curl -F "file=@sample/pan-test.png" -F "doc_type=PAN" \\
         http://127.0.0.1:8000/api/v1/ocr

The response body is exactly the contract documented in
``output_schema.py`` — i.e. the same JSON that the Streamlit UI displays.
"""
from __future__ import annotations

import os
import tempfile
from typing import Literal

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kyc_pipeline import DocumentPipeline
from output_schema import failure_envelope

DocType = Literal["PAN", "AADHAAR", "PASSPORT", "VOTER_ID", "DRIVING_LICENSE"]

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


@app.post("/api/v1/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    doc_type: DocType = Form(...),
):
    """Run the full locate→read→mask pipeline on an uploaded document and
    return the structured JSON contract."""
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
