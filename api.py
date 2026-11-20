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

Full-page OCR of selected PDF pages (no ID-card crop/classify/mask):

    curl -F "file=@report.pdf" -F "pages=4,5" \\
         http://127.0.0.1:8000/api/v1/ocr/pdf-pages

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
from pdf_pages import PdfPagesError, parse_page_list, read_pdf_pages

log = logging.getLogger("uvicorn.error")

# Uploads larger than this are rejected with 413 before any decoding.
MAX_UPLOAD_BYTES = _env_int("KYC_MAX_UPLOAD_MB", 25) * 1024 * 1024

# Server-side deadlines. Each sits just below its CALLER's HTTP timeout, so a
# request that would have been cut by the client comes back as an explicit
# 504 `error_code: "timeout"` it can count separately, instead of a dropped
# connection that looks like an OCR failure. Never set one above its
# caller's budget's reach for normal documents: a deadline that fires on a
# request the caller would still have waited for turns a slow success into a
# guaranteed failure on every retry. 0 disables. The OCR thread itself can't
# be interrupted — it finishes in the background; only the response is cut.
#
# OCR extraction (/ocr, /ocr/<doc>): callers give it 30 s.
REQUEST_TIMEOUT_SEC = _env_int("KYC_REQUEST_TIMEOUT_SEC", 25)
# /mask-identity: recriauth gives masking 120 s — heavy masks (a full
# e-Aadhaar letter page) routinely exceed 30 s, plus time queued for a mask
# slot, which this deadline also covers. 110 s returns a clean 504 before
# the caller's 120 s cut without ever pre-empting a mask it would wait for.
MASK_TIMEOUT_SEC = _env_int("KYC_MASK_TIMEOUT_SEC", 110)
# /ocr/pdf-pages: recriauth gives full-page reads 120 s — two A4 pages plus
# time queued for the one-at-a-time pdf-pages slot, which this deadline also
# covers. Same 110 s reasoning as masking.
PDF_PAGES_TIMEOUT_SEC = _env_int("KYC_PDF_PAGES_TIMEOUT_SEC", 110)

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

# Full-page reads are several times the work of a card read (an A4 page at
# ~180 dpi, per page). Same reasoning as masking: cap them per worker so
# they can't starve user-facing KYC extractions.
_pdf_pages_slots = asyncio.Semaphore(
    max(1, _env_int("KYC_PDF_PAGES_CONCURRENCY", 1)))

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


def _error(message: str, status: int,
           error_code: str = "failed") -> JSONResponse:
    return JSONResponse(
        content=failure_envelope(message, status=status,
                                 error_code=error_code),
        status_code=status)


async def _guarded(coro_fn, file: UploadFile, *args,
                   slots: Optional[asyncio.Semaphore] = None,
                   timeout: Optional[int] = None):
    """Run a pipeline entry point on the uploaded file, mapping bad input to
    4xx, the deadline to 504 and anything unexpected to a logged 500 in the
    standard envelope (instead of FastAPI's bare-text 500). ``slots`` (a
    concurrency cap) is acquired inside the deadline, so queueing time counts
    toward it. ``timeout`` defaults to REQUEST_TIMEOUT_SEC (read at call
    time); 0 disables."""
    limit = REQUEST_TIMEOUT_SEC if timeout is None else timeout
    async def _run():
        async with _upload_to_tempfile(file) as path:
            if slots is None:
                return await coro_fn(path, *args)
            async with slots:
                return await coro_fn(path, *args)

    try:
        if limit > 0:
            return await asyncio.wait_for(_run(), limit)
        return await _run()
    except asyncio.TimeoutError:
        log.warning("request exceeded %ss deadline (%s)",
                    limit, getattr(coro_fn, "__name__", "?"))
        return _error(
            f"Processing exceeded {limit}s — server busy or "
            "document too dense; retry", 504, "timeout")
    except _UploadError as e:
        return _error(str(e), e.status, "invalid_upload")
    except DocumentLoadError as e:        # undecodable image / PDF
        return _error(str(e), 422, e.code)
    except Exception:
        log.exception("pipeline failed")
        return _error("Internal error while processing the document", 500,
                      "internal_error")


async def _run_pipeline(file: UploadFile, doc_type: str) -> JSONResponse:
    result = await _guarded(_pipeline.process_and_verify, file,
                            doc_type.upper())
    if isinstance(result, JSONResponse):
        return result

    payload = result.get("output_json")
    if payload is None:
        payload = failure_envelope(
            result.get("message") or "processing failed",
            error_code="internal_error")

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
                f"Unsupported document_type: {document_type}", status=400,
                error_code="invalid_upload"),
            status_code=400,
        )

    result = await _guarded(_pipeline.mask_identity, file, internal,
                            slots=_mask_slots, timeout=MASK_TIMEOUT_SEC)
    if isinstance(result, JSONResponse):
        return result

    # Document couldn't be processed at all ⇒ failure (bad type / OCR down /
    # encode error). 4xx = permanent (don't retry), 5xx = transient.
    if result.get("unmasked_image") is None:
        status = result.get("status", 422)
        code = {400: "invalid_upload", 503: "ocr_unavailable",
                500: "internal_error"}.get(status, "failed")
        return JSONResponse(
            content=failure_envelope(
                result.get("message") or "document processing failed",
                status=status, error_code=code),
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
        # True for an issuer-masked Aadhaar ('XXXX XXXX 1234'): nothing needed
        # redacting, masked_image is the document as uploaded (cropped).
        "already_masked": result.get("already_masked", False),
        "masked_regions": result["masked_regions"],
        "message": result.get("message"),
        "client_id": client_id,
    }
    return JSONResponse(content=success_envelope(data), status_code=200)


@app.post("/api/v1/ocr/pdf-pages")
async def pdf_pages_endpoint(
    file: UploadFile = File(...),
    pages: str = Form(...),
):
    """OCR whole PDF pages as they are — for report pages whose content is
    an embedded image (e.g. CIBIL pages inside a credit report).

    ``pages`` is a comma-separated, 1-based list (``"4,5"``), at most
    ``KYC_PDF_PAGES_MAX`` per call. Returns every recognised line with its
    confidence (0..1) and ``[x1, y1, x2, y2]`` box in the rendered page's
    pixel space. The text is not interpreted here: the caller parses it and
    decides which confidences to trust."""
    try:
        page_list = parse_page_list(pages)
    except PdfPagesError as e:
        return _error(str(e), e.status)
    if not (file.filename or "").lower().endswith(".pdf"):
        return _error("Only PDF files are supported", 415)

    _pipeline._ensure_ocr()
    if _pipeline.primary_ocr is None:
        return _error("OCR engine unavailable", 503)

    async def _read(path: str):
        # Bad PDF / out-of-range page is the caller's input error, not a 500.
        try:
            return await read_pdf_pages(path, page_list, _pipeline.primary_ocr)
        except PdfPagesError as e:
            return _error(str(e), e.status)

    result = await _guarded(_read, file, slots=_pdf_pages_slots,
                            timeout=PDF_PAGES_TIMEOUT_SEC)
    if isinstance(result, JSONResponse):
        return result
    return JSONResponse(content=success_envelope(result), status_code=200)
