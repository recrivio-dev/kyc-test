"""Full-page OCR of selected PDF pages.

The KYC pipeline (kyc_pipeline.py) is built for ID cards: it reads page 1
only, crops a card out of the frame, straightens it, classifies it and
masks the ID number. None of that fits an A4 report page whose content is
an embedded image — e.g. the TransUnion CIBIL pages inside a KonnectNXT
credit report, which carry no text layer.

This module reads a page as-is: render at a fixed DPI, one detect +
recognise pass over the whole page, and return every line with its
confidence and box. It does not interpret the text — the caller owns the
document-specific parsing and decides which confidences it trusts.

Nothing is written to disk or logged beyond page counts and timings: these
pages are credit reports.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Sequence

import numpy as np
import pypdfium2 as pdfium

from config import _env_int
from ocr_engines import RapidOCREngine

log = logging.getLogger("uvicorn.error")

# A full A4 page OCRs in ~3-5 s per worker thread; cap the pages per call so
# one request can't hold a worker for minutes.
MAX_PAGES_PER_CALL = max(1, _env_int("KYC_PDF_PAGES_MAX", 4))

# 2.5x of a 72-dpi PDF point ≈ 180 dpi: small table text on a report page
# reads cleanly, and an A4 page stays ~1490x2100 px.
RENDER_SCALE = 2.5
# Guard against a huge page size turning RENDER_SCALE into a giant bitmap.
MAX_RENDER_SIDE = 3200


class PdfPagesError(ValueError):
    """Bad input (not a PDF, bad page list). The API maps it to a 4xx."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def parse_page_list(raw: str) -> List[int]:
    """'4,5' → [4, 5]. 1-based, de-duplicated, order kept."""
    pages: List[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) < 1:
            raise PdfPagesError(f"Invalid page number: {part!r}")
        page = int(part)
        if page not in pages:
            pages.append(page)
    if not pages:
        raise PdfPagesError("No pages requested")
    if len(pages) > MAX_PAGES_PER_CALL:
        raise PdfPagesError(
            f"At most {MAX_PAGES_PER_CALL} pages per call")
    return pages


def _render(pdf: "pdfium.PdfDocument", page_number: int) -> np.ndarray:
    page = pdf[page_number - 1]
    width, height = page.get_size()
    scale = min(RENDER_SCALE, MAX_RENDER_SIDE / max(width, height, 1))
    pil = page.render(scale=scale).to_pil().convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB → BGR, as the engine expects


def _read_page(engine: RapidOCREngine, image: np.ndarray) -> List[Dict]:
    # `read` keeps the sub-gate lines RapidOCR's own call would drop, so a
    # caller sees a low-confidence read instead of a silent gap.
    return [
        {
            "text": word.text,
            "confidence": round(float(word.confidence or 0.0), 4),
            "bbox": [int(v) for v in word.bbox],
        }
        for word in engine.read(image)
        if word.text and word.text.strip()
    ]


async def read_pdf_pages(path: str, pages: Sequence[int],
                         engine: RapidOCREngine) -> Dict:
    try:
        pdf = pdfium.PdfDocument(path)
    except Exception as e:  # noqa: BLE001
        raise PdfPagesError(f"Cannot read PDF: {e}", status=422) from e
    try:
        page_count = len(pdf)
        missing = [p for p in pages if p > page_count]
        if missing:
            raise PdfPagesError(
                f"Page(s) {missing} out of range; document has "
                f"{page_count} page(s)")

        results = []
        started = time.perf_counter()
        for page_number in pages:
            image = _render(pdf, page_number)
            height, width = image.shape[:2]
            lines = await asyncio.to_thread(_read_page, engine, image)
            results.append({
                "page": page_number,
                "width": width,
                "height": height,
                "lines": lines,
            })
        log.info("pdf-pages: read %d page(s) of %d in %.1fs", len(pages),
                 page_count, time.perf_counter() - started)
        return {"page_count": page_count, "pages": results}
    finally:
        pdf.close()
