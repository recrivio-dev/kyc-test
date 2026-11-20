"""/api/v1/ocr/pdf-pages — full-page OCR of selected PDF pages.

Fixtures are generated on the fly (synthetic text rendered into an image
and saved as a PDF page), so no real document is committed.
"""
from __future__ import annotations

import io
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_pages import MAX_PAGES_PER_CALL, PdfPagesError, parse_page_list  # noqa: E402


def _page(lines):
    """A white A4-ish page (as an image, like a scanned report page) with
    large dark text lines."""
    img = np.full((1400, 1000, 3), 255, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (60, 150 + i * 110), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (20, 20, 20), 3, cv2.LINE_AA)
    return Image.fromarray(img[:, :, ::-1])


def _pdf(pages) -> bytes:
    images = [_page(lines) for lines in pages]
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True,
                   append_images=images[1:], resolution=150)
    return buf.getvalue()


# ── page-list parsing (no OCR) ───────────────────────────────────────────────

def test_parse_page_list_keeps_order_and_dedupes():
    assert parse_page_list(" 4, 5 ,4") == [4, 5]


@pytest.mark.parametrize("raw", ["", "0", "-1", "a", "4,x"])
def test_parse_page_list_rejects_bad_input(raw):
    with pytest.raises(PdfPagesError):
        parse_page_list(raw)


def test_parse_page_list_caps_pages_per_call():
    too_many = ",".join(str(i) for i in range(1, MAX_PAGES_PER_CALL + 2))
    with pytest.raises(PdfPagesError):
        parse_page_list(too_many)


# ── endpoint (loads the OCR engine once for the module) ──────────────────────

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api import app
    with TestClient(app) as c:  # runs the lifespan warm-up / self-test
        yield c


def _post(client, pdf: bytes, pages: str, filename="report.pdf"):
    return client.post(
        "/api/v1/ocr/pdf-pages",
        files={"file": (filename, pdf, "application/pdf")},
        data={"pages": pages},
    )


def test_reads_only_the_requested_page(client):
    pdf = _pdf([["FIRST PAGE"], ["SCORE 802", "TOTAL 45"]])
    res = _post(client, pdf, "2")
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["page_count"] == 2
    [page] = data["pages"]
    assert page["page"] == 2
    text = " ".join(line["text"] for line in page["lines"]).upper()
    assert "802" in text and "45" in text
    assert "FIRST" not in text
    for line in page["lines"]:
        assert 0.0 <= line["confidence"] <= 1.0
        x1, y1, x2, y2 = line["bbox"]
        assert x2 > x1 and y2 > y1


def test_out_of_range_page_is_a_400(client):
    res = _post(client, _pdf([["ONLY PAGE"]]), "3")
    assert res.status_code == 400
    assert res.json()["success"] is False


def test_non_pdf_is_a_415(client):
    res = _post(client, b"not a pdf", "1", filename="photo.jpg")
    assert res.status_code == 415


def test_corrupt_pdf_is_a_422(client):
    res = _post(client, b"%PDF-1.4 garbage", "1")
    assert res.status_code == 422


def test_existing_routes_still_healthy(client):
    # The new route must not disturb the KYC pipeline's self-test.
    assert client.get("/healthz").status_code == 200
