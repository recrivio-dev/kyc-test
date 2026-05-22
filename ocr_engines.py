"""OCR engines for the fast KYC pipeline.

Design:
  * RapidOCREngine — primary. PaddleOCR's PP-OCR models served through
    ONNX Runtime (the `rapidocr-onnxruntime` package). No PaddlePaddle,
    no GPU required. The ONNX session releases the GIL during inference,
    so concurrent crops dispatched via `asyncio.to_thread` truly overlap.
  * SuryaOCREngine — fallback. A heavier transformer OCR, invoked ONLY on
    individual low-confidence crops — never on a full page.

Tesseract has been removed entirely: masking no longer depends on
word-level text recognition (see layout_detector.py / kyc_pipeline.py).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OCRWordBox:
    text: str
    bbox: Tuple[int, int, int, int]        # (x1, y1, x2, y2)
    confidence: Optional[float] = None


@dataclass
class OCRResult:
    text: str
    words: List[OCRWordBox]
    avg_confidence: Optional[float]
    engine: str
    decision_reason: Optional[str] = None


def _poly_to_xyxy(poly) -> Tuple[int, int, int, int]:
    """Collapse any polygon / quad / [x1,y1,x2,y2] into an axis-aligned box."""
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────

class OCREngine:
    name = "base"

    def extract(self, image) -> OCRResult:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Primary — RapidOCR (PaddleOCR models via ONNX Runtime)
# ──────────────────────────────────────────────────────────────────────────────

class RapidOCREngine(OCREngine):

    def __init__(self, lang: str = "en"):
        from rapidocr_onnxruntime import RapidOCR

        self.name = "rapidocr"
        self.lang = lang
        # First construction downloads/loads the ONNX models once.
        self.engine = RapidOCR()

    def extract(self, image) -> OCRResult:
        """Full detect + recognize. On a tiny crop this is effectively a
        single-line recognition and returns near-instantly."""
        result, _ = self.engine(image)

        words, texts, confs = [], [], []
        for item in (result or []):
            box, txt, score = item[0], item[1], float(item[2])
            words.append(OCRWordBox(txt, _poly_to_xyxy(box), score))
            texts.append(txt)
            confs.append(score)

        return OCRResult(
            text="\n".join(texts),
            words=words,
            avg_confidence=(sum(confs) / len(confs) if confs else None),
            engine="rapidocr",
        )

    def detect(self, image) -> List[Tuple[int, int, int, int]]:
        """Detection-only pass — locate text regions without recognising
        them. This is the cheap 'Locate First' primitive used by the
        layout detector."""
        result, _ = self.engine(
            image, use_det=True, use_cls=False, use_rec=False
        )
        return [_poly_to_xyxy(box) for box in (result or [])]

    def extract_rec_only(self, image) -> OCRResult:
        """Recognise an already-cropped single text line — detection OFF.

        The caller supplies a crop that already bounds one text line, so
        the detector is skipped entirely (`use_det=False`). This is the
        recovery path for lines the full detect+recognise pass dropped:
        RapidOCR's detector sometimes hands the recogniser a clipped quad,
        the recogniser then scores it below RapidOCR's internal
        `text_score` gate (0.5), and the whole line is silently discarded.
        Re-recognising the padded crop with detection off reads it cleanly
        (a tight DOB date is a frequent victim of this).
        """
        result, _ = self.engine(
            image, use_det=False, use_cls=True, use_rec=True
        )
        texts, confs = [], []
        for item in (result or []):
            # With detection off RapidOCR yields [text, score] pairs.
            txt, score = item[0], float(item[1])
            texts.append(txt)
            confs.append(score)
        return OCRResult(
            text=" ".join(t for t in texts if t),
            words=[],
            avg_confidence=(sum(confs) / len(confs) if confs else None),
            engine="rapidocr",
        )

    def extract_no_cls(self, image) -> OCRResult:
        """Extract WITHOUT the angle-class model.

        With `cls` enabled, RapidOCR silently rotates each detected text
        line so 0° and 180° score about the same — useless as an
        orientation probe. With `cls` disabled the recogniser reads pixels
        as-is, so upright text scores high and upside-down text scores
        very low, giving a clean 4-way orientation signal.
        """
        result, _ = self.engine(image, use_cls=False)
        words, texts, confs = [], [], []
        for item in (result or []):
            box, txt, score = item[0], item[1], float(item[2])
            words.append(OCRWordBox(txt, _poly_to_xyxy(box), score))
            texts.append(txt)
            confs.append(score)
        return OCRResult(
            text="\n".join(texts),
            words=words,
            avg_confidence=(sum(confs) / len(confs) if confs else None),
            engine="rapidocr",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Fallback — Surya (low-confidence crops only)
# ──────────────────────────────────────────────────────────────────────────────

class SuryaOCREngine(OCREngine):
    """Surya 0.17 predictor API. Models are heavy, so the predictors are
    built lazily on first use. Any failure here degrades gracefully — the
    pipeline simply keeps the primary RapidOCR result for that crop."""

    def __init__(self, lang: str = "en"):
        self.name = "surya"
        self.lang = lang
        self._rec = None
        self._det = None
        try:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            self._rec = RecognitionPredictor(FoundationPredictor())
            self._det = DetectionPredictor()
        except Exception as e:                       # noqa: BLE001
            print(f"[WARN] Surya unavailable: {e}")

    def extract(self, image) -> OCRResult:
        if self._rec is None or self._det is None:
            raise RuntimeError("Surya not available")

        from PIL import Image
        if getattr(image, "ndim", 2) == 3:
            pil = Image.fromarray(image[:, :, ::-1])  # BGR → RGB
        else:
            pil = Image.fromarray(image)

        preds = self._rec([pil], det_predictor=self._det)
        lines = getattr(preds[0], "text_lines", []) or []
        texts = [getattr(ln, "text", "") for ln in lines]
        confs = [c for c in (getattr(ln, "confidence", None) for ln in lines)
                 if c is not None]
        text = "\n".join(t for t in texts if t)

        return OCRResult(
            text=text,
            words=[],
            avg_confidence=(sum(confs) / len(confs) if confs
                            else (0.80 if text.strip() else 0.0)),
            engine="surya",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Async helper
# ──────────────────────────────────────────────────────────────────────────────

async def run_extract_async(engine: OCREngine, image) -> OCRResult:
    """Run a blocking OCR engine in a worker thread.

    onnxruntime releases the GIL during inference, so a batch of these
    awaited together with `asyncio.gather` overlaps real CPU work.
    """
    return await asyncio.to_thread(engine.extract, image)
