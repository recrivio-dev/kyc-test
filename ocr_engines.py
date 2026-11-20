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
import copy
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
    # Angle-classifier verdict for this line ("0" / "180") and its score.
    # Only populated by RapidOCREngine.read().
    cls_label: str = "0"
    cls_score: float = 0.0


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

    # RapidOCR drops any line whose recognition score is below this gate.
    TEXT_SCORE = 0.5

    def __init__(self, lang: str = "en", threads: int = 0,
                 det_max_side: int = 0, rec_model_path: str = "",
                 rec_keys_path: str = ""):
        """``threads`` — ONNX Runtime intra-op threads per session (0 = ORT
        default, i.e. one per physical core). ``det_max_side`` — cap the
        longest side the *detector* sees (0 = RapidOCR default). Recognition
        always crops from the full-resolution image, so lowering the detector
        resolution speeds up the dominant cost without blurring the glyphs
        the recogniser reads."""
        from rapidocr_onnxruntime import RapidOCR

        self.name = "rapidocr"
        self.lang = lang
        kwargs = {}
        if threads > 0:
            kwargs["intra_op_num_threads"] = threads
            kwargs["inter_op_num_threads"] = 1
        # Swap the recognition model (e.g. an English/Latin one); keys_path is
        # only needed when the model doesn't embed its character list.
        if rec_model_path:
            kwargs["rec_model_path"] = rec_model_path
        if rec_keys_path:
            kwargs["rec_keys_path"] = rec_keys_path
        # First construction loads the ONNX models once.
        self.engine = RapidOCR(**kwargs)
        if det_max_side > 0:
            from rapidocr_onnxruntime.ch_ppocr_det.utils import DetPreProcess
            det = self.engine.text_det
            det.get_preprocess = lambda _max_wh: DetPreProcess(
                det_max_side, "max", det.mean, det.std)

    def _detect_boxes(self, img, box_thresh: float = 0.0):
        """RapidOCR's detection step, optionally with a per-call box
        threshold. The override runs on a shallow copy of the detector so
        concurrent requests sharing this engine never see each other's
        threshold."""
        det = self.engine.text_det
        if box_thresh > 0:
            det = copy.copy(det)
            det.postprocess_op = copy.copy(det.postprocess_op)
            det.postprocess_op.box_thresh = box_thresh
        dt_boxes, _ = det(img)
        if dt_boxes is None or len(dt_boxes) < 1:
            return None
        return self.engine.sorted_boxes(dt_boxes)

    def read(self, image, box_thresh: float = 0.0) -> List[OCRWordBox]:
        """One fused detect → angle-classify → recognise pass that returns
        EVERY detected line, including the ones RapidOCR's own ``__call__``
        would silently drop below ``TEXT_SCORE``, together with each line's
        angle-class verdict.

        Returning the sub-gate lines lets the caller re-read them (see
        kyc_pipeline._recover_dropped_lines) without a second detection
        pass, and the per-line angle class doubles as a free orientation
        check. Uses RapidOCR's own stage objects, so boxes, crops and
        ordering are identical to ``engine(image)``."""
        eng = self.engine
        raw_h, raw_w = image.shape[:2]
        img, ratio_h, ratio_w = eng.preprocess(image)
        op_record = {"preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w}}
        img, op_record = eng.maybe_add_letterbox(img, op_record)
        dt_boxes = self._detect_boxes(img, box_thresh)
        if dt_boxes is None:
            return []
        crops = eng.get_crop_img_list(img, dt_boxes)
        crops, cls_res, _ = eng.text_cls(crops)
        rec_res, _ = eng.text_rec(crops)
        quads = eng._get_origin_points(dt_boxes, op_record, raw_h, raw_w)
        return [
            OCRWordBox(rec[0], _poly_to_xyxy(quad), float(rec[1]),
                       str(cls[0]), float(cls[1]))
            for quad, cls, rec in zip(quads, cls_res, rec_res)
        ]

    def probe_orientation(self, image) -> List[Tuple[float, float, str, float]]:
        """Detection + angle-class only (no recognition) — the cheap
        orientation probe. Returns ``(width, height, cls_label, cls_score)``
        per detected line, where width/height are the line quad's own side
        lengths *before* RapidOCR stands tall crops upright (it rotates any
        crop with h/w >= 1.5 by 90° counter-clockwise before classifying)."""
        eng = self.engine
        img, _, _ = eng.preprocess(image)
        img, _ = eng.maybe_add_letterbox(img, {})
        dt_boxes, _ = eng.auto_text_det(img)
        if dt_boxes is None:
            return []
        crops = eng.get_crop_img_list(img, dt_boxes)
        _, cls_res, _ = eng.text_cls(crops)
        out = []
        for box, cls in zip(dt_boxes, cls_res):
            w = float(max(np.linalg.norm(box[0] - box[1]),
                          np.linalg.norm(box[2] - box[3])))
            h = float(max(np.linalg.norm(box[0] - box[3]),
                          np.linalg.norm(box[1] - box[2])))
            out.append((w, h, str(cls[0]), float(cls[1])))
        return out

    def recognize(self, crops) -> List[Tuple[str, float]]:
        """Angle-classify + recognise already-cropped single lines in one
        batched call (detection OFF). Returns ``(text, score)`` per crop."""
        if not crops:
            return []
        crops, _, _ = self.engine.text_cls(list(crops))
        rec_res, _ = self.engine.text_rec(crops)
        return [(r[0], float(r[1])) for r in rec_res]

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
