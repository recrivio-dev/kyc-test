"""Lightweight 'Locate First' layout detector.

The fast pipeline never OCRs a full page. Instead it first finds *where*
text lives, then OCRs only those small crops concurrently.

Two interchangeable backends:

  * ``rapidocr_det`` (default) — reuse the ONNX text-detection model that
    already ships with RapidOCR. Zero extra model files, returns tight
    text-line regions. This is the recommended generic backend.

  * ``yolo`` — a generic document-layout YOLOv8 ONNX model (e.g.
    DocLayout-YOLO / PP-DocLayout exported to ONNX). Set the path in
    ``config.LayoutSettings.yolo_model_path``. If the file is missing the
    detector transparently degrades to ``rapidocr_det``.

NOTE on field semantics:
    A *generic* layout model returns geometric regions (text lines,
    titles, figures) — not semantic fields like "DOB" or "Sensitive ID".
    Geometry alone cannot say which line holds the Aadhaar number, so the
    pipeline tags regions semantically afterwards by regex over the OCR'd
    crop text (see kyc_pipeline._sensitive_boxes). Masking still becomes
    independent of Tesseract: the *box* comes from this detector, only the
    *which-box* decision uses recognised text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Region:
    bbox: Tuple[int, int, int, int]        # (x1, y1, x2, y2)
    label: str = "text"
    score: float = 1.0

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def is_landscape(self) -> bool:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) >= (y2 - y1)


# ──────────────────────────────────────────────────────────────────────────────
# Generic YOLOv8 ONNX backend
# ──────────────────────────────────────────────────────────────────────────────

class _YoloLayout:
    """Minimal YOLOv8-style ONNX inference (no ultralytics dependency)."""

    def __init__(self, model_path: str, classes, score_threshold: float):
        import os
        import onnxruntime as ort

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        _, _, h, w = self.session.get_inputs()[0].shape
        self.in_h = int(h) if isinstance(h, int) else 640
        self.in_w = int(w) if isinstance(w, int) else 640
        self.classes = list(classes)
        self.score_threshold = score_threshold

    def _letterbox(self, img):
        h, w = img.shape[:2]
        scale = min(self.in_w / w, self.in_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        dx, dy = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, scale, dx, dy

    def detect(self, image) -> List[Region]:
        canvas, scale, dx, dy = self._letterbox(image)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        out = self.session.run(None, {self.input_name: blob})[0]
        # YOLOv8 head: (1, 4 + nc, N) → (N, 4 + nc)
        preds = np.squeeze(out, 0).T
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        cls_ids = class_scores.argmax(axis=1)
        confs = class_scores.max(axis=1)

        keep = confs >= self.score_threshold
        boxes_xywh, confs, cls_ids = boxes_xywh[keep], confs[keep], cls_ids[keep]
        if len(boxes_xywh) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2, then undo letterbox
        xy = np.empty_like(boxes_xywh)
        xy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        xy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        xy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        xy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        xy[:, [0, 2]] = (xy[:, [0, 2]] - dx) / scale
        xy[:, [1, 3]] = (xy[:, [1, 3]] - dy) / scale

        idxs = cv2.dnn.NMSBoxes(
            [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
             for x1, y1, x2, y2 in xy],
            confs.tolist(), self.score_threshold, 0.45,
        )
        regions: List[Region] = []
        for i in np.array(idxs).flatten():
            x1, y1, x2, y2 = xy[i]
            cid = int(cls_ids[i])
            label = self.classes[cid] if cid < len(self.classes) else str(cid)
            regions.append(Region(
                (int(x1), int(y1), int(x2), int(y2)), label, float(confs[i])
            ))
        return regions


# ──────────────────────────────────────────────────────────────────────────────
# Public detector
# ──────────────────────────────────────────────────────────────────────────────

class LayoutDetector:

    def __init__(self, settings, det_engine=None):
        """``det_engine`` is a RapidOCREngine, reused for the detection-only
        backend so we never load a second model."""
        self.settings = settings
        self.det_engine = det_engine
        self.backend = settings.backend
        self._yolo = None

        if self.backend == "yolo":
            try:
                self._yolo = _YoloLayout(
                    settings.yolo_model_path,
                    settings.yolo_classes,
                    settings.score_threshold,
                )
            except Exception as e:                   # noqa: BLE001
                print(f"[WARN] YOLO layout model unavailable ({e}); "
                      f"falling back to 'rapidocr_det'.")
                self.backend = "rapidocr_det"

    def detect(self, image) -> List[Region]:
        """Return text-bearing regions, sorted top-to-bottom."""
        if self.backend == "yolo" and self._yolo is not None:
            regions = self._yolo.detect(image)
        else:
            regions = self._detect_via_rapidocr(image)

        regions = [r for r in regions if r.area > 0]
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def _detect_via_rapidocr(self, image) -> List[Region]:
        if self.det_engine is None:
            raise RuntimeError("rapidocr_det backend needs a det_engine")
        return [
            Region(box, label="text_line", score=1.0)
            for box in self.det_engine.detect(image)
        ]
