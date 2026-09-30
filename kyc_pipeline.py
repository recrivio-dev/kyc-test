"""Fast KYC document pipeline.

Old flow (slow):  4× full-page PaddleOCR  →  Surya fallback (full page)
                  →  Tesseract 4× full-page passes just for mask boxes.

New flow (fast):
  1. LOCATE   — one cheap layout-detection pass finds text regions.
  2. CROP-OCR — every region crop is OCR'd concurrently (asyncio.gather)
                with RapidOCR (PaddleOCR models on ONNX Runtime).
  3. FALLBACK — Surya re-OCRs ONLY the crops that came back low-confidence.
  4. MASK     — the 'Sensitive ID Zone' is blacked out with cv2.rectangle
                using the layout detector's box. No Tesseract anywhere.

The full page is never sent to an OCR engine, and Surya never sees more
than a single failing crop.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium

from config import SETTINGS, Settings
from layout_detector import LayoutDetector, Region
from ocr_engines import RapidOCREngine, SuryaOCREngine, run_extract_async
from output_schema import build_output_json, failure_envelope
from preprocessing import (auto_crop_document, crop_document, deskew,
                          resize_for_ocr, rotate_image)

log = logging.getLogger(__name__)


class DocumentLoadError(ValueError):
    """The upload isn't a readable image / PDF (the API answers 422)."""


class DocumentPipeline:

    DISPLAY_NAMES = {
        "AADHAAR": "Aadhaar Card",
        "PAN": "PAN Card",
        "PASSPORT": "Passport",
        "VOTER_ID": "Voter ID",
        "DRIVING_LICENSE": "Driving License",
        "UNKNOWN": "Unknown",
    }

    # The contract field name reported for each document's redacted ID
    # number in the /api/v1/ocr/mask-identity `masked_regions` payload.
    ID_FIELD = {
        "AADHAAR": "aadhaar_number",
        "PAN": "pan_number",
        "VOTER_ID": "epic_number",
        "PASSPORT": "passport_num",
        "DRIVING_LICENSE": "license_number",
    }

    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self.primary_ocr: RapidOCREngine | None = None
        self.fallback_ocr: SuryaOCREngine | None = None
        self.layout: LayoutDetector | None = None
        self.ocr_available = False

        self.patterns = {
            "PAN": r"[A-Z]{5}[0-9]{4}[A-Z]",
            "AADHAAR": r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)",
            "VOTER_ID": r"(?<![A-Z0-9])[A-Z]{3}\d{7}(?![A-Z0-9])",
            "PASSPORT": r"[A-Z]{1,2}\d{6,7}",
            # Real DL number OR Vehicle Registration Certificate number
            # (e.g. CH01CY1547). The frontend treats both under 'license'.
            # The state code is routinely printed with a hyphen or space
            # separator ("MH-1220050000188", "DL 04 20110149646"), so a
            # `[\s-]?` separator is allowed between every token group.
            "DRIVING_LICENSE":
                r"[A-Z]{2}[\s-]?\d{2}[\s-]?\d{11}"
                r"|[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,2}[\s-]?\d{3,5}",
        }

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def _ensure_ocr(self):
        """Load the primary engine + layout detector once (idempotent)."""
        if self.primary_ocr is None:
            try:
                self.primary_ocr = RapidOCREngine(
                    lang=self.settings.ocr.primary_lang,
                    threads=self.settings.ocr.threads,
                    det_max_side=self.settings.ocr.det_max_side,
                    rec_model_path=self.settings.ocr.rec_model_path,
                    rec_keys_path=self.settings.ocr.rec_keys_path)
                self.ocr_available = True
            except Exception:
                log.exception("RapidOCR init failed")
                self.ocr_available = False
        if self.layout is None and self.primary_ocr is not None:
            self.layout = LayoutDetector(
                self.settings.layout, det_engine=self.primary_ocr)

    def _ensure_fallback(self):
        """Load Surya lazily — only the first time a crop actually fails,
        and only if the fallback is enabled in config."""
        if not self.settings.ocr.enable_surya_fallback:
            return
        if self.fallback_ocr is None:
            try:
                self.fallback_ocr = SuryaOCREngine(
                    lang=self.settings.ocr.primary_lang)
            except Exception as e:                   # noqa: BLE001
                log.warning("Surya fallback unavailable: %s", e)

    def get_printable_name(self, doc: str) -> str:
        return self.DISPLAY_NAMES.get(doc, "Unknown")

    # ── Image loading ────────────────────────────────────────────────────────

    def load_document_image(self, path: str) -> np.ndarray:
        """Decode the upload. Raises DocumentLoadError for anything that isn't
        a readable image / PDF — the API maps that to a 422."""
        if path.lower().rsplit(".", 1)[-1] == "pdf":
            try:
                pdf = pdfium.PdfDocument(path)
                page = pdf[0]
                # Render straight at working resolution instead of a fixed 3x
                # (a large-format page at 3x is a multi-hundred-MB bitmap).
                pw, ph = page.get_size()
                scale = min(3.0, self.settings.work_max_side / max(pw, ph, 1))
                pil = page.render(scale=scale).to_pil()
            except Exception as e:                   # noqa: BLE001
                raise DocumentLoadError(f"Cannot read PDF: {e}") from e
            return cv2.cvtColor(np.array(pil.convert("RGB")),
                                cv2.COLOR_RGB2BGR)
        img = cv2.imread(path)
        if img is None:
            raise DocumentLoadError("Cannot decode image")
        return img

    # ── Stage helper: orientation ────────────────────────────────────────────

    # Orientation-probe decision thresholds: the dominant line direction must
    # outweigh the other by this factor, and the angle-class vote must be at
    # least this lopsided, before the cheap probe's answer is trusted.
    _ORIENT_DOMINANCE = 3.0
    _ORIENT_VOTE = 0.3
    # Only angle-class verdicts at least this sure count as votes — small or
    # blurry lines (dense e-Aadhaar letters) otherwise split the vote.
    _ORIENT_CLS_MIN = 0.9

    @classmethod
    def _line_direction(cls, lines) -> Optional[str]:
        """"wide" / "tall" when one text direction clearly dominates."""
        wide_len = sum(w for w, h, _, _ in lines if w >= 1.5 * h)
        tall_len = sum(h for w, h, _, _ in lines if h >= 1.5 * w)
        if wide_len >= cls._ORIENT_DOMINANCE * tall_len and wide_len > 0:
            return "wide"
        if tall_len >= cls._ORIENT_DOMINANCE * wide_len and tall_len > 0:
            return "tall"
        return None

    @classmethod
    def _orientation_from_lines(cls, lines) -> Optional[int]:
        """Decide the upright angle from ``(w, h, cls_label, cls_score)`` line
        stats (RapidOCREngine.probe_orientation). Returns None when the signal
        is ambiguous.

          * Line direction: horizontal text yields wide quads, sideways text
            tall ones. Tall crops are stood up by a 90° counter-clockwise turn
            before the angle classifier sees them.
          * Angle class: says whether each (stood-up) line reads upside down.

        So wide + "0" ⇒ upright; wide + "180" ⇒ 180°; tall + "0" ⇒ the page
        needs a 90° counter-clockwise turn (270); tall + "180" ⇒ 90°. Votes are
        weighted by line length so long, confidently-read lines dominate."""
        if len(lines) < 3:
            return None
        direction = cls._line_direction(lines)
        if direction == "wide":
            group = [(w, c) for w, h, c, s in lines
                     if w >= 1.5 * h and s >= cls._ORIENT_CLS_MIN]
            upright, flipped = 0, 180
        elif direction == "tall":
            group = [(h, c) for w, h, c, s in lines
                     if h >= 1.5 * w and s >= cls._ORIENT_CLS_MIN]
            upright, flipped = 270, 90
        else:
            return None
        total = sum(n for n, _ in group)
        if not total:
            return None
        frac180 = sum(n for n, c in group if c == "180") / total
        if frac180 <= cls._ORIENT_VOTE:
            return upright
        if frac180 >= 1.0 - cls._ORIENT_VOTE:
            return flipped
        return None

    async def _detect_orientation(self, img: np.ndarray) -> int:
        """Upright angle (0/90/180/270) for `img`.

        A single detection + angle-class probe on a downscaled copy settles
        the clear cases (the vast majority) for a fraction of the cost of the
        4-angle full-OCR probe, which is kept as the fallback for ambiguous
        pages (few lines, mixed directions, split vote)."""
        if not self.settings.layout.detect_orientation:
            return 0
        small = resize_for_ocr(
            img, max_side=self.settings.layout.orientation_probe_side)
        lines = await asyncio.to_thread(
            self.primary_ocr.probe_orientation, small)
        angle = self._orientation_from_lines(lines)
        if angle is not None:
            return angle
        # Direction known but the up/down vote split: only two angles remain.
        direction = self._line_direction(lines)
        candidates = {"wide": (0, 180), "tall": (90, 270)}.get(
            direction, (0, 90, 180, 270))
        return await self._detect_orientation_full(img, candidates)

    async def _detect_orientation_full(
            self, img: np.ndarray,
            candidates: Tuple[int, ...] = (0, 90, 180, 270)) -> int:
        """Pick the upright angle by combining two cheap signals:

          * landscape-bbox count — RapidOCR returns each text line as a
            polygon; the axis-aligned bbox is wide for horizontal text and
            tall for vertical text. So 0°/180° (image upright or flipped)
            score high here and 90°/270° (image sideways) score low.
            Differentiator for landscape vs portrait orientations.

          * cls-disabled recognition confidence — with the angle-class
            model OFF the recogniser reads pixels as-is, so upside-down
            Latin/Devanagari text scores notably lower than upright text.
            Differentiator for 0° vs 180°.

        Score = landscape_count × avg_conf × text_len. The four probes
        run concurrently on a 720-px downscaled copy."""
        small = resize_for_ocr(img, max_side=720)
        rots = [rotate_image(small, a) for a in candidates]
        outs = await asyncio.gather(
            *(asyncio.to_thread(self.primary_ocr.extract_no_cls, r)
              for r in rots),
            return_exceptions=True,
        )
        best_angle, best_score = candidates[0], -1.0
        for angle, res in zip(candidates, outs):
            if isinstance(res, Exception) or res is None:
                continue
            n_landscape = sum(
                1 for w in res.words
                if (w.bbox[2] - w.bbox[0]) > (w.bbox[3] - w.bbox[1])
            )
            conf = res.avg_confidence or 0.0
            text_len = sum(len(w.text) for w in res.words)
            score = n_landscape * conf * text_len
            if score > best_score:
                best_angle, best_score = angle, score
        return best_angle

    @staticmethod
    def _crop(img: np.ndarray, bbox: Tuple[int, int, int, int],
              pad: int = 4) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        return img[y1:y2, x1:x2]

    # ── Stage 1 + 2: locate text and read it ─────────────────────────────────

    async def _read_field_regions(self, work: np.ndarray) -> List[Dict]:
        """YOLO field backend: crop the few coarse field regions and OCR
        them concurrently with asyncio.gather."""
        regions = self.layout.detect(work)
        crops = [self._crop(work, r.bbox) for r in regions]
        outs = await asyncio.gather(
            *(run_extract_async(self.primary_ocr, c) for c in crops),
            return_exceptions=True,
        )
        results: List[Dict] = []
        for region, crop, res in zip(regions, crops, outs):
            if isinstance(res, Exception) or res is None:
                results.append({"region": region, "crop": crop,
                                "text": "", "conf": 0.0, "engine": "none"})
            else:
                results.append({"region": region, "crop": crop,
                                "text": res.text,
                                "conf": res.avg_confidence or 0.0,
                                "engine": res.engine})
        return results

    # Aadhaar backs print the English address in a light, often worn font; a
    # lower detector box threshold keeps those faint lines (measured: address
    # 10/17 -> 13/17 on the labelled set). Other document types keep RapidOCR's
    # default — on passports the extra boxes break back-page classification.
    _BOX_THRESH = {"AADHAAR": 0.3}

    async def _read_fused(self, work: np.ndarray,
                          box_thresh: float = 0.0) -> List[Dict]:
        """Generic line backend: detection + recognition are a single fused
        ONNX pass. Re-cropping each detected line and re-OCRing it only
        fragments words and loses context, so we don't — the fused pass
        already yields per-line box + text + confidence.

        Lines scoring below RapidOCR's `text_score` gate are returned by
        `read()` too (the stock call drops them); `_recover_dropped_lines`
        re-reads those from a padded crop. That step is purely additive —
        every line that passed the gate is left exactly as-is."""
        lines = await asyncio.to_thread(self.primary_ocr.read, work,
                                        box_thresh)
        gate = self.primary_ocr.TEXT_SCORE
        kept = [wb for wb in lines if (wb.confidence or 0.0) >= gate]
        dropped = [wb for wb in lines if (wb.confidence or 0.0) < gate]
        results = [
            {"region": Region(wb.bbox, "text_line", 1.0),
             "crop": self._crop(work, wb.bbox),
             "text": wb.text,
             "conf": wb.confidence or 0.0,
             "engine": "rapidocr",
             "cls": (wb.cls_label, wb.cls_score)}
            for wb in kept
        ]
        if dropped:
            results += await asyncio.to_thread(
                self._recover_dropped_lines, work, kept, dropped)
        return results

    @staticmethod
    def _box_covered(box: Tuple[int, int, int, int], words) -> bool:
        """True when `box` is already represented by a recognised word —
        its centre falls inside a found word box, or it overlaps one by a
        substantial fraction of its own area. Used to keep line recovery
        purely additive (never re-emit a line the fused pass already
        returned)."""
        bx1, by1, bx2, by2 = box
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        barea = max(1, (bx2 - bx1) * (by2 - by1))
        for w in words:
            wx1, wy1, wx2, wy2 = w.bbox
            if wx1 <= cx <= wx2 and wy1 <= cy <= wy2:
                return True
            ix = max(0, min(bx2, wx2) - max(bx1, wx1))
            iy = max(0, min(by2, wy2) - max(by1, wy1))
            if ix * iy >= 0.4 * barea:
                return True
        return False

    def _recover_dropped_lines(self, work: np.ndarray, found_words,
                               dropped_words) -> List[Dict]:
        """Re-OCR text lines the fused pass located but scored below the gate.

        The detector reports a box for every text line, but a line whose
        recognition confidence is below `text_score` is discarded — which
        happens when the detector hands the recogniser a clipped quad (a
        tight DOB date is a frequent victim). Recognising the padded
        axis-aligned crop instead reads it at full confidence. All crops go
        through the recogniser in one batched call.
        """
        boxes, crops = [], []
        for wb in dropped_words:
            if self._box_covered(wb.bbox, found_words):
                continue
            crop = self._crop(work, wb.bbox, pad=8)
            if crop.size == 0:
                continue
            boxes.append(wb.bbox)
            crops.append(crop)
        recovered: List[Dict] = []
        for box, crop, (text, conf) in zip(
                boxes, crops, self.primary_ocr.recognize(crops)):
            text = text.strip()
            # Keep only confident, non-trivial reads so a forced
            # recognition of a stray graphic cannot inject noise.
            if conf >= 0.5 and any(ch.isalnum() for ch in text):
                recovered.append({
                    "region": Region(box, "text_line", 1.0),
                    "crop": crop, "text": text, "conf": conf,
                    "engine": "rapidocr",
                })
        return recovered

    async def _locate_and_read(self, work: np.ndarray,
                               box_thresh: float = 0.0) -> List[Dict]:
        """Stage 1+2 then Stage 3: produce per-region text, then Surya-retry
        only the low-confidence regions (concurrently)."""
        if self.layout.backend == "yolo":
            results = await self._read_field_regions(work)
        else:
            results = await self._read_fused(work, box_thresh)

        # ── Stage 3: Surya — ONLY on the low-confidence crops ──
        weak = [i for i, r in enumerate(results)
                if r["conf"] < self.settings.ocr.fallback_threshold]
        if weak:
            self._ensure_fallback()
            if self.fallback_ocr is not None:
                fb = await asyncio.gather(
                    *(run_extract_async(self.fallback_ocr, results[i]["crop"])
                      for i in weak),
                    return_exceptions=True,
                )
                for i, res in zip(weak, fb):
                    if isinstance(res, Exception) or res is None:
                        continue
                    fb_conf = res.avg_confidence or 0.0
                    if res.text.strip() and fb_conf >= results[i]["conf"]:
                        results[i].update(text=res.text, conf=fb_conf,
                                          engine="surya")
        return results

    async def _read_upright(self, work: np.ndarray, angle: int,
                            box_thresh: float = 0.0
                            ) -> Tuple[np.ndarray, int, List[Dict]]:
        """Read `work`, then double-check its orientation with the per-line
        angle classes the read produced for free. If the lines say the page
        is still sideways/upside down (the probe got it wrong), turn it and
        read once more. Returns ``(work, angle, region_results)``."""
        results = await self._locate_and_read(work, box_thresh)
        if not self.settings.layout.detect_orientation:
            return work, angle, results
        stats = []
        for r in results:
            if "cls" not in r:
                continue
            x1, y1, x2, y2 = r["region"].bbox
            stats.append((x2 - x1, y2 - y1, r["cls"][0], r["cls"][1]))
        fix = self._orientation_from_lines(stats)
        if fix:
            log.info("orientation re-check: turning page a further %d°", fix)
            work = rotate_image(work, fix)
            results = await self._locate_and_read(work, box_thresh)
            angle = (angle + fix) % 360
        return work, angle, results

    # ── Classification & extraction ──────────────────────────────────────────

    def classify_document(self, text: str) -> str:
        text = text.upper()
        # RapidOCR frequently runs adjacent words together (e.g. it reads
        # "REPUBLIC OF INDIA" as "REPUBLICOFINDIA"). Match signatures against
        # both the raw and whitespace-stripped text.
        nospace = re.sub(r"\s+", "", text)
        scores = {k: 0 for k in self.patterns}

        signatures = {
            "AADHAAR": [("UIDAI", 30), ("AADHAAR", 30)],
            "PAN": [("INCOME TAX", 40), ("PAN", 30)],
            "VOTER_ID": [("ELECTION COMMISSION", 50), ("ELECTOR", 30),
                         ("PHOTO IDENTITY", 30), ("EPIC", 50)],
            "DRIVING_LICENSE": [("DL NO", 50), ("VALID TILL", 30), ("MCWG", 25),
                                ("LMV", 25), ("DRIVING LICENCE", 40),
                                ("DRIVING LICENSE", 40),
                                # Vehicle Registration Certificate (RC card)
                                # — issued by the RTO and treated by the
                                # frontend under the same 'license' bucket.
                                ("REGISTRATION CERTIFICATE", 60),
                                ("VEHICLE CLASS", 50), ("CHASSIS", 40),
                                ("ENGINE/MOTOR", 40), ("MAKER", 30),
                                ("FORM 23A", 60), ("REGN NO", 50),
                                ("REGN. NUMBER", 50),
                                ("TRANSPORT DEPARTMENT", 50)],
            "PASSPORT": [("PASSPORT", 60), ("REPUBLIC OF INDIA", 50),
                         ("GIVEN NAME", 25), ("DATE OF EXPIRY", 25),
                         ("PLACE OF ISSUE", 25),
                         # Back-side keywords (no MRZ on back — these are
                         # what makes the back identifiable as a passport
                         # rather than e.g. a driving licence whose number
                         # format collides with the passport File No).
                         ("NAME OF FATHER", 60), ("NAME OF MOTHER", 60),
                         ("NAME OF SPOUSE", 40), ("LEGAL GUARDIAN", 40),
                         ("OLD PASSPORT", 60), ("FILE NO", 50)],
        }
        for doc, vals in signatures.items():
            for keyword, weight in vals:
                kw_ns = re.sub(r"\s+", "", keyword)
                if keyword in text or kw_ns in nospace:
                    scores[doc] += weight

        # Voter EPIC pattern — strict word boundaries so we don't match
        # the 'FCS' fragment inside a chassis number like ME3J3C5FCS2016714.
        if re.search(r"(?<![A-Z0-9])[A-Z]{3}\d{7}(?![A-Z0-9])", text):
            scores["VOTER_ID"] += 50
        if re.search(r"[A-Z]{2}[\s-]?\d{2}[\s-]?\d{11}", text):
            scores["DRIVING_LICENSE"] += 60
        if re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text):   scores["PAN"] += 50
        # Anchored exactly like the AADHAAR extraction pattern: the
        # (?<!\d)/(?!\d) guards stop a clean 12-digit run being matched
        # inside a longer number (e.g. a 13-digit DL number).
        if re.search(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)", text):
            scores["AADHAAR"] += 50

        # Passport MRZ line is a very strong, format-specific signal.
        if re.search(r"P<[A-Z]{3}", text) or re.search(r"P<<[A-Z]", text):
            scores["PASSPORT"] += 80
        # Indian passport number — 1-2 letters + 6-7 digits as a whole token.
        if re.search(r"\b[A-Z]{1,2}\d{6,7}\b", text):
            scores["PASSPORT"] += 30

        log.debug("class scores: %s", scores)
        top = max(scores, key=scores.get)
        return top if scores[top] else "UNKNOWN"

    def verify_and_extract(self, text: str, doc: str) -> str | None:
        matches = re.findall(self.patterns[doc], text.upper())
        if not matches:
            return None
        # Collapse any internal whitespace/newlines the match spanned.
        return re.sub(r"\s+", " ", matches[0]).strip()

    def mask_id(self, val: str, doc: str) -> str:
        """Display-only masked string (for the UI text field)."""
        if doc == "AADHAAR":
            digits = re.sub(r"\D", "", val)
            return f"XXXX XXXX {digits[-4:]}"
        if doc == "PAN":
            return val[:2] + "XXXXX" + val[-3:]
        return "X" * max(0, len(val) - 4) + val[-4:]

    # ── Stage 4: direct masking via layout boxes ─────────────────────────────

    def _is_id_hit(self, text: str, pattern: str, id_clean: str,
                   doc_type: str) -> bool:
        """Does this recognised region hold the ID number, or part of it?"""
        up = text.upper()
        word = re.sub(r"[^A-Z0-9]", "", up)
        if not word:
            return False
        if re.search(pattern, up):
            return True
        if not id_clean:
            return False
        # whole ID inside the crop, or crop is a long ID fragment
        if id_clean in word or (len(word) >= 6 and word in id_clean):
            return True
        # Aadhaar sets the 12 digits as three space-separated 4-digit groups.
        # A detector that splits on those gaps returns 4-char crops, which the
        # 12-digit pattern cannot match and the >=6 fragment rule rejects — so
        # a wide-set number got no hits at all and the card went out unmasked.
        # _merge_nearby_boxes rejoins the groups into one box afterwards.
        return (doc_type == "AADHAAR" and len(word) >= 4
                and word.isdigit() and word in id_clean)

    def _merge_gap_for(self, doc_type: str, boxes) -> int:
        """Join distance handed to _merge_nearby_boxes.

        Aadhaar's digit groups sit roughly a character-width apart, which on a
        high-resolution scan is wider than the flat `merge_gap` — leaving three
        separately-masked groups instead of one box. Scale the gap to the glyph
        height so the groups merge however large the card was scanned."""
        gap = self.settings.mask.merge_gap
        if doc_type != "AADHAAR" or not boxes:
            return gap
        heights = sorted(y2 - y1 for (_, y1, _, y2) in boxes)
        return max(gap, int(1.2 * heights[len(heights) // 2]))

    def _sensitive_boxes(self, region_results: List[Dict], doc_type: str,
                         extracted_id: str) -> List[Tuple[int, int, int, int]]:
        """Pick the layout regions that hold the sensitive ID.

        Geometry comes from the layout detector; the *which-region* decision
        uses the recognised crop text — so no Tesseract word pass is needed.
        """
        pattern = self.patterns[doc_type]
        id_clean = re.sub(r"[^A-Z0-9]", "", extracted_id.upper())
        return [rr["region"].bbox for rr in region_results
                if self._is_id_hit(rr["text"], pattern, id_clean, doc_type)]

    def _merge_nearby_boxes(self, boxes, gap: int):
        """Join boxes on the same row within `gap` px (Aadhaar's 3 groups)."""
        if not boxes:
            return []
        rects = sorted(
            [(min(a, c), min(b, d), max(a, c), max(b, d))
             for (a, b, c, d) in boxes],
            key=lambda r: (r[1], r[0]),
        )
        merged, cur = [], list(rects[0])
        for x1, y1, x2, y2 in rects[1:]:
            cur_h = max(1, cur[3] - cur[1])
            overlap_y = min(y2, cur[3]) - max(y1, cur[1])
            if overlap_y >= 0.4 * cur_h and x1 <= cur[2] + gap:
                cur = [min(cur[0], x1), min(cur[1], y1),
                       max(cur[2], x2), max(cur[3], y2)]
            else:
                merged.append(tuple(cur))
                cur = [x1, y1, x2, y2]
        merged.append(tuple(cur))
        return merged

    def create_masked_image(self, img: np.ndarray, region_results: List[Dict],
                            extracted_id: str, doc_type: str,
                            output_path: str) -> str | None:
        """Black out the sensitive ID zone directly with cv2.rectangle."""
        masked = img.copy()
        h, w = masked.shape[:2]
        ms = self.settings.mask

        raw = self._sensitive_boxes(region_results, doc_type, extracted_id)
        boxes = self._merge_nearby_boxes(
            raw, gap=self._merge_gap_for(doc_type, raw),
        )
        drawn = 0

        for (x1, y1, x2, y2) in boxes:
            pad = max(4, int((y2 - y1) * ms.pad_ratio))
            bx1 = max(0, x1 - pad); by1 = max(0, y1 - pad)
            bx2 = min(w - 1, x2 + pad); by2 = min(h - 1, y2 + pad)

            # Aadhaar prints the 12-digit number on BOTH sides, so every
            # matched occurrence is partial-masked: cover the leading 8
            # digits, keep only the last 4 readable.
            if doc_type == "AADHAAR":
                bx2 = bx2 - int((bx2 - bx1) * ms.aadhaar_visible_ratio)

            cv2.rectangle(masked, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            drawn += 1

        if drawn == 0:
            log.warning("No sensitive region located — nothing masked.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                    exist_ok=True)
        cv2.imwrite(output_path, masked)
        return output_path if drawn > 0 else None

    # ── Mask-identity: geometry-only redaction for the frontend ──────────────

    @staticmethod
    def _box_agg(box: Tuple[int, int, int, int], hits) -> Tuple[float, int]:
        """Aggregate the hit regions whose centre falls inside a (possibly
        merged) mask box: report the strongest recognition confidence and the
        total alphanumeric char count, used to size the 'keep last 4' window
        off the regions' own glyph width."""
        mx1, my1, mx2, my2 = box
        best, chars = 0.0, 0
        for (x1, y1, x2, y2), conf, nchars in hits:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                best = max(best, conf)
                chars += nchars
        return best, chars

    def _id_mask_rects(self, region_results: List[Dict], doc_type: str,
                       extracted_id: str, img_shape) -> List[Tuple]:
        """Compute the exact rectangles that black out every printed
        occurrence of the ID number, keeping only the last 4 characters
        visible. Returns (x1, y1, x2, y2, conf, field) tuples — the same
        geometry that gets drawn, so a frontend can replicate the mask."""
        h, w = img_shape[:2]
        ms = self.settings.mask
        field_name = self.ID_FIELD.get(doc_type, "id_number")

        # Which regions hold the ID — same decision as _sensitive_boxes, but
        # we keep each region's recognition confidence and char count.
        pattern = self.patterns[doc_type]
        id_clean = re.sub(r"[^A-Z0-9]", "", extracted_id.upper())
        # (bbox, conf, nchars)
        hits: List[Tuple[Tuple[int, int, int, int], float, int]] = []
        for rr in region_results:
            word = re.sub(r"[^A-Z0-9]", "", rr["text"].upper())
            if not word:
                continue
            if self._is_id_hit(rr["text"], pattern, id_clean, doc_type):
                hits.append((rr["region"].bbox, rr["conf"], len(word)))

        boxes = [b for b, _, _ in hits]
        merged = self._merge_nearby_boxes(
            boxes, gap=self._merge_gap_for(doc_type, boxes))
        rects: List[Tuple] = []
        for (x1, y1, x2, y2) in merged:
            conf, nchars = self._box_agg((x1, y1, x2, y2), hits)
            pad = max(4, int((y2 - y1) * ms.pad_ratio))
            bx1 = max(0, x1 - pad); by1 = max(0, y1 - pad)
            rx2 = min(w - 1, x2 + pad); by2 = min(h - 1, y2 + pad)

            # Keep only the last 4 glyphs readable by shrinking the right edge.
            if doc_type == "AADHAAR":
                # Tuned ratio: the 12-digit number prints across up to three
                # groups; keep the right ~third (last 4 digits).
                bx2 = rx2 - int((rx2 - bx1) * ms.aadhaar_visible_ratio)
            else:
                # Size the reveal off the region's own glyph width rather than
                # the regex match length. A long File No. matched by a 9-char
                # passport pattern would otherwise leave most of it readable.
                char_w = (x2 - x1) / max(nchars, 1)
                bx2 = int(round(x2 - 4 * char_w))

            bx2 = min(rx2, max(bx1 + 1, bx2))
            rects.append((bx1, by1, bx2, by2, conf, field_name))
        return rects

    async def _preprocess_for_display(
            self, img: np.ndarray, crop: bool = True) -> Tuple[np.ndarray, bool]:
        """Produce the render returned to the caller by /mask-identity:
        the document cropped out of its background, straightened, stood
        upright, and sized for OCR.

        Unlike `_preprocess_ocr` (used by the OCR contract, whose
        `auto_crop_document` deliberately refuses any crop that would drop
        >30% of the frame) this uses `crop_document`, which finds the document
        quad against the background and perspective-warps it. That is what
        actually crops a card photographed on a desk/hand — the conservative
        bbox crop always bailed out and returned the full original.

        Every mask rectangle is computed on this render, so the returned
        masked/unmasked images and `masked_regions` all share its coordinates.

        With ``crop=False`` the quad crop/warp is skipped and the full frame is
        merely straightened and stood upright — the retry path for when the
        crop amputated the ID strip. Returns ``(work, did_crop)`` so the caller
        can tell whether a crop actually happened and decide to retry."""
        def _crop_and_straighten() -> Tuple[np.ndarray, bool]:
            if crop:
                cropped, dbg = crop_document(img)
                did = bool(dbg.get("cropped"))
            else:
                cropped, did = img, False
            # Residual skew (the quad warp already removes rotation; this is a
            # no-op on a straight image, and it rescues the fallback crop path).
            return deskew(cropped)[0], did

        # CPU-bound OpenCV work runs off the event loop.
        straight, did_crop = await asyncio.to_thread(_crop_and_straighten)
        angle = await self._detect_orientation(straight)
        work = resize_for_ocr(rotate_image(straight, angle),
                              max_side=self.settings.work_max_side)
        return work, did_crop

    async def _preprocess_ocr(
            self, img: np.ndarray, crop: bool = True
    ) -> Tuple[np.ndarray, bool, int]:
        """crop → deskew → orient → resize for the OCR-contract path
        (`process_and_verify`).

        With ``crop=False`` the contour `auto_crop_document` is skipped and the
        full frame is used — the retry path for when a crop amputated the ID
        strip. Returns ``(work, did_crop, angle)``."""
        def _crop_and_straighten() -> Tuple[np.ndarray, bool]:
            if crop:
                base, crop_dbg = auto_crop_document(img)
                did = bool(crop_dbg.get("cropped"))
            else:
                base, did = img, False
            return deskew(base)[0], did

        # CPU-bound OpenCV work runs off the event loop.
        desk, did_crop = await asyncio.to_thread(_crop_and_straighten)
        angle = await self._detect_orientation(desk)
        work = resize_for_ocr(rotate_image(desk, angle),
                              max_side=self.settings.work_max_side)
        return work, did_crop, angle

    async def mask_identity(self, path: str, doc_type: str) -> Dict:
        """Return two cropped + deskewed + upright ("work"-space) renders of
        the document:

          * ``unmasked_image`` — the clean cropped/rotated document, for
            authorised/internal use (it shows the full ID number);
          * ``masked_image``   — the same render with the ID number redacted
            (last 4 kept), safe to display to a user.

        Both share the SAME coordinate space, so ``masked_regions`` (also in
        that space) overlay either image directly — no projection back to the
        original upload is needed.

        `doc_type` is the internal upper-case key (AADHAAR / PAN / VOTER_ID /
        PASSPORT / DRIVING_LICENSE). ``unmasked_image`` is present whenever the
        document could be OCR-processed; ``masked_image`` is present only when
        the ID number was located — it fails closed, so a "masked" render is
        never a silently-unmasked one."""
        self._ensure_ocr()
        out: Dict = {
            "document_detected": False,
            "unmasked_image": None,      # PNG bytes: cropped + rotated
            "masked_image": None,        # PNG bytes: cropped + rotated + redacted
            "masked_regions": [],        # [x, y, w, h] in the work-image space
            "message": None,
            "status": 200,
        }
        if not self.ocr_available:
            out.update(message="OCR backend unavailable", status=503)
            return out

        doc_type = doc_type.upper()
        if doc_type not in self.patterns:
            out.update(message=f"Unsupported document_type: {doc_type}",
                       status=400)
            return out

        # crop → straighten → orient → resize. Both output images ARE this
        # `work` render, so masks drawn on it need no projection back to the
        # original upload.
        #
        # Over-crop retry: crop_document can shear off the ID strip on some
        # photos. Try the cropped render first; if no ID is located AND a crop
        # actually happened, re-read the full uncropped frame. The unmasked
        # image is encoded from whichever render won, so a recovered document
        # is returned cropped-or-full rather than lost.
        img = await asyncio.to_thread(self.load_document_image, path)

        async def _attempt(crop: bool) -> Dict:
            w, did_crop = await self._preprocess_for_display(img, crop=crop)
            w, _, rr = await self._read_upright(
                w, 0, self._BOX_THRESH.get(doc_type, 0.0))
            ft = "\n".join(r["text"] for r in rr if r["text"]).upper()
            ext = self.verify_and_extract(ft, doc_type)
            return {"work": w, "did_crop": did_crop,
                    "region_results": rr, "extracted": ext}

        att = await _attempt(crop=True)
        if att["extracted"] is None and att["did_crop"]:
            retry = await _attempt(crop=False)
            if retry["extracted"] is not None:
                att = retry

        work = att["work"]
        region_results = att["region_results"]
        extracted = att["extracted"]

        ok, buf = cv2.imencode(".png", work)
        if not ok:
            out.update(message="Failed to encode document image", status=500)
            return out
        out["unmasked_image"] = buf.tobytes()
        out["document_detected"] = extracted is not None

        work_rects: List[Tuple] = []
        if extracted:
            work_rects.extend(
                self._id_mask_rects(region_results, doc_type, extracted,
                                    work.shape))

        if not work_rects:
            # The unmasked render is still returned; the masked one is withheld
            # (fail closed) so the caller never mistakes an unredacted image for
            # a masked one.
            out["message"] = (
                f"No {doc_type} identity number located"
                if not extracted else
                "Identity number found but could not be localized for masking")
            return out

        # The rectangles from _id_mask_rects are already in `work` space (final
        # geometry, last-4 kept) — draw them straight onto the work render.
        masked = work.copy()
        masked_regions: List[Dict] = []
        for (x1, y1, x2, y2, conf, field_name) in work_rects:
            cv2.rectangle(masked, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 0), -1)
            masked_regions.append({
                "field": field_name,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": round(float(conf), 2),
            })

        ok, buf = cv2.imencode(".png", masked)
        if not ok:
            out.update(message="Failed to encode masked image", status=500)
            return out

        out.update(
            masked_image=buf.tobytes(),
            masked_regions=masked_regions,
            message=None,
            status=200,
        )
        return out

    # ── Top-level entry point ────────────────────────────────────────────────

    async def process_and_verify(self, path: str, intended: str) -> Dict:
        """Async entry point. Streamlit calls it via `asyncio.run(...)`."""
        t0 = time.time()
        self._ensure_ocr()

        result: Dict = {
            "status": "FAILED", "actual_type": "UNKNOWN",
            "ocr_engine": "rapidocr", "ocr_avg_confidence": None,
            "ocr_decision_reason": "", "extracted_text": "",
            "output_json": None,
        }
        if not self.ocr_available:
            result["message"] = "OCR backend unavailable"
            result["output_json"] = failure_envelope(
                "OCR backend unavailable", status=503)
            return result

        # ── preprocess + read + classify + extract, with an over-crop retry ──
        # The contour crop can amputate the ID strip on some real photos. Run
        # the normal cropped path first (unchanged for the working majority);
        # if it yields nothing usable AND a crop actually happened, re-read the
        # full uncropped frame before giving up. If neither finds the ID we
        # still fail — the retry only ever rescues an over-crop, never masks a
        # genuinely unreadable document.
        img = await asyncio.to_thread(self.load_document_image, path)

        async def _attempt(crop: bool) -> Dict:
            work, did_crop, angle = await self._preprocess_ocr(img, crop=crop)
            work, angle, rr = await self._read_upright(
                work, angle, self._BOX_THRESH.get(intended, 0.0))
            ft = "\n".join(r["text"] for r in rr if r["text"]).upper()
            act = self.classify_document(ft)
            ext = (self.verify_and_extract(ft, act)
                   if act == intended else None)
            return {"work": work, "did_crop": did_crop, "angle": angle,
                    "region_results": rr, "full_text": ft,
                    "actual": act, "extracted": ext}

        att = await _attempt(crop=True)
        retried = False
        if att["extracted"] is None and att["did_crop"]:
            retried = True
            retry = await _attempt(crop=False)
            # Keep the uncropped read only when it actually recovered the ID.
            if retry["extracted"] is not None:
                att = retry

        work = att["work"]
        region_results = att["region_results"]
        full_text = att["full_text"]
        actual = att["actual"]
        angle = att["angle"]
        confs = [rr["conf"] for rr in region_results if rr["conf"] > 0]
        used_surya = any(rr["engine"] == "surya" for rr in region_results)

        result.update({
            "actual_type": actual,
            "extracted_text": full_text,
            "ocr_avg_confidence": (sum(confs) / len(confs)) if confs else None,
            "ocr_decision_reason": (
                f"located {len(region_results)} regions; "
                f"surya_fallback={'yes' if used_surya else 'no'}; "
                f"retry_uncropped={'yes' if retried else 'no'}"),
            "ocr_engine": "rapidocr+surya" if used_surya else "rapidocr",
            "ocr_debug": {"angle": angle, "regions": len(region_results),
                          "retry_uncropped": retried,
                          "used_crop": att["did_crop"]},
        })

        if actual != intended:
            result["message"] = f"Expected {intended}, got {actual}"
            result["output_json"] = failure_envelope(result["message"])
            result["elapsed_sec"] = round(time.time() - t0, 3)
            return result

        extracted = att["extracted"]
        if not extracted:
            result["message"] = "ID extraction failed"
            result["output_json"] = failure_envelope(result["message"])
            result["elapsed_sec"] = round(time.time() - t0, 3)
            return result

        # ── Stage 4: direct masking (debug/CLI only — the API never serves
        # this file, so by default nothing is written to disk) ──
        masked = None
        if self.settings.save_masked_output:
            # Force a raster extension: the input may be a PDF/webp whose
            # extension cv2.imwrite can't encode.
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(
                self.settings.output_dir, f"masked_{stem}.png")
            masked = await asyncio.to_thread(
                self.create_masked_image,
                work, region_results, extracted, actual, out_path)

        result.update({
            "status": "SUCCESS",
            "extracted_id": extracted,
            "masked_id": self.mask_id(extracted, actual),
            "masked_image_file": masked,
            "output_json": build_output_json(actual, region_results, full_text),
            "elapsed_sec": round(time.time() - t0, 3),
        })
        return result

    # ── Debug helper (used by the Streamlit debug panel) ─────────────────────

    def build_preprocess_debug(self, path: str) -> Dict:
        img = self.load_document_image(path)
        cropped, crop_dbg = auto_crop_document(img)
        desk, desk_dbg = deskew(cropped)
        self._ensure_ocr()
        angle = asyncio.run(self._detect_orientation(desk))
        work = resize_for_ocr(rotate_image(desk, angle),
                              max_side=self.settings.work_max_side)
        regions = self.layout.detect(work) if self.layout else []

        overlay = work.copy()
        for r in regions:
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)

        return {
            "base_color": work,
            "regions_overlay": overlay,
            "crop": crop_dbg,
            "deskew": desk_dbg,
            "angle": angle,
            "region_count": len(regions),
        }
