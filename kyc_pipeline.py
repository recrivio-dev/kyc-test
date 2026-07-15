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
import os
import re
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium

from config import SETTINGS, Settings
from layout_detector import LayoutDetector, Region
from ocr_engines import RapidOCREngine, SuryaOCREngine, run_extract_async
from output_schema import build_output_json, failure_envelope
from preprocessing import (auto_crop_document, crop_document, deskew,
                          resize_for_ocr, rotate_image)


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
                    lang=self.settings.ocr.primary_lang)
                self.ocr_available = True
            except Exception as e:                   # noqa: BLE001
                print(f"[ERROR] RapidOCR init failed: {e}")
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
                print(f"[WARN] Surya fallback unavailable: {e}")

    def get_printable_name(self, doc: str) -> str:
        return self.DISPLAY_NAMES.get(doc, "Unknown")

    # ── Image loading ────────────────────────────────────────────────────────

    def load_document_image(self, path: str) -> np.ndarray:
        if path.lower().rsplit(".", 1)[-1] == "pdf":
            pdf = pdfium.PdfDocument(path)
            pil = pdf[0].render(scale=3).to_pil()
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        return img

    # ── Stage helper: orientation ────────────────────────────────────────────

    async def _detect_orientation(self, img: np.ndarray) -> int:
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
        if not self.settings.layout.detect_orientation:
            return 0

        small = resize_for_ocr(img, max_side=720)
        rots = [rotate_image(small, a) for a in (0, 90, 180, 270)]
        outs = await asyncio.gather(
            *(asyncio.to_thread(self.primary_ocr.extract_no_cls, r)
              for r in rots),
            return_exceptions=True,
        )
        best_angle, best_score = 0, -1.0
        for angle, res in zip((0, 90, 180, 270), outs):
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

    async def _read_fused(self, work: np.ndarray) -> List[Dict]:
        """Generic line backend: detection + recognition are a single fused
        ONNX pass. Re-cropping each detected line and re-OCRing it only
        fragments words and loses context, so we don't — the fused pass
        already yields per-line box + text + confidence.

        The fused pass does, however, silently drop any line whose
        recognition confidence falls below RapidOCR's internal
        `text_score` gate. `_recover_dropped_lines` adds those back; that
        step is purely additive — every line the fused pass returned is
        left exactly as-is."""
        res = await run_extract_async(self.primary_ocr, work)
        results = [
            {"region": Region(wb.bbox, "text_line", 1.0),
             "crop": self._crop(work, wb.bbox),
             "text": wb.text,
             "conf": wb.confidence or 0.0,
             "engine": "rapidocr"}
            for wb in res.words
        ]
        results += await asyncio.to_thread(
            self._recover_dropped_lines, work, res.words)
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

    def _recover_dropped_lines(self, work: np.ndarray,
                               found_words) -> List[Dict]:
        """Re-OCR text regions the fused pass located but discarded.

        RapidOCR's detector reports a box for every text line, but the
        fused detect+recognise pass silently drops any line whose
        recognition confidence is below its internal `text_score` gate —
        which happens when the detector hands the recogniser a clipped
        quad (a tight DOB date is a frequent victim). A detection-only
        pass still reports the box; recognising the padded crop with
        detection OFF reads it at full confidence.
        """
        recovered: List[Dict] = []
        for box in self.primary_ocr.detect(work):
            if self._box_covered(box, found_words):
                continue
            crop = self._crop(work, box, pad=8)
            if crop.size == 0:
                continue
            rec = self.primary_ocr.extract_rec_only(crop)
            text = rec.text.strip()
            conf = rec.avg_confidence or 0.0
            # Keep only confident, non-trivial reads so a forced
            # recognition of a stray graphic cannot inject noise.
            if conf >= 0.5 and any(ch.isalnum() for ch in text):
                recovered.append({
                    "region": Region(box, "text_line", 1.0),
                    "crop": crop, "text": text, "conf": conf,
                    "engine": "rapidocr",
                })
        return recovered

    async def _locate_and_read(self, work: np.ndarray) -> List[Dict]:
        """Stage 1+2 then Stage 3: produce per-region text, then Surya-retry
        only the low-confidence regions (concurrently)."""
        if self.layout.backend == "yolo":
            results = await self._read_field_regions(work)
        else:
            results = await self._read_fused(work)

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

        print("CLASS SCORES:", scores)
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
            print("[WARN] No sensitive region located — nothing masked.")

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

    # ── Coordinate transform: work-image space ⇄ original-image space ────────

    @staticmethod
    def _rot90_affine(img: np.ndarray, angle: int):
        """Apply rotate_image(img, angle) and return (3×3 affine, rotated).

        The affine maps a point in `img` to its location in the rotated
        output — matching cv2.rotate's 90° steps exactly — so it can be
        composed into the full original→work transform."""
        H, W = img.shape[:2]
        rotated = rotate_image(img, angle)
        a = angle % 360
        if a == 90:        # ROTATE_90_CLOCKWISE:        (x, y) → (H-1-y, x)
            T = np.array([[0, -1, H - 1], [1, 0, 0], [0, 0, 1]], dtype=float)
        elif a == 180:     # ROTATE_180:                 (x, y) → (W-1-x, H-1-y)
            T = np.array([[-1, 0, W - 1], [0, -1, H - 1], [0, 0, 1]], dtype=float)
        elif a == 270:     # ROTATE_90_COUNTERCLOCKWISE: (x, y) → (y, W-1-x)
            T = np.array([[0, 1, 0], [-1, 0, W - 1], [0, 0, 1]], dtype=float)
        else:              # 0° (or generic, treated as identity for mapping)
            T = np.eye(3, dtype=float)
        return T, rotated

    async def _preprocess_for_display(
            self, img: np.ndarray, crop: bool = True) -> Tuple[np.ndarray, bool]:
        """Produce the render returned to the caller by /mask-identity:
        the document cropped out of its background, straightened, stood
        upright, and sized for OCR.

        Unlike `_preprocess_with_transform` (used by the OCR contract, whose
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
        if crop:
            cropped, dbg = crop_document(img)
            did_crop = bool(dbg.get("cropped"))
        else:
            cropped, did_crop = img, False
        # Residual skew (the quad warp already removes rotation; this is a
        # no-op on a straight image, and it rescues the fallback crop path).
        straight, _ = deskew(cropped)
        angle = await self._detect_orientation(straight)
        upright = rotate_image(straight, angle)
        work = resize_for_ocr(upright, max_side=self.settings.work_max_side)
        return work, did_crop

    async def _preprocess_ocr(
            self, img: np.ndarray, crop: bool = True
    ) -> Tuple[np.ndarray, bool, int]:
        """crop → deskew → orient → resize for the OCR-contract path
        (`process_and_verify`).

        With ``crop=False`` the contour `auto_crop_document` is skipped and the
        full frame is used — the retry path for when a crop amputated the ID
        strip. Returns ``(work, did_crop, angle)``."""
        if crop:
            base, crop_dbg = auto_crop_document(img)
            did_crop = bool(crop_dbg.get("cropped"))
        else:
            base, did_crop = img, False
        desk, _ = deskew(base)
        angle = await self._detect_orientation(desk)
        work = resize_for_ocr(rotate_image(desk, angle),
                              max_side=self.settings.work_max_side)
        return work, did_crop, angle

    async def _preprocess_with_transform(self, img: np.ndarray):
        """Run the same crop→deskew→orient→resize chain as process_and_verify,
        but also accumulate the 3×3 affine that maps ORIGINAL image coords to
        the returned `work` image coords. Inverting it projects mask boxes
        found on `work` back onto the caller's original image."""
        T = np.eye(3, dtype=float)

        # 1. auto-crop — an axis-aligned crop, i.e. a pure translation.
        cropped, crop_dbg = auto_crop_document(img)
        if crop_dbg.get("cropped"):
            cx1, cy1, _, _ = crop_dbg["bbox"]
            Tc = np.eye(3, dtype=float); Tc[0, 2] = -cx1; Tc[1, 2] = -cy1
            T = Tc @ T

        # 2. deskew — rotation about the cropped image centre (dims preserved).
        desk, desk_dbg = deskew(cropped)
        if desk_dbg.get("deskewed"):
            ch, cw = cropped.shape[:2]
            M = cv2.getRotationMatrix2D((cw // 2, ch // 2),
                                        desk_dbg["angle_deg"], 1.0)
            Td = np.eye(3, dtype=float); Td[:2, :] = M
            T = Td @ T

        # 3. orientation — 0/90/180/270.
        angle = await self._detect_orientation(desk)
        Tr, rotated = self._rot90_affine(desk, angle)
        T = Tr @ T

        # 4. resize for OCR — uniform downscale (only if larger than max_side).
        work = resize_for_ocr(rotated, max_side=self.settings.work_max_side)
        rh, rw = rotated.shape[:2]
        if max(rh, rw) > self.settings.work_max_side:
            s = self.settings.work_max_side / float(max(rh, rw))
            Ts = np.eye(3, dtype=float); Ts[0, 0] = s; Ts[1, 1] = s
            T = Ts @ T

        return work, T

    @staticmethod
    def _project_box(box, T_inv, orig_shape):
        """Map an axis-aligned work-space box through T_inv and return its
        axis-aligned bounding box in original-image coords (clamped). For a
        rotated/deskewed page the work box maps to a tilted quad — taking its
        bounding box over-masks slightly, which is the safe direction."""
        h, w = orig_shape[:2]
        x1, y1, x2, y2 = box
        corners = np.array([[x1, y1, 1], [x2, y1, 1],
                            [x2, y2, 1], [x1, y2, 1]], dtype=float).T
        proj = T_inv @ corners
        xs = proj[0] / proj[2]; ys = proj[1] / proj[2]
        ox1 = max(0, int(np.floor(xs.min()))); oy1 = max(0, int(np.floor(ys.min())))
        ox2 = min(w - 1, int(np.ceil(xs.max()))); oy2 = min(h - 1, int(np.ceil(ys.max())))
        return ox1, oy1, ox2, oy2

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
        img = self.load_document_image(path)

        async def _attempt(crop: bool) -> Dict:
            w, did_crop = await self._preprocess_for_display(img, crop=crop)
            rr = await self._locate_and_read(w)
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
        img = self.load_document_image(path)

        async def _attempt(crop: bool) -> Dict:
            work, did_crop, angle = await self._preprocess_ocr(img, crop=crop)
            rr = await self._locate_and_read(work)
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

        # ── Stage 4: direct masking ──
        # Force a raster extension: the masked output is always an image, but
        # the input may be a PDF/webp whose extension cv2.imwrite can't encode.
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(
            self.settings.output_dir, f"masked_{stem}.png")
        masked = self.create_masked_image(
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
