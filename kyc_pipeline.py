import os
import re
import cv2
import numpy as np
import pypdfium2 as pdfium

from ocr_engines import (
    PaddleOCREngine,
    SuryaOCREngine,
    TesseractOCREngine,
    select_ocr,
)
from preprocessing import (
    enhance_for_ocr,
    resize_for_ocr,
    rotate_image,
    auto_crop_document,
    deskew,
)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

def _invert_rotation(bbox, angle, rotated_w, rotated_h):
    """
    Map a bbox (x1,y1,x2,y2) from a *rotated* image back to the
    coordinate space of the image BEFORE that rotation was applied.

    rotate_image() uses cv2.rotate which applies these transforms:
        0°   → identity          (W_rot=W_orig, H_rot=H_orig)
        90°  → ROTATE_90_CW      (W_rot=H_orig, H_rot=W_orig)
        180° → ROTATE_180        (W_rot=W_orig, H_rot=H_orig)
        270° → ROTATE_90_CCW     (W_rot=H_orig, H_rot=W_orig)

    Inverse mapping (point in rotated space → point in original space):
        0°  : (x, y) → (x, y)
        90° : (x, y) → (y,  W_rot-1-x)   [original W = H_rot, H = W_rot]
        180°: (x, y) → (W_rot-1-x, H_rot-1-y)
        270°: (x, y) → (H_rot-1-y, x)    [original W = H_rot, H = W_rot]
    """
    x1, y1, x2, y2 = bbox
    W, H = rotated_w, rotated_h

    def inv(x, y):
        if angle == 0:
            return x, y
        if angle == 90:
            return y, W - 1 - x
        if angle == 180:
            return W - 1 - x, H - 1 - y
        if angle == 270:
            return H - 1 - y, x
        return x, y

    pts = [inv(x1, y1), inv(x2, y1), inv(x2, y2), inv(x1, y2)]
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _invert_autocrop(bbox, crop_x1, crop_y1):
    """Shift bbox back from cropped-image space to original-image space."""
    x1, y1, x2, y2 = bbox
    return x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class DocumentPipeline:

    def __init__(self):
        self.primary_ocr   = None
        self.fallback_ocr  = None
        self.tesseract_ocr = None      # used only for masking
        self.ocr_available = False

        self.patterns = {
            "PAN":             r"[A-Z]{5}[0-9]{4}[A-Z]",
            "AADHAAR":         r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)",
            "VOTER_ID":        r"[A-Z]{3}\d{7}",
            "PASSPORT":        r"[A-Z]{1,2}\d{6,7}",
            "DRIVING_LICENSE": r"[A-Z]{2}\d{2}\s?\d{11}",
        }

    # ── OCR bootstrap ────────────────────────────────────────────────────────

    def _ensure_ocr(self):
        try:
            if self.primary_ocr is None:
                self.primary_ocr = PaddleOCREngine()
            self.ocr_available = True
        except Exception:
            self.ocr_available = False
        try:
            if self.fallback_ocr is None:
                self.fallback_ocr = SuryaOCREngine()
        except Exception:
            pass
        try:
            if self.tesseract_ocr is None:
                self.tesseract_ocr = TesseractOCREngine()
        except Exception:
            pass

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_printable_name(self, doc):
        return {
            "AADHAAR":         "Aadhaar Card",
            "PAN":             "PAN Card",
            "PASSPORT":        "Passport",
            "VOTER_ID":        "Voter ID",
            "DRIVING_LICENSE": "Driving License",
            "UNKNOWN":         "Unknown",
        }.get(doc, "Unknown")

    def load_document_image(self, path):
        ext = path.lower().split(".")[-1]
        if ext == "pdf":
            pdf  = pdfium.PdfDocument(path)
            page = pdf[0]
            pil  = page.render(scale=4).to_pil()
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        return img

    # ── OCR with orientation detection ───────────────────────────────────────

    def extract_and_orient(self, path):
        """
        Load the document, find the best reading orientation via OCR, and return:
            text        – extracted text (upper-cased)
            orig_img    – the ORIGINAL image (no rotation applied)
            ocr_result  – OCR result whose word bboxes are in ROTATED space
            debug       – dict with 'angle', 'crop_x1', 'crop_y1', etc.

        The caller must use debug['angle'] and debug['crop_*'] to map
        word bboxes back to orig_img coordinates before drawing.
        """
        self._ensure_ocr()
        orig_img = self.load_document_image(path)

        # ── auto-crop + deskew (record crop offset for later inversion) ──
        crop, crop_dbg = auto_crop_document(orig_img)
        desk, desk_dbg = deskew(crop)

        # crop offset in original image coordinates
        crop_x1 = crop_dbg.get("bbox", (0, 0, 0, 0))[0] if crop_dbg.get("cropped") else 0
        crop_y1 = crop_dbg.get("bbox", (0, 0, 0, 0))[1] if crop_dbg.get("cropped") else 0

        best = None
        for angle in [0, 90, 180, 270]:
            rotated  = rotate_image(desk, angle)
            variants = enhance_for_ocr(rotated)
            res = select_ocr(
                variants["color"],
                primary=self.primary_ocr,
                fallback=self.fallback_ocr,
                confidence_threshold=0.82,
            )
            txt   = (res.text or "").upper()
            score = len(txt) + int((res.avg_confidence or 0) * 100)
            if best is None or score > best[0]:
                best = (score, txt, rotated, res, angle)

        _, text, rotated_img, ocr_result, best_angle = best

        debug = {
            "angle":   best_angle,
            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
            "rot_w":   rotated_img.shape[1],
            "rot_h":   rotated_img.shape[0],
            "crop":    crop_dbg,
            "deskew":  desk_dbg,
        }

        return text, orig_img, ocr_result, debug

    # ── Document classification ───────────────────────────────────────────────

    def classify_document(self, text):
        text   = text.upper()
        scores = {k: 0 for k in self.patterns}

        signatures = {
            "AADHAAR": [("UIDAI", 30), ("AADHAAR", 30)],
            "PAN":     [("INCOME TAX", 40), ("PAN", 30)],
            "VOTER_ID":[("ELECTION COMMISSION", 50), ("ELECTOR", 30),
                        ("PHOTO IDENTITY", 30), ("EPIC", 50)],
            "DRIVING_LICENSE": [("DL NO", 50), ("VALID TILL", 30), ("MCWG", 25),
                                ("LMV", 25), ("DRIVING LICENCE", 40),
                                ("DRIVING LICENSE", 40)],
            "PASSPORT": [("PASSPORT", 60), ("REPUBLIC OF INDIA", 50)],
        }

        for doc, vals in signatures.items():
            for keyword, weight in vals:
                if keyword in text:
                    scores[doc] += weight

        if re.search(r"[A-Z]{3}\d{7}", text):          scores["VOTER_ID"]        += 50
        if re.search(r"[A-Z]{2}\d{2}\s?\d{11}", text): scores["DRIVING_LICENSE"] += 60
        if re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text): scores["PAN"]             += 50
        if re.search(r"\d{4}\s?\d{4}\s?\d{4}", text):  scores["AADHAAR"]         += 50

        print("CLASS SCORES:", scores)
        mx = max(scores, key=scores.get)
        return mx if scores[mx] else "UNKNOWN"

    # ── ID extraction + display masking ──────────────────────────────────────

    def verify_and_extract(self, text, doc):
        matches = re.findall(self.patterns[doc], text)
        return matches[0] if matches else None

    def mask_id(self, val, doc):
        """Return a display-safe masked string (for UI only, not image)."""
        if doc == "AADHAAR":
            digits = re.sub(r"\D", "", val)
            return f"XXXX XXXX {digits[-4:]}"
        if doc == "PAN":
            return val[:2] + "XXXXX" + val[-3:]
        return "X" * (len(val) - 4) + val[-4:]

    # ── Redaction primitives ─────────────────────────────────────────────────

    def _apply_solid_redaction(self, img, x1, y1, x2, y2):
        """
        Solid black filled rectangle over the bbox.

        Padding scales with the box's MINOR dimension (its text thickness),
        not its length — so a tall vertical Voter-ID box is not blown up
        into a wide black slab, while small scans still get covered fully.
        """
        h, w = img.shape[:2]
        minor = max(1, min(x2 - x1, y2 - y1))
        pad   = max(5, minor // 4)
        cv2.rectangle(
            img,
            (max(0, x1 - pad), max(0, y1 - pad)),
            (min(w - 1, x2 + pad), min(h - 1, y2 + pad)),
            (0, 0, 0),
            -1,
        )

    def _is_date_like(self, text):
        t = text.strip()
        if re.match(r"^\d{1,2}[\/\-\.]\d{1,2}([\/\-\.]\d{2,4})?$", t):
            return True
        digits = re.sub(r"\D", "", t)
        if any(sep in t for sep in ["/", "-", "."]) and 6 <= len(digits) <= 8:
            return True
        return False

    # ── Per-doc word-match logic ─────────────────────────────────────────────

    def _should_mask_aadhaar(self, word_text, sensitive_digits, clear_last4):
        """
        Mask any word whose digit content is a substring of the first-8
        sensitive digits. Never mask a word that contains ONLY the last-4.
        """
        digits = re.sub(r"\D", "", word_text)
        if not digits or len(digits) < 2:
            return False
        if digits == clear_last4:          # only the visible group → keep
            return False
        if digits in sensitive_digits:     # substring of first-8 → mask
            return True
        full = sensitive_digits + clear_last4
        if len(digits) >= 8 and digits in full:  # all 12 fused → mask
            return True
        return False

    def _should_mask_pan(self, word_text, pan_clean):
        clean       = re.sub(r"[^A-Z0-9]", "", word_text.upper())
        digits_pan  = re.sub(r"\D", "", pan_clean)
        digits_word = re.sub(r"\D", "", clean)
        if not clean:
            return False
        if pan_clean in clean or clean in pan_clean:
            return True
        if len(digits_word) >= 3 and digits_pan and digits_pan in digits_word:
            return True
        return False

    def _should_mask_generic(self, word_text, id_clean, id_digits):
        """
        Strict match for PAN / Voter ID / Passport / Driving License.

        We require a substantial overlap with the ID so stray short numbers
        (and OCR garbage from wrong-orientation passes) are never masked.
        """
        clean_word  = re.sub(r"[^A-Z0-9]", "", word_text.upper())
        digits_word = re.sub(r"\D", "", word_text)
        if len(clean_word) < 4:
            return False
        # the full ID sits inside the word, or the word is a long
        # contiguous fragment of the ID
        if id_clean and (
            id_clean in clean_word
            or (len(clean_word) >= 5 and clean_word in id_clean)
        ):
            return True
        # strong digit-run overlap (handles tokens split off the ID)
        if (
            len(digits_word) >= 5
            and id_digits
            and (digits_word in id_digits or id_digits in digits_word)
        ):
            return True
        return False

    def _build_mask_context(self, id_string, doc_type):
        id_clean  = re.sub(r"[^A-Z0-9]", "", str(id_string).upper())
        id_digits = re.sub(r"\D", "", str(id_string))
        ctx = {"doc": doc_type, "id_clean": id_clean, "id_digits": id_digits}
        if doc_type == "AADHAAR":
            ctx["sensitive_digits"] = id_digits[:-4]
            ctx["clear_last4"]      = id_digits[-4:]
        return ctx

    def _word_should_be_masked(self, word_text, ctx):
        orig = word_text.strip()
        if not orig or self._is_date_like(orig):
            return False
        doc = ctx["doc"]
        if doc == "AADHAAR":
            return self._should_mask_aadhaar(
                orig, ctx["sensitive_digits"], ctx["clear_last4"])
        elif doc == "PAN":
            return self._should_mask_pan(orig, ctx["id_clean"])
        else:
            return self._should_mask_generic(
                orig, ctx["id_clean"], ctx["id_digits"])

    # ── Bbox merging ─────────────────────────────────────────────────────────

    def _merge_nearby_boxes(self, boxes, gap=20):
        """
        Merge bboxes that are on the same horizontal band and within `gap`
        pixels of each other horizontally.  Returns merged (x1,y1,x2,y2) list.

        This prevents Aadhaar's three separate 4-digit group boxes from
        leaving thin unredacted gaps between them.
        """
        if not boxes:
            return []

        rects = [(min(a,c), min(b,d), max(a,c), max(b,d)) for (a,b,c,d) in boxes]
        rects.sort(key=lambda r: (r[1], r[0]))

        merged = []
        cur = list(rects[0])

        for (x1, y1, x2, y2) in rects[1:]:
            cur_h     = max(1, cur[3] - cur[1])
            overlap_y = min(y2, cur[3]) - max(y1, cur[1])
            # Merge if rows share >40% height AND close horizontally
            if overlap_y >= 0.4 * cur_h and x1 <= cur[2] + gap:
                cur[0] = min(cur[0], x1)
                cur[1] = min(cur[1], y1)
                cur[2] = max(cur[2], x2)
                cur[3] = max(cur[3], y2)
            else:
                merged.append(tuple(cur))
                cur = [x1, y1, x2, y2]

        merged.append(tuple(cur))
        return merged

    # ── Fresh OCR passes for masking ─────────────────────────────────────────

    def _tesseract_words(self, img):
        """Tesseract word boxes for `img` as-is (coords in img's space)."""
        if self.tesseract_ocr is None:
            return []
        try:
            return self.tesseract_ocr.extract(img).words or []
        except Exception as e:
            print(f"[WARN] Tesseract masking pass failed: {e}")
            return []

    def _sane_boxes(self, boxes):
        """Drop only degenerate (near-zero area) boxes.

        Tall, thin boxes are kept on purpose: the Voter ID number is
        printed vertically on the card, so a tight vertical box around it
        is a correct redaction — not an artifact.
        """
        return [
            (x1, y1, x2, y2) for (x1, y1, x2, y2) in boxes
            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3
        ]

    def _collect_hits_all_orientations(self, base, ctx, words_fn):
        """
        Run `words_fn` at all four 90° rotations of `base`, keep the word
        boxes that should be redacted, and map every box back into `base`
        coordinate space.

        This catches every occurrence of the ID regardless of how it is
        printed — horizontal text, or the vertical EPIC number on a Voter
        ID — without having to guess a single global orientation.
        """
        boxes = []
        for angle in (0, 90, 180, 270):
            work    = rotate_image(base, angle)
            wh, ww  = work.shape[:2]
            for w in words_fn(work):
                if self._word_should_be_masked(w.text, ctx):
                    boxes.append(_invert_rotation(w.bbox, angle, ww, wh))
        return boxes

    # ── Main masking method ───────────────────────────────────────────────────

    def create_masked_image(
        self,
        orig_img,       # ORIGINAL image — never rotated, never cropped
        id_string,
        doc_type,
        output_path,
        ocr_result,     # kept for API compatibility, not used for bbox coords
        debug,          # kept for API compatibility
        debug_dir=None,
    ):
        """
        Redact every sensitive ID occurrence on the document.

        Strategy — orientation-robust, Tesseract-driven masking:

          1. KYC docs are scanned at any angle, and the Voter ID number is
             even printed VERTICALLY on the card. So we run Tesseract at
             all four 90° rotations and collect ID boxes from every one,
             mapping each box back to the original image space. Every
             occurrence is caught, whatever its orientation.
          2. Redaction uses Tesseract's tight WORD-level boxes — PaddleOCR's
             line-level boxes caused the old over-sized black bars.
          3. The saved image keeps the same orientation as the input.
        """
        if not id_string:
            return None

        ctx = self._build_mask_context(id_string, doc_type)

        # Normalise working size: downscale huge PDF renders (fast 4×
        # Tesseract passes) and upscale small scans (Tesseract reads tiny
        # text poorly). Coordinates stay consistent with the saved image.
        base = resize_for_ocr(orig_img, max_side=2200)
        bh, bw = base.shape[:2]
        if max(bh, bw) < 1400:
            s    = 1400.0 / max(bh, bw)
            base = cv2.resize(base, (int(bw * s), int(bh * s)),
                              interpolation=cv2.INTER_CUBIC)

        # ── 1. Collect ID boxes across every orientation (Tesseract) ───────
        hit_boxes = self._collect_hits_all_orientations(
            base, ctx, self._tesseract_words)

        # Fallback: Tesseract found nothing → try PaddleOCR the same way.
        if not hit_boxes and self.primary_ocr is not None:
            print("[INFO] Tesseract found no ID region — PaddleOCR fallback.")

            def _paddle_words(im):
                try:
                    return self.primary_ocr.extract(im).words or []
                except Exception:
                    return []

            hit_boxes = self._collect_hits_all_orientations(
                base, ctx, _paddle_words)

        # ── 2. Redact tight word boxes (merged so groups join cleanly) ─────
        masked        = base.copy()
        debug_overlay = base.copy() if debug_dir else None
        boxes_drawn   = 0

        hit_boxes = self._sane_boxes(hit_boxes)
        merged    = self._merge_nearby_boxes(hit_boxes, gap=20)

        for (x1, y1, x2, y2) in merged:
            if debug_overlay is not None:
                cv2.rectangle(debug_overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)
            self._apply_solid_redaction(masked, x1, y1, x2, y2)
            boxes_drawn += 1

        if boxes_drawn == 0:
            print("[WARN] No sensitive regions found. "
                  "Check OCR engine and ID extraction.")

        # ── Save ─────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if debug_dir and debug_overlay is not None:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(debug_dir,
                             f"mask_debug_{os.path.basename(output_path)}"),
                debug_overlay,
            )

        cv2.imwrite(output_path, masked)
        return output_path if boxes_drawn > 0 else None

    # ── Top-level pipeline entry point ───────────────────────────────────────

    def process_and_verify(self, path, intended):
        text, orig_img, ocr, debug = self.extract_and_orient(path)
        actual = self.classify_document(text)

        result = {
            "status":              "FAILED",
            "actual_type":         actual,
            "ocr_engine":          ocr.engine,
            "ocr_avg_confidence":  ocr.avg_confidence,
            "ocr_decision_reason": ocr.decision_reason,
            "ocr_debug":           debug,
            "extracted_text":      text,
        }

        if actual != intended:
            result["message"] = f"Expected {intended}, got {actual}"
            return result

        extracted = self.verify_and_extract(text, actual)
        if not extracted:
            result["message"] = "ID extraction failed"
            return result

        os.makedirs("sample-docs", exist_ok=True)

        masked = self.create_masked_image(
            orig_img=orig_img,
            id_string=extracted,
            doc_type=actual,
            output_path=f"sample-docs/masked_{os.path.basename(path)}",
            ocr_result=ocr,
            debug=debug,
            debug_dir="sample-docs/mask_debug",
        )

        result.update({
            "status":            "SUCCESS",
            "extracted_id":      extracted,
            "masked_id":         self.mask_id(extracted, actual),
            "masked_image_file": masked,
        })
        return result

    # ── Debug helper ─────────────────────────────────────────────────────────

    def build_preprocess_debug(self, path):
        img = self.load_document_image(path)
        crop, crop_dbg = auto_crop_document(img)
        desk, desk_dbg = deskew(crop)
        variants = enhance_for_ocr(desk)
        return {
            "base_color": desk,
            "variants":   variants,
            "crop":       crop_dbg,
            "deskew":     desk_dbg,
        }