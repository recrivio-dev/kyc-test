from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Any
import numpy as np


@dataclass
class OCRWordBox:
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: Optional[float] = None


@dataclass
class OCRResult:
    text: str
    words: List[OCRWordBox]
    avg_confidence: Optional[float]
    engine: str
    decision_reason: Optional[str] = None


############################################
# FALLBACK HEURISTICS
############################################

def should_fallback_to_surya(
        txt: str,
        conf: Optional[float],
        threshold: float = 0.82
):

    txt = txt.strip()

    if len(txt) < 10:
        return "too_little_text"

    if conf is not None and conf < threshold:
        return "low_confidence"

    non_ascii = sum(ord(c) > 127 for c in txt)

    if len(txt) > 0:

        ratio = non_ascii / len(txt)

        if ratio > 0.10:
            return "multilingual"

    return None


############################################
# BASE ENGINE
############################################

class OCREngine:

    def extract(self, image):

        raise NotImplementedError


############################################
# PADDLE OCR
############################################

class PaddleOCREngine(OCREngine):

    def __init__(self, lang="en"):

        from paddleocr import PaddleOCR

        self.name = "paddleocr"

        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang=lang
        )

    def extract(self, image):

        result = self.ocr.predict(image)

        text_lines = []
        confs = []

        words = []

        for page in result:

            texts = page.get("rec_texts", [])

            scores = page.get("rec_scores", [])

            boxes = page.get("rec_boxes", [])

            for txt, conf, box in zip(
                    texts,
                    scores,
                    boxes
            ):

                try:

                    x1,y1,x2,y2 = map(
                        int,
                        box
                    )

                except:

                    x1=y1=x2=y2=0

                words.append(

                    OCRWordBox(

                        text=txt,

                        bbox=(
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        confidence=float(conf)
                    )
                )

                text_lines.append(txt)

                confs.append(float(conf))

        return OCRResult(

            text="\n".join(text_lines),

            words=words,

            avg_confidence=(
                sum(confs)/len(confs)
                if confs else None
            ),

            engine="paddleocr"
        )


############################################
# SURYA OCR
############################################

class SuryaOCREngine(OCREngine):

    def __init__(self, lang="en"):

        self.name = "surya"

        self.lang = lang

        try:

            from surya.ocr import run_ocr

            self.run = run_ocr

        except:

            self.run = None

    def extract(self,image):

        if self.run is None:

            raise RuntimeError(
                "Surya not installed"
            )

        out = self.run(
            [image],
            languages=[self.lang]
        )

        page = out[0]

        text = str(page)

        return OCRResult(

            text=text,

            words=[],

            avg_confidence=None,

            engine="surya"
        )


############################################
# TESSERACT OCR  (used for masking only)
############################################

class TesseractOCREngine(OCREngine):
    """
    Tesseract gives tight, word-level bounding boxes which makes it a far
    better fit for redaction/masking than PaddleOCR's line-level boxes.
    Used exclusively by the masking stage — extraction still uses Paddle.
    """

    def __init__(self, lang="eng"):

        import pytesseract

        self.pt   = pytesseract
        self.name = "tesseract"
        self.lang = lang

    def extract(self, image):

        import cv2

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if getattr(image, "ndim", 2) == 3
            else image
        )

        data = self.pt.image_to_data(
            gray,
            lang=self.lang,
            output_type=self.pt.Output.DICT,
        )

        words, texts, confs = [], [], []

        for i in range(len(data["text"])):

            txt  = (data["text"][i] or "").strip()
            conf = float(data["conf"][i])

            if not txt or conf < 0:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            words.append(
                OCRWordBox(
                    text=txt,
                    bbox=(x, y, x + w, y + h),
                    confidence=conf / 100.0,
                )
            )
            texts.append(txt)
            confs.append(conf / 100.0)

        return OCRResult(
            text=" ".join(texts),
            words=words,
            avg_confidence=(sum(confs) / len(confs) if confs else None),
            engine="tesseract",
        )


############################################
# SELECT OCR
############################################

def select_ocr(

        image,

        primary,

        fallback=None,

        confidence_threshold=0.82

):

    primary_res = primary.extract(image)

    reason = should_fallback_to_surya(

        primary_res.text,

        primary_res.avg_confidence,

        confidence_threshold

    )

    primary_res.decision_reason = (

        reason or

        "primary_ok"

    )

    if fallback and reason:

        try:

            fb = fallback.extract(image)

            fb.decision_reason = (

                f"fallback:{reason}"

            )

            return fb

        except:

            pass

    return primary_res