"""Quick smoke test for the ONNX-backed primary OCR engine."""
import cv2

from ocr_engines import RapidOCREngine

ocr = RapidOCREngine(lang="en")

img = cv2.imread("sample/adhar-test.png")
if img is None:
    raise Exception("Image not found")

result = ocr.extract(img)
print("engine     :", result.engine)
print("avg_conf   :", result.avg_confidence)
print("text       :\n", result.text)
