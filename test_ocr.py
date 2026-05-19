from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="en"
)

img = cv2.imread("sample/adhar-test.png")

if img is None:
    raise Exception("Image not found")

result = ocr.predict(img)

print(result)