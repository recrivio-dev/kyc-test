import cv2
import numpy as np


def resize_for_ocr(img: np.ndarray, max_side: int = 2000) -> np.ndarray:
    """Resize image to a reasonable size for OCR while keeping aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def enhance_for_ocr(img_bgr: np.ndarray) -> dict:
    """Return multiple enhanced variants for OCR.

    Output keys:
      - color: resized color image
      - gray: grayscale
      - bin: binarized (adaptive)

    Notes:
      We keep preprocessing conservative so it helps both PaddleOCR and Surya.
    """
    color = resize_for_ocr(img_bgr)

    # Denoise (fast and usually helpful)
    den = cv2.fastNlMeansDenoisingColored(color, None, 10, 10, 7, 21)

    gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)

    # Contrast boost via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_c = clahe.apply(gray)

    # Mild sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(gray_c, -1, kernel)

    # Adaptive threshold helps classical OCR; PaddleOCR generally prefers grayscale/RGB,
    # but we keep it for fallback attempts.
    bin_img = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )

    return {"color": color, "gray": sharp, "bin": bin_img}


def auto_crop_document(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Attempt to find the document contour and crop to it.

    Returns (cropped_image, debug_info). If no good contour is found, returns original.
    """
    img = img_bgr
    h, w = img.shape[:2]
    debug = {"cropped": False, "contour_found": False}

    # Work on a downsized image for contour detection.
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, debug

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.15 * (small.shape[0] * small.shape[1]):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        x, y, ww, hh = cv2.boundingRect(approx)
        debug.update({"contour_found": True, "bbox_small": (int(x), int(y), int(ww), int(hh))})

        # Scale bbox back to original size.
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + ww) / scale)
        y2 = int((y + hh) / scale)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) * (y2 - y1) < 0.2 * (w * h):
            continue

        cropped = img_bgr[y1:y2, x1:x2].copy()
        debug.update({"cropped": True, "bbox": (x1, y1, x2, y2)})
        return cropped, debug

    return img_bgr, debug


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate skew angle in degrees using Hough lines on edges."""
    # edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=100, minLineLength=gray.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Normalize near-horizontal lines
        if angle < -45:
            angle += 90
        if angle > 45:
            angle -= 90
        angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(img_bgr: np.ndarray, max_abs_angle: float = 10.0) -> tuple[np.ndarray, dict]:
    """Deskew image by estimating dominant text/document line angle.

    Returns (deskewed_img, debug_info). If estimated angle is small, returns original.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray)
    debug = {"deskewed": False, "angle_deg": angle}
    if abs(angle) < 0.5 or abs(angle) > max_abs_angle:
        return img_bgr, debug

    (h, w) = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    debug["deskewed"] = True
    return rotated, debug


def rotate_image(img: np.ndarray, angle_deg: int) -> np.ndarray:
    if angle_deg % 360 == 0:
        return img
    if angle_deg % 360 == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle_deg % 360 == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle_deg % 360 == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # generic rotate (not used by default)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
