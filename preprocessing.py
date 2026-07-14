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


def auto_crop_document(
    img_bgr: np.ndarray,
    min_keep_ratio: float = 0.70,
    margin_edge_thresh: float = 0.015,
    pad_ratio: float = 0.012,
) -> tuple[np.ndarray, dict]:
    """Attempt to find the document contour and crop to it.

    A crop is only accepted when it is *safe* — i.e. it does not throw away
    real document content. Canny on a light card over a light background
    routinely latches onto a partial inner contour, and a crop to that box
    silently amputates part of the card (the bottom strip carrying the ID
    number, a slogan band, etc.). Two guards prevent that:

      * keep-area guard — the padded crop must retain at least
        `min_keep_ratio` of the original image area; a contour covering
        only part of the card is rejected outright.
      * margin-content guard — every strip the crop would discard is
        checked for edge density. A blank photo border has almost no
        edges; a discarded strip whose edge fraction exceeds
        `margin_edge_thresh` still carries text/graphics, meaning the
        contour is cutting *into* the document — so the crop is rejected.

    The detected box is also padded outward by `pad_ratio` so a slightly
    tight contour never shaves off characters.

    Returns (cropped_or_original, debug_info). When no contour passes the
    guards the original image is returned unchanged — over-cropping
    destroys OCR input, so falling back to the full frame is the safe
    failure mode.
    """
    img = img_bgr
    h, w = img.shape[:2]
    debug = {"cropped": False, "contour_found": False}

    # Work on a downsized image for contour detection.
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Close gaps for contour finding. Keep the *raw* `edges` map intact —
    # it is what the margin-content guard measures.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, debug

    def _strip_edge_fraction(y1: int, y2: int, x1: int, x2: int) -> float:
        """Edge density of a discarded strip in the raw-edge map.

        Returns 0 for strips too thin to matter (the padding ring) so a
        few-pixel border never trips the content guard."""
        if y2 <= y1 or x2 <= x1:
            return 0.0
        strip = edges[y1:y2, x1:x2]
        if strip.size < 0.01 * sh * sw:
            return 0.0
        return float(np.count_nonzero(strip)) / float(strip.size)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    pad_x = int(round(sw * pad_ratio))
    pad_y = int(round(sh * pad_ratio))

    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.15 * (sh * sw):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        x, y, ww, hh = cv2.boundingRect(approx)
        # Pad the box outward so a tight contour never clips characters.
        sx1 = max(0, x - pad_x)
        sy1 = max(0, y - pad_y)
        sx2 = min(sw, x + ww + pad_x)
        sy2 = min(sh, y + hh + pad_y)

        keep_ratio = ((sx2 - sx1) * (sy2 - sy1)) / float(sh * sw)
        margins = {
            "top":    _strip_edge_fraction(0,   sy1, 0,   sw),
            "bottom": _strip_edge_fraction(sy2, sh,  0,   sw),
            "left":   _strip_edge_fraction(sy1, sy2, 0,   sx1),
            "right":  _strip_edge_fraction(sy1, sy2, sx2, sw),
        }
        worst_margin = max(margins.values())
        debug.update({
            "contour_found": True,
            "bbox_small": (int(x), int(y), int(ww), int(hh)),
            "keep_ratio": round(keep_ratio, 3),
            "margin_edges": {k: round(v, 4) for k, v in margins.items()},
        })

        # Reject crops that drop too much area or cut into real content.
        if keep_ratio < min_keep_ratio:
            debug["reject_reason"] = (
                f"keep_ratio {keep_ratio:.2f} < {min_keep_ratio}")
            continue
        if worst_margin > margin_edge_thresh:
            debug["reject_reason"] = (
                f"discarded margin still has content "
                f"(edge fraction {worst_margin:.3f} > {margin_edge_thresh})")
            continue

        # Scale the accepted box back to original resolution.
        x1 = max(0, int(sx1 / scale))
        y1 = max(0, int(sy1 / scale))
        x2 = min(w, int(sx2 / scale))
        y2 = min(h, int(sy2 / scale))

        cropped = img_bgr[y1:y2, x1:x2].copy()
        debug.update({"cropped": True, "bbox": (x1, y1, x2, y2)})
        return cropped, debug

    debug.setdefault("reject_reason", "no contour passed the safety guards")
    return img_bgr, debug


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]          # tl has the smallest x+y
    rect[2] = pts[np.argmax(s)]          # br has the largest  x+y
    d = np.diff(pts, axis=1).ravel()     # y - x
    rect[1] = pts[np.argmin(d)]          # tr
    rect[3] = pts[np.argmax(d)]          # bl
    return rect


def four_point_transform(img: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Warp the quadrilateral `quad` (4×2, original-image coords) to a
    straight, axis-aligned rectangle. This crops to the document AND removes
    its rotation/perspective in one step."""
    rect = _order_quad(quad)
    tl, tr, br, bl = rect
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if width < 10 or height < 10:
        return None
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (width, height))


def _texture_energy(gray: np.ndarray) -> np.ndarray:
    """Per-pixel edge energy (0/1). A document is densely printed; a desk,
    hand or soft shadow is smooth — so texture separates them even when their
    colours are nearly identical."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return (mag > 30).astype(np.uint8)


def _candidate_masks(small: np.ndarray) -> dict:
    """Three independent document masks. None is reliable alone:

      * colordiff — background colour estimated from the border ring. Great on
        a contrasting background, but a soft shadow also 'differs' and gets
        merged into the card.
      * texture   — Sobel energy, morphologically closed. Finds the printed
        body; immune to shadows. The workhorse for a white card on a white desk.
      * canny     — classic edges. Sharp on a clear border, but latches onto
        strong *inner* rectangles (a framed paragraph) and then amputates.
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    out = {}

    border = np.concatenate([small[0, :], small[-1, :],
                             small[:, 0], small[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.abs(small.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    out["colordiff"] = cv2.morphologyEx(
        (diff > 40).astype(np.uint8) * 255, cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8), iterations=2)

    out["texture"] = cv2.morphologyEx(
        _texture_energy(gray) * 255, cv2.MORPH_CLOSE,
        np.ones((25, 25), np.uint8), iterations=2)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    out["canny"] = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
    return out


def _quad_score(rect, tex: np.ndarray) -> float:
    """How document-like is this rect? Texture density inside minus outside.

    A rect snapped to the real card is dense inside and sits on a smooth
    background; one that swallowed a shadow dilutes its inside density."""
    h, w = tex.shape[:2]
    poly = cv2.boxPoints(rect).astype(np.int32)
    inside = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inside, poly, 1)
    n_in = int(inside.sum())
    n_out = h * w - n_in
    if n_in < 0.02 * h * w:
        return -1.0
    d_in = float((tex * inside).sum()) / n_in
    d_out = float((tex * (1 - inside)).sum()) / n_out if n_out > 0 else 0.0
    return d_in - d_out


def detect_document_quad(img: np.ndarray,
                         min_area_ratio: float = 0.08,
                         aspect_range: tuple = (1.10, 2.60),
                         pad_ratio: float = 0.015):
    """Locate the document as a rotated quadrilateral in original-image coords.

    Builds several candidate masks, keeps every candidate whose rect covers at
    least `min_area_ratio` of the frame and whose long/short ratio looks like a
    real document (ID card ~1.58, e-card page ~1.21), then picks the one that
    scores best on texture-inside-vs-outside. Returns (quad_4x2 | None, debug).
    """
    h, w = img.shape[:2]
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(img, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    tex = _texture_energy(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    frame_area = float(small.shape[0] * small.shape[1])

    best = None
    rejected = {}
    for name, mask in _candidate_masks(small).items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            rejected[name] = "no contour"
            continue
        cnt = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(cnt) / frame_area
        if area_ratio < min_area_ratio:
            rejected[name] = f"area {area_ratio:.2f}"
            continue
        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), angle = rect
        if min(rw, rh) < 1:
            rejected[name] = "degenerate"
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if not (aspect_range[0] <= aspect <= aspect_range[1]):
            rejected[name] = f"aspect {aspect:.2f}"
            continue
        score = _quad_score(rect, tex)
        if best is None or score > best[0]:
            best = (score, name, rect, area_ratio, aspect, angle)

    if best is None:
        return None, {"reason": "no candidate passed", "rejected": rejected}

    score, name, rect, area_ratio, aspect, angle = best
    (cx, cy), (rw, rh), _ = rect
    # Pad outward so a tight contour never shaves off border characters.
    quad = cv2.boxPoints(((cx, cy),
                          (rw * (1 + 2 * pad_ratio), rh * (1 + 2 * pad_ratio)),
                          angle)) / scale
    skew = min(abs(angle), abs(abs(angle) - 90.0))
    return quad, {"source": name, "score": round(score, 3),
                  "area_ratio": round(area_ratio, 3),
                  "aspect": round(aspect, 3),
                  "angle": round(angle, 2), "skew": round(skew, 2)}


def crop_document(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Crop to the document and remove its rotation/perspective.

    Strategy: detect the document quad on a background-difference mask and warp
    it to a straight rectangle. When the document already fills the frame
    squarely there is nothing to gain, so the original is returned untouched
    (avoiding a needless resample). If no quad is trustworthy we fall back to
    the conservative `auto_crop_document`, and finally to the original — an
    over-crop destroys the document, so the full frame is the safe failure."""
    h, w = img.shape[:2]
    quad, dbg = detect_document_quad(img)

    if quad is not None:
        quad_area = cv2.contourArea(quad.astype(np.float32)) / float(w * h)
        # Already a tight, square-on document → don't resample it.
        if quad_area >= 0.985 and dbg.get("skew", 90) < 1.0:
            return img, {**dbg, "cropped": False, "method": "already_tight"}
        warped = four_point_transform(img, quad)
        if warped is not None:
            return warped, {**dbg, "cropped": True, "method": "quad_warp"}
        dbg["reason"] = "warp failed"

    cropped, adbg = auto_crop_document(img)
    return cropped, {**adbg, "method": "auto_crop_fallback",
                     "quad_reject": dbg.get("reason")}


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate skew angle in degrees using Hough lines on edges."""
    # edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=100, minLineLength=gray.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return 0.0

    angles = []
    # HoughLinesP's return shape varies across OpenCV builds — (N,1,4) on some,
    # (N,4) on others (e.g. the opencv-python rapidocr floats in). `lines[:,0]`
    # only works for (N,1,4) and 500s the whole OCR path otherwise; reshape is
    # shape-agnostic. (Root cause of the July-2026 rollback.)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
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
