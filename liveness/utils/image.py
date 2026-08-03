"""Image / frame decoding and manipulation helpers.

All functions operate on / return ``numpy`` arrays in **BGR** order (OpenCV
convention) unless stated otherwise. These helpers are deliberately dependency
light so they can be reused by every layer.
"""
from __future__ import annotations

import base64
from typing import Iterable

import cv2
import numpy as np


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (PNG/JPEG/...) into a BGR ndarray."""
    if not data:
        raise ValueError("Empty image payload")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes (unsupported format?)")
    return img


def decode_base64_image(b64: str) -> np.ndarray:
    """Decode a base64 (optionally data-URI prefixed) image into BGR ndarray."""
    if "," in b64 and b64.strip().lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return decode_image_bytes(base64.b64decode(b64))


def encode_image_b64(img: np.ndarray, ext: str = ".jpg") -> str:
    """Encode a BGR ndarray to a base64 string (no data-URI prefix)."""
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_video_bytes(data: bytes, max_frames: int = 90) -> list[np.ndarray]:
    """Decode a short video clip into a list of BGR frames.

    Writes to a temporary file because OpenCV's ``VideoCapture`` cannot read
    from an in-memory buffer reliably across platforms.
    """
    import os
    import tempfile

    if not data:
        raise ValueError("Empty video payload")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)

        # Sample evenly across the WHOLE clip rather than truncating to the
        # first N frames — otherwise a blink / head-turn performed late in the
        # clip would be cut off. Fall back to sequential reads when the frame
        # count is not reported reliably by the codec.
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[np.ndarray] = []
        if total > max_frames:
            wanted = set(np.linspace(0, total - 1, max_frames).round().astype(int).tolist())
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in wanted:
                    frames.append(frame)
                idx += 1
                if len(frames) >= max_frames:
                    break
        else:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)

        cap.release()
        if not frames:
            raise ValueError("No frames decoded from video")
        return frames
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def white_balance(img: np.ndarray) -> np.ndarray:
    """Gray-world white balance — removes colour casts (e.g. the red tint on a
    laminated / faded ID-card photo) that degrade face embeddings."""
    result = img.astype(np.float32)
    means = result.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = gray / (means + 1e-6)
    return np.clip(result * scale, 0, 255).astype(np.uint8)


def enhance_for_recognition(img: np.ndarray, upscale_to: int = 320) -> np.ndarray:
    """Restore a degraded face photo for more reliable embeddings.

    Applies gray-world white balance (kills colour casts), CLAHE contrast
    equalisation on luminance, mild denoise + unsharp masking, and upscales tiny
    crops. Designed to help faded / low-contrast / small ID-card photos without
    materially altering already-good captures.
    """
    out = white_balance(img)

    # CLAHE on the L channel of LAB for local contrast.
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Upscale small images so the detector/recogniser sees more detail.
    h, w = out.shape[:2]
    if max(h, w) < upscale_to:
        scale = upscale_to / float(max(h, w))
        out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Gentle denoise + unsharp mask to recover edges.
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
    return out


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB (MediaPipe expects RGB)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_max(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Resize so the longest side equals ``max_side`` (keeps aspect ratio)."""
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def crop_bbox(img: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.0) -> np.ndarray:
    """Crop ``img`` to ``bbox`` (x1, y1, x2, y2) with optional fractional padding."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def sample_frames(frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    """Uniformly sub-sample a list of frames down to ``target`` frames."""
    n = len(frames)
    if n <= target:
        return frames
    idx = np.linspace(0, n - 1, target).round().astype(int)
    return [frames[i] for i in idx]


def stack_grid(images: Iterable[np.ndarray], cols: int = 4, cell: int = 96) -> np.ndarray:
    """Build a contact-sheet grid thumbnail from images (for reports/debug)."""
    imgs = [cv2.resize(im, (cell, cell)) for im in images]
    if not imgs:
        return np.zeros((cell, cell, 3), dtype=np.uint8)
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = im
    return canvas
