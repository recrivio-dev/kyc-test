"""Numerical / signal-processing helpers shared across the layers.

These are deliberately pure functions (numpy in, scalar/array out) so they can
be unit tested in isolation and reused by Layer 1 (timing/entropy), Layer 2
(embedding similarity) and Layer 3 (motion / periodicity).
"""
from __future__ import annotations

import numpy as np


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a value into ``[lo, hi]``."""
    return float(max(lo, min(hi, value)))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Divide with a guard against division by zero."""
    return a / b if b not in (0, 0.0) else default


def median_abs_deviation(x: np.ndarray) -> float:
    """Robust scale estimate: median absolute deviation (scaled to ~std)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad * 1.4826)


def coefficient_of_variation(x: np.ndarray) -> float:
    """Ratio of std to mean (dimensionless dispersion)."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x) if x.size else 0.0
    if mean == 0:
        return 0.0
    return float(np.std(x) / abs(mean))


def shannon_entropy(values: np.ndarray, bins: int = 64) -> float:
    """Shannon entropy (bits) of a 1-D signal via histogram binning."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, density=False)
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return 0.0
    p = p[p > 0] / total
    return float(-np.sum(p * np.log2(p)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, in ``[-1, 1]``."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return ``v`` scaled to unit L2 norm."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + eps)


def linear_map(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Map ``x`` from range [x0,x1] to [y0,y1], clamped to the output range."""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def dominant_frequency(signal: np.ndarray, fps: float) -> tuple[float, float]:
    """Return ``(frequency_hz, normalized_power)`` of the dominant non-DC peak.

    Used by replay detection (periodic motion) and rPPG (pulse band). The
    normalized power is the peak's share of total AC spectral energy in [0,1].
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = signal.size
    if n < 4 or fps <= 0:
        return 0.0, 0.0
    signal = signal - np.mean(signal)
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(signal * window)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    # Ignore DC component.
    spectrum[0] = 0.0
    total = spectrum.sum()
    if total <= 0:
        return 0.0, 0.0
    peak = int(np.argmax(spectrum))
    return float(freqs[peak]), float(spectrum[peak] / total)


def periodicity_score(signal: np.ndarray) -> float:
    """Strength of periodic structure via normalized autocorrelation peak.

    Returns a value in ``[0, 1]``; high values indicate a strongly repeating
    (looping / replayed) signal.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = signal.size
    if n < 8:
        return 0.0
    signal = signal - np.mean(signal)
    var = np.dot(signal, signal)
    if var <= 0:
        return 0.0
    ac = np.correlate(signal, signal, mode="full")[n - 1:]
    ac = ac / var
    # Skip lag-0; look for the strongest secondary peak.
    if ac.size <= 2:
        return 0.0
    return float(np.clip(np.max(ac[2:]), 0.0, 1.0))
