from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_range_01",
    "clip_01",
    "normalize_chroma_midpoint",
    "normalize_peak",
    "normalize_energy",
    "match_peak",
    "match_rms",
    "normalize_impulse",
    "normalize_to_reference",
]


def normalize_range_01(x: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """
    Global min/max normalization for value-range data (e.g., images).
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr
    xmin = float(arr.min())
    xmax = float(arr.max())
    if xmax - xmin < eps:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - xmin) / (xmax - xmin)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def clip_01(x: np.ndarray) -> np.ndarray:
    """
    Clip values to [0, 1] without rescaling.
    """
    arr = np.asarray(x, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def normalize_chroma_midpoint(
    ch: np.ndarray,
    *,
    center: float = 0.5,
    percentile: float = 99.0,
    target_radius: float = 0.25,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize chroma around a midpoint, scaling a percentile of |delta|.
    """
    arr = np.asarray(ch, dtype=np.float32) - float(center)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    scale = float(np.percentile(np.abs(arr), percentile))
    if scale > eps:
        arr = arr / scale * float(target_radius)
    arr = np.clip(arr + float(center), 0.0, 1.0)
    return arr.astype(np.float32)


def _peak_level(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _rms_level(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr**2)))


def normalize_peak(x: np.ndarray) -> np.ndarray:
    """
    Normalize a signal to unit peak.
    """
    arr = np.asarray(x, dtype=np.float32)
    peak = _peak_level(arr)
    if peak <= 0.0:
        return arr
    return (arr / peak).astype(np.float32)


def normalize_energy(x: np.ndarray) -> np.ndarray:
    """
    Normalize a signal to unit energy (L2 norm).
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr
    rms = _rms_level(arr)
    denom = rms * float(np.sqrt(arr.size))
    if denom <= 0.0:
        return arr
    return (arr / denom).astype(np.float32)


def match_peak(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Scale x so its peak matches the reference peak.
    """
    arr = np.asarray(x, dtype=np.float32)
    ref_peak = _peak_level(reference)
    out_peak = _peak_level(arr)
    if ref_peak > 0.0 and out_peak > 0.0:
        arr = arr * (ref_peak / out_peak)
    return arr.astype(np.float32)


def match_rms(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Scale x so its RMS matches the reference RMS.
    """
    arr = np.asarray(x, dtype=np.float32)
    ref_rms = _rms_level(reference)
    out_rms = _rms_level(arr)
    if ref_rms > 0.0 and out_rms > 0.0:
        arr = arr * (ref_rms / out_rms)
    return arr.astype(np.float32)


def normalize_impulse(x: np.ndarray, mode: str | None) -> np.ndarray:
    """
    Normalize an impulse with a simple mode switch.
    """
    arr = np.asarray(x, dtype=np.float32)
    if mode is None or mode == "none":
        return arr
    if mode == "peak":
        return normalize_peak(arr)
    if mode == "energy":
        return normalize_energy(arr)
    raise ValueError(f"Unknown impulse_norm: {mode}")


def normalize_to_reference(
    x: np.ndarray, reference: np.ndarray, mode: str | None
) -> np.ndarray:
    """
    Normalize x to match a reference by peak or RMS.
    """
    arr = np.asarray(x, dtype=np.float32)
    if mode is None or mode == "none":
        return arr
    if mode == "match_peak":
        return match_peak(arr, reference)
    if mode == "match_rms":
        return match_rms(arr, reference)
    raise ValueError(f"Unknown out_norm: {mode}")
