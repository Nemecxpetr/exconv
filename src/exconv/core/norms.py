# src/exconv/core/norms.py
"""Normalization utilities for audio signals."""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "rms_normalize",
    "peak_normalize",
    "apply_normalize",
]


def rms_normalize(x: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    RMS-normalize signal to a target RMS.

    Parameters
    ----------
    x : ndarray, shape (N,) or (N, C)
        Input signal.
    target_rms : float, default=0.1
        Desired RMS across all samples and channels.

    Returns
    -------
    y : ndarray
        Scaled signal. If input RMS is 0, returns x unchanged.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x_arr**2)) if x_arr.size > 0 else 0.0
    if rms <= 0.0:
        return x_arr
    scale = float(target_rms) / float(rms)
    return x_arr * scale


def peak_normalize(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """
    Peak-normalize signal to a target absolute peak.

    Parameters
    ----------
    x : ndarray, shape (N,) or (N, C)
        Input signal.
    peak : float, default=0.99
        Desired maximum absolute value.

    Returns
    -------
    y : ndarray
        Scaled signal. If max abs is 0, returns x unchanged.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    m = float(np.max(np.abs(x_arr))) if x_arr.size > 0 else 0.0
    if m <= 0.0:
        return x_arr
    scale = float(peak) / m
    return x_arr * scale


def apply_normalize(x: np.ndarray, mode: Optional[str]) -> np.ndarray:
    """
    Dispatch normalization by mode.

    Parameters
    ----------
    x : ndarray
        Input signal.
    mode : {"rms","peak",None,"none"}
        Normalization mode. None or "none" → no-op.

    Returns
    -------
    y : ndarray
        Normalized (or unchanged) signal.

    Raises
    ------
    ValueError
        For unknown mode strings.
    """
    if mode is None:
        return np.asarray(x, dtype=np.float64)

    if isinstance(mode, str):
        m = mode.strip().lower()
        if m in ("none", ""):
            return np.asarray(x, dtype=np.float64)
        if m == "rms":
            return rms_normalize(x)
        if m == "peak":
            return peak_normalize(x)
        raise ValueError(f"Unknown normalization mode {mode!r}")

    raise ValueError(f"Normalization mode must be str or None, got {type(mode)}")
