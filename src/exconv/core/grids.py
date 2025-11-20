# src/exconv/core/grids.py
"""2D frequency and radial grids, and 1D window functions."""
from __future__ import annotations

from typing import Tuple
import numpy as np

__all__ = [
    "freq_grid_2d",
    "radial_grid_2d",
    "hann",
    "tukey",
]

def freq_grid_2d(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return 2D integer frequency coordinates (k1, k2) using the
    fftshift convention: zero at center, negative frequencies on the left/top.

    Parameters
    ----------
    H, W : int
        Spatial size.

    Returns
    -------
    k1, k2 : ndarray, shape (H, W)
        Row (vertical) and column (horizontal) frequency indices.
        Values run from -floor(N/2) ... ceil(N/2)-1 along each axis.
    """
    k1 = np.arange(-(H // 2), -(H // 2) + H, dtype=int)
    k2 = np.arange(-(W // 2), -(W // 2) + W, dtype=int)
    return np.meshgrid(k1, k2, indexing="ij")


def radial_grid_2d(H: int, W: int, norm: str = "unit") -> np.ndarray:
    """
    Radial coordinate ρ in [0, 1] with fftshift convention (zero at center).

    For normalization "unit", distances are divided by the farthest reachable
    frequency bin corner, which is at (floor(H/2), floor(W/2)).
    """
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")

    k1, k2 = freq_grid_2d(H, W)
    r = np.hypot(k1.astype(float), k2.astype(float))  # sqrt(k1^2 + k2^2)

    if norm == "unit":
        rmax = float(np.hypot(H // 2, W // 2))  # correct for odd/even sizes
        return r / (1.0 if rmax == 0.0 else rmax)

    raise ValueError(f"Unknown norm {norm!r}")



def hann(N: int) -> np.ndarray:
    """
    1D Hann window (periodic variant aligning with FFT usage).

    Parameters
    ----------
    N : int
        Length.

    Returns
    -------
    w : ndarray, shape (N,)
    """
    if N <= 1:
        return np.ones(max(1, N), dtype=float)
    # Use np.hanning (periodic Hann), which is equivalent to the common FFT window
    return np.hanning(N)


def tukey(N: int, alpha: float = 0.5) -> np.ndarray:
    """
    1D Tukey (tapered cosine) window. Implemented without SciPy dependency.

    Parameters
    ----------
    N : int
        Length.
    alpha : float
        Fraction of window inside the cosine taper, in [0,1].

    Returns
    -------
    w : ndarray, shape (N,)
    """
    if N <= 1:
        return np.ones(max(1, N), dtype=float)
    if alpha <= 0:
        return np.ones(N, dtype=float)
    if alpha >= 1:
        return np.hanning(N)

    w = np.ones(N, dtype=float)
    # cosine taper lengths
    edge = int(np.floor(alpha * (N - 1) / 2.0))
    n = np.arange(N, dtype=float)

    # Rising edge
    if edge > 0:
        i = np.arange(edge)
        w[i] = 0.5 * (1 + np.cos(np.pi * ((2 * i) / (alpha * (N - 1)) - 1)))
        # Falling edge
        j = np.arange(N - edge, N)
        ii = np.arange(edge - 1, -1, -1)  # symmetric
        w[j] = w[ii]
    return w
