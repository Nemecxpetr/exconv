# src/exconv/conv2d/kernels.py
"""2D convolution kernels: Gaussian, Laplacian, Gabor, separable conv."""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

__all__ = [
    "gaussian_1d",
    "gaussian_2d",
    "gaussian_separable",
    "laplacian_3x3",
    "gabor_kernel",
    "separable_conv2d",
]


ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Gaussian kernels
# ---------------------------------------------------------------------------

def gaussian_1d(
    sigma: float,
    radius: Optional[int] = None,
    truncate: float = 3.0,
    normalize: bool = True,
) -> ArrayLike:
    """
    1D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    radius : int | None
        Number of samples on each side of zero. If None, computed as
        round(truncate * sigma).
    truncate : float
        Truncation in standard deviations if radius is None.
    normalize : bool
        If True, kernel sums to 1.

    Returns
    -------
    w : ndarray, shape (2*radius+1,)
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if radius is None:
        radius = int(round(truncate * sigma))
        radius = max(1, radius)

    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-0.5 * (x / sigma) ** 2)

    if normalize:
        g_sum = g.sum()
        if g_sum != 0:
            g /= g_sum
    return g


def gaussian_2d(
    sigma: float,
    radius: Optional[int] = None,
    truncate: float = 3.0,
    normalize: bool = True,
) -> ArrayLike:
    """
    Isotropic 2D Gaussian kernel constructed as outer product of 1D kernels.

    Parameters
    ----------
    sigma, radius, truncate, normalize
        See `gaussian_1d`.

    Returns
    -------
    k : ndarray, shape (K, K)
    """
    g = gaussian_1d(sigma, radius=radius, truncate=truncate, normalize=False)
    k = np.outer(g, g)
    if normalize:
        s = k.sum()
        if s != 0:
            k /= s
    return k


def gaussian_separable(
    sigma: float,
    radius: Optional[int] = None,
    truncate: float = 3.0,
    normalize: bool = True,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Separable representation of a 2D isotropic Gaussian:
    returns (k_row, k_col), where both are 1D Gaussians.

    Useful for spatial reference convolutions.
    """
    g = gaussian_1d(sigma, radius=radius, truncate=truncate, normalize=normalize)
    return g, g


# ---------------------------------------------------------------------------
# Laplacian kernel
# ---------------------------------------------------------------------------

def laplacian_3x3(center_weight: float = -4.0, eight_connected: bool = False) -> ArrayLike:
    """
    Classic 3x3 Laplacian kernel.

    Parameters
    ----------
    center_weight : float
        Center value (usually -4 or -8).
    eight_connected : bool
        If True, use 8-connected Laplacian; otherwise 4-connected.

    Returns
    -------
    k : ndarray, shape (3, 3)
    """
    if eight_connected:
        k = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, center_weight, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
    else:
        k = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, center_weight, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    return k


# ---------------------------------------------------------------------------
# Gabor kernel
# ---------------------------------------------------------------------------

def gabor_kernel(
    ksize: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float = 0.5,
    psi: float = 0.0,
    normalize: bool = False,
) -> ArrayLike:
    """
    2D Gabor kernel.

    Parameters
    ----------
    ksize : int
        Kernel size (ksize x ksize). Must be positive odd for symmetry.
    sigma : float
        Gaussian envelope standard deviation.
    theta : float
        Orientation in radians.
    lambd : float
        Wavelength of the cosine factor.
    gamma : float
        Spatial aspect ratio.
    psi : float
        Phase offset.
    normalize : bool
        If True, normalize absolute sum to 1.

    Returns
    -------
    g : ndarray, shape (ksize, ksize)
    """
    if ksize <= 0:
        raise ValueError("ksize must be positive")
    if lambd <= 0:
        raise ValueError("lambd must be positive")

    half = ksize // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]

    # Rotation
    x_prime = x * np.cos(theta) + y * np.sin(theta)
    y_prime = -x * np.sin(theta) + y * np.cos(theta)

    gaussian = np.exp(
        -0.5 * ((x_prime ** 2 + (gamma ** 2) * y_prime ** 2) / (sigma ** 2))
    )
    sinusoid = np.cos(2.0 * np.pi * x_prime / lambd + psi)

    g = gaussian * sinusoid

    if normalize:
        s = np.sum(np.abs(g))
        if s != 0:
            g /= s
    return g


# ---------------------------------------------------------------------------
# Separable spatial convolution (reference)
# ---------------------------------------------------------------------------

def _same_center_slice_1d(y_full: ArrayLike, N: int) -> ArrayLike:
    """
    Center crop (or pad) a 1D full convolution result back to length N.

    This matches the 'same-center' convention used in the 1D audio conv
    and 2D FFT-based conv.
    """
    L = y_full.shape[0]
    if N >= L:
        out = np.zeros(N, dtype=y_full.dtype)
        start = (N - L) // 2
        out[start : start + L] = y_full
        return out
    start = (L - N) // 2
    return y_full[start : start + N]


def _same_center_conv1d(x: ArrayLike, k: ArrayLike) -> ArrayLike:
    """
    1D reference convolution: full linear conv + center crop to len(x).
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    y_full = np.convolve(x, k, mode="full")
    return _same_center_slice_1d(y_full, x.shape[0])


def separable_conv2d(
    img: np.ndarray,
    k_row: np.ndarray,
    k_col: np.ndarray | None = None,
    mode: str = "same",
) -> np.ndarray:
    """
    Apply separable 2D convolution using 1D kernels along rows and columns.

    Parameters
    ----------
    img : np.ndarray
        2D or 3D image.
    k_row : np.ndarray
        1D kernel applied along rows (axis=0).
    k_col : np.ndarray or None
        1D kernel applied along columns (axis=1). If None, use k_row.
    mode : {"same"}
        Currently only "same" / "same-center"-style output is supported. The
        parameter is accepted for API compatibility (e.g., tests).
    """
    if mode not in ("same", "same-center"):
        raise ValueError(f"Unsupported mode for separable_conv2d: {mode!r}")

    img = np.asarray(img, dtype=np.float64)
    k_row = np.asarray(k_row, dtype=np.float64)
    if k_col is None:
        k_col = k_row
    else:
        k_col = np.asarray(k_col, dtype=np.float64)

    # Convolve along rows (axis=0)
    tmp = np.apply_along_axis(_same_center_conv1d, axis=0, arr=img, k=k_row)
    # Convolve along columns (axis=1)
    out = np.apply_along_axis(_same_center_conv1d, axis=1, arr=tmp, k=k_col)
    return out
