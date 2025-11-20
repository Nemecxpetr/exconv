# src/exconv/conv2d/color.py
"""Color space and gamma correction utilities for images."""
import numpy as np

def apply_gamma(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to an image array.

    Parameters
    ----------
    x : ndarray, float or uint8
        Image data. Can be 2D or 3D.
        If float: expected range 0..1 or 0..255.
    gamma : float
        Gamma exponent. 
        - gamma < 1: brighten
        - gamma > 1: darken

    Returns
    -------
    ndarray
        Gamma-adjusted image, same dtype as input.
    """
    arr = np.asarray(x)

    # convert uint8 to float 0..1
    was_uint8 = arr.dtype == np.uint8
    if was_uint8:
        arr_f = arr.astype(np.float32) / 255.0
    else:
        # assume float in 0..1 or 0..255; normalize softly
        arr_f = arr.astype(np.float64)
        if arr_f.max() > 1.0:
            arr_f /= 255.0

    # apply gamma
    out = np.power(arr_f, gamma)

    # convert back
    if was_uint8:
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
    else:
        return out.astype(x.dtype)

