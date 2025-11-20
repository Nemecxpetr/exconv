# exconv/io/image.py
"""
Image I/O utilities using Pillow."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Union, Optional

import numpy as np
from PIL import Image

ArrayLike = np.ndarray
PathLike = Union[str, Path]
ImageMode = Literal["L", "RGB", "RGBA", "keep"]

__all__ = [
    "read_image",
    "write_image",
    "as_float32",
    "as_uint8",
    "rgb_to_luma",
    "luma_to_rgb",
]


def _pathify(path: PathLike) -> str:
    return str(Path(path))


def read_image(
    path: PathLike,
    *,
    mode: ImageMode = "RGB",
    dtype: Union[np.dtype, str] = "uint8",
) -> np.ndarray:
    """
    Read an image via Pillow.

    Parameters
    ----------
    path : str or Path
        Input image path.
    mode : {"L", "RGB", "RGBA", "keep"}, default="RGB"
        - "keep": use the file's native mode.
        - otherwise: convert via Pillow's .convert(mode).
    dtype : numpy dtype or str, default="uint8"
        Output dtype for the ndarray.

    Returns
    -------
    img : ndarray
        Image data, 2D for "L" or 3D for color.
    """
    p = _pathify(path)
    with Image.open(p) as im:
        if mode != "keep":
            im = im.convert(mode)
        arr = np.asarray(im)

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    return arr


def as_float32(x: ArrayLike) -> np.ndarray:
    """
    Convert image-like array to float32 in a reasonable normalized range.

    Rules
    -----
    - uint8: scaled to [0, 1] by dividing 255
    - other unsigned ints: scaled by max value to [0, 1]
    - signed ints: scaled by max(|min|, |max|) to [-1, 1]
    - floats: cast to float32 without rescaling

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    y : ndarray, float32
    """
    arr = np.asarray(x)

    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)

    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) / 255.0).astype(np.float32)

    if np.issubdtype(arr.dtype, np.unsignedinteger):
        info = np.iinfo(arr.dtype)
        return (arr.astype(np.float32) / float(info.max)).astype(np.float32)

    if np.issubdtype(arr.dtype, np.signedinteger):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        if scale == 0:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr.astype(np.float32) / scale).astype(np.float32)

    # fallback: just cast
    return arr.astype(np.float32)


def as_uint8(x: ArrayLike) -> np.ndarray:
    """
    Convert an array to uint8 for deterministic image saving.

    Rules
    -----
    - float: if min>=0 and max<=1 → scale by 255; else clip to [0, 255]
    - ints: scaled linearly to [0, 255] based on dtype range
    - otherwise: best-effort cast with clipping

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    y : ndarray, uint8
    """
    arr = np.asarray(x)

    if np.issubdtype(arr.dtype, np.floating):
        arr_f = arr.astype(np.float64)
        vmin = float(np.nanmin(arr_f))
        vmax = float(np.nanmax(arr_f))

        if np.isfinite(vmin) and np.isfinite(vmax) and 0.0 <= vmin and vmax <= 1.0 + 1e-8:
            arr_f = arr_f * 255.0
        arr_f = np.clip(arr_f, 0.0, 255.0)
        return arr_f.astype(np.uint8)

    if np.issubdtype(arr.dtype, np.unsignedinteger):
        info = np.iinfo(arr.dtype)
        arr_f = arr.astype(np.float64) / float(info.max) * 255.0
        arr_f = np.clip(arr_f, 0.0, 255.0)
        return arr_f.astype(np.uint8)

    if np.issubdtype(arr.dtype, np.signedinteger):
        info = np.iinfo(arr.dtype)
        span = float(info.max - info.min)
        if span <= 0:
            return np.zeros_like(arr, dtype=np.uint8)
        arr_f = (arr.astype(np.float64) - float(info.min)) / span * 255.0
        arr_f = np.clip(arr_f, 0.0, 255.0)
        return arr_f.astype(np.uint8)

    # fallback
    return np.clip(arr.astype(np.float64), 0.0, 255.0).astype(np.uint8)


def rgb_to_luma(x: ArrayLike) -> np.ndarray:
    """
    Convert RGB(A) image to 2D luminance using Rec.709 coefficients.

    Parameters
    ----------
    x : ndarray
        - 2D: returned as float64
        - 3D: (..., 3 or 4 channels), uses first 3 channels

    Returns
    -------
    luma : ndarray, shape (H, W), float64
    """
    arr = np.asarray(x)

    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got {arr.shape}")

    if arr.shape[2] < 3:
        return arr[..., 0].astype(np.float64, copy=False)

    r = arr[..., 0].astype(np.float64)
    g = arr[..., 1].astype(np.float64)
    b = arr[..., 2].astype(np.float64)

    # Rec.709 / sRGB
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def luma_to_rgb(x: ArrayLike, *, dtype: Optional[Union[np.dtype, str]] = None) -> np.ndarray:
    """
    Expand a 2D luminance image to RGB by channel replication.

    Parameters
    ----------
    x : ndarray
        2D or 3D with a singleton channel dimension.
    dtype : numpy dtype or str, optional
        If given, cast result to this dtype.

    Returns
    -------
    rgb : ndarray, shape (H, W, 3)
    """
    arr = np.asarray(x)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D luma image, got {arr.shape}")

    rgb = np.stack([arr, arr, arr], axis=-1)
    if dtype is not None:
        rgb = rgb.astype(dtype, copy=False)
    return rgb


def write_image(
    path: PathLike,
    data: ArrayLike,
    *,
    mode: Optional[str] = None,
) -> None:
    """
    Save an image via Pillow with deterministic uint8 conversion.

    Parameters
    ----------
    path : str or Path
        Output file path (extension decides format).
    data : ndarray
        Image data, 2D or 3D.
    mode : str or None
        Pillow image mode. If None, deduced from data shape:
        - 2D → "L"
        - 3D, C=1 → "L"
        - 3D, C=3 → "RGB"
        - 3D, C=4 → "RGBA"
    """
    arr = np.asarray(data)

    if arr.ndim == 2:
        img_mode = "L"
    elif arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            img_mode = "L"
            arr = arr[..., 0]
        elif c == 3:
            img_mode = "RGB"
        elif c == 4:
            img_mode = "RGBA"
        else:
            raise ValueError(f"Unsupported channel count {c}")
    else:
        raise ValueError(f"Expected 2D or 3D array, got {arr.shape}")

    if mode is not None:
        img_mode = mode

    # Convert to uint8 deterministically
    arr_u8 = as_uint8(arr)
    img = Image.fromarray(arr_u8, mode=img_mode)
    img.save(_pathify(path))
