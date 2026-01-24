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
    "upscale_image",
    "UPSCALE_METHODS",
]


_PIL_RESAMPLE_METHODS = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}

_OPENCV_SUPERRES_METHODS = {
    "opencv-espcn": "espcn",
    "opencv-fsrcnn": "fsrcnn",
    "opencv-lapsrn": "lapsrn",
    "opencv-edsr": "edsr",
}

UPSCALE_METHODS = tuple(list(_PIL_RESAMPLE_METHODS) + list(_OPENCV_SUPERRES_METHODS))

_SUPERRES_CACHE: dict[tuple[str, str, int], object] = {}


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


def _parse_upscale_method(method: str) -> tuple[str, str]:
    name = method.strip().lower()
    if name in _PIL_RESAMPLE_METHODS:
        return "pillow", name
    if name in _OPENCV_SUPERRES_METHODS:
        return "opencv", _OPENCV_SUPERRES_METHODS[name]
    if name.startswith("opencv:") or name.startswith("opencv-"):
        algo = name.split(":", 1)[1] if ":" in name else name.split("-", 1)[1]
        if algo in _OPENCV_SUPERRES_METHODS.values():
            return "opencv", algo
    raise ValueError(f"Unknown upscale method {method!r}. Use one of: {UPSCALE_METHODS}")


def _validate_scale(scale: float) -> float:
    try:
        scale_f = float(scale)
    except (TypeError, ValueError):
        raise ValueError(f"scale must be a number, got {scale!r}") from None
    if scale_f <= 0:
        raise ValueError(f"scale must be > 0, got {scale_f!r}")
    return scale_f


def _resize_pillow(img_u8: np.ndarray, scale: float, method: str) -> np.ndarray:
    if scale == 1.0:
        return img_u8
    h, w = img_u8.shape[:2]
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if new_w < 1 or new_h < 1:
        raise ValueError(f"scale {scale!r} produces invalid size {(new_h, new_w)}")
    resample = _PIL_RESAMPLE_METHODS[method]
    pil_img = Image.fromarray(img_u8)
    out = pil_img.resize((new_w, new_h), resample=resample)
    return np.asarray(out, dtype=np.uint8)


def _resize_opencv_superres(
    img_u8: np.ndarray,
    *,
    scale: float,
    model: PathLike,
    algo: str,
) -> np.ndarray:
    scale_f = _validate_scale(scale)
    if scale_f <= 1.0:
        raise ValueError("OpenCV super-res requires scale > 1.")
    if abs(scale_f - int(scale_f)) > 1e-6:
        raise ValueError("OpenCV super-res scale must be an integer (e.g., 2, 3, 4).")
    scale_i = int(scale_f)

    if model is None:
        raise ValueError("OpenCV super-res requires --upscale-model.")

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV super-res requires opencv-contrib-python.") from exc

    if not hasattr(cv2, "dnn_superres"):
        raise ImportError("OpenCV build lacks dnn_superres; install opencv-contrib-python.")

    model_path = Path(model)
    if not model_path.exists():
        raise ValueError(f"OpenCV super-res model not found: {model_path}")

    algo = algo.lower()
    inferred_algo = _infer_superres_algo_from_model(model_path)
    if inferred_algo is not None and inferred_algo != algo:
        raise ValueError(
            "OpenCV super-res model does not match the selected algorithm. "
            f"Model {model_path.name!r} looks like {inferred_algo!r}, "
            f"but --upscale-method requested {algo!r}. "
            "Use a matching method (e.g. opencv-edsr for EDSR_x2.pb)."
        )

    cache_key = (str(model_path), algo, scale_i)
    sr = _SUPERRES_CACHE.get(cache_key)
    if sr is None:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel(algo, scale_i)
        _SUPERRES_CACHE[cache_key] = sr

    alpha = None
    if img_u8.ndim == 2:
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        mode = "gray"
    elif img_u8.ndim == 3 and img_u8.shape[2] == 1:
        bgr = cv2.cvtColor(img_u8[..., 0], cv2.COLOR_GRAY2BGR)
        mode = "gray"
    elif img_u8.ndim == 3 and img_u8.shape[2] == 3:
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        mode = "rgb"
    elif img_u8.ndim == 3 and img_u8.shape[2] == 4:
        alpha = img_u8[..., 3]
        bgr = cv2.cvtColor(img_u8[..., :3], cv2.COLOR_RGB2BGR)
        mode = "rgba"
    else:
        raise ValueError(f"Expected 2D or 3D image with 1,3,4 channels, got {img_u8.shape}")

    out_bgr = sr.upsample(bgr)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    if mode == "gray":
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2GRAY)
    if mode == "rgba":
        out_h, out_w = out_rgb.shape[:2]
        alpha_up = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return np.dstack([out_rgb, alpha_up])
    return out_rgb


def _infer_superres_algo_from_model(model_path: Path) -> Optional[str]:
    name = model_path.stem.lower()
    for algo in ("edsr", "espcn", "fsrcnn", "lapsrn"):
        if algo in name:
            return algo
    return None


def upscale_image(
    x: ArrayLike,
    *,
    scale: float = 1.0,
    method: str = "lanczos",
    model: Optional[PathLike] = None,
) -> np.ndarray:
    """
    Resize/upscale an image using classic resampling or optional ML backends.

    Parameters
    ----------
    x : ndarray
        2D or 3D image array.
    scale : float, default=1.0
        Uniform scale factor. Values > 1 upscale; values < 1 downscale.
    method : str, default="lanczos"
        Resampling or backend method. Supported:
        - classic: nearest, box, bilinear, hamming, bicubic, lanczos
        - ML: opencv-espcn, opencv-fsrcnn, opencv-lapsrn, opencv-edsr
    model : str or Path, optional
        Path to the model file required by OpenCV super-res backends.

    Returns
    -------
    ndarray
        Upscaled image as uint8 (same shape if scale=1 and Pillow backend).
    """
    backend, method_key = _parse_upscale_method(method)
    if backend == "opencv":
        img_u8 = as_uint8(x)
        return _resize_opencv_superres(img_u8, scale=scale, model=model, algo=method_key)

    scale_f = _validate_scale(scale)
    img_u8 = as_uint8(x)
    return _resize_pillow(img_u8, scale_f, method_key)


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
