# src/exconv/conv2d/image.py
"""2D image convolution via FFT: auto- and pair-convolution."""
from __future__ import annotations

from typing import Literal, Tuple
import numpy as np

from exconv.core.fft import linear_freq_multiply
from exconv.conv2d.color import apply_gamma

ArrayLike = np.ndarray
Mode = Literal["full", "same-first", "same-center"]
ColorMode = Literal["luma", "channels"]

__all__ = [
    "auto_convolve",
    "pair_convolve",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_2d(x: ArrayLike) -> ArrayLike:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for conv, got shape {x.shape}")
    return x


def _conv2d_real(
    img2d: ArrayLike,
    ker2d: ArrayLike,
    mode: Mode = "same-center",
    circular: bool = False,
) -> ArrayLike:
    """
    Core 2D real-valued convolution via FFT. No clipping/casting here.
    """
    img2d = _ensure_2d(img2d).astype(np.float64, copy=False)
    ker2d = _ensure_2d(ker2d).astype(np.float64, copy=False)

    fft_mode = "circular" if circular else mode
    y = linear_freq_multiply(
        img2d,
        ker2d,
        axes=(0, 1),
        mode=fft_mode,
        use_real_fft=True,
    )
    # linear_freq_multiply already returns real for real inputs
    return np.asarray(y, dtype=np.float64)

def _postprocess_output(y: np.ndarray, ref: np.ndarray, normalize: str = "clip") -> np.ndarray:
    """
    Post-process convolution output `y`.

    Parameters
    ----------
    y : np.ndarray
        Convolution result (float64).
    ref : np.ndarray
        Reference image (used mainly for range in 'clip' mode).
    normalize : {"none", "clip", "rescale"}
        - "none": leave in float64 (no clipping/rescaling).
        - "clip": clip to ref's data range, cast to ref dtype.
        - "rescale": linearly rescale to [0,1], float32.

    Notes
    -----
    For image examples we want:
    - normalize="none"  → raw float result (for analytical comparisons).
    - normalize="rescale" → float32 in [0,1], regardless of ref dtype.
    """
    y = np.asarray(y, dtype=np.float64)

    if normalize == "none":
        # Keep raw float64 result (used in Gaussian reference tests, etc.)
        return y

    if normalize == "rescale":
        # Always map to [0,1] in float32 for image usage
        ymin = float(np.min(y))
        ymax = float(np.max(y))

        # Handle NaNs / infs robustly
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            ymin = float(np.min(y))
            ymax = float(np.max(y))

        eps = np.finfo(np.float64).eps
        if ymax <= ymin + eps:
            # Degenerate case: flat image
            return np.zeros_like(y, dtype=np.float32)

        y_norm = (y - ymin) / (ymax - ymin)
        return y_norm.astype(np.float32)

    if normalize == "clip":
        ref = np.asarray(ref)
        if np.issubdtype(ref.dtype, np.integer):
            info = np.iinfo(ref.dtype)
            rmin, rmax = float(info.min), float(info.max)
        elif np.issubdtype(ref.dtype, np.floating):
            rmin, rmax = float(np.min(ref)), float(np.max(ref))
        else:
            rmin, rmax = float(np.min(ref)), float(np.max(ref))

        y_clipped = np.clip(y, rmin, rmax)
        return y_clipped.astype(ref.dtype)

    raise ValueError(f"Unknown normalize mode: {normalize!r}")


def _rgb_to_luma(img: ArrayLike) -> ArrayLike:
    """
    Convert an RGB(A) image to 2D luminance using Rec.709 coefficients.
    If the image is already 2D or single-channel, it's returned as float64.
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image for luma, got {arr.shape}")

    if arr.shape[2] < 3:
        # Treat as single channel
        return arr[..., 0].astype(np.float64, copy=False)

    r = arr[..., 0].astype(np.float64)
    g = arr[..., 1].astype(np.float64)
    b = arr[..., 2].astype(np.float64)

    # Rec. 709 / sRGB luminance
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luma


def _prepare_kernel_for_luma(kernel: ArrayLike) -> ArrayLike:
    """
    For luma-mode convolution, reduce kernel to 2D if necessary.
    - 2D: used as-is
    - 3D with at least 3 channels: convert to luma the same way as image
    """
    ker = np.asarray(kernel)
    if ker.ndim == 2:
        return ker.astype(np.float64, copy=False)
    if ker.ndim == 3:
        # Convert spatially to luma in the same way (channel-wise)
        if ker.shape[2] >= 3:
            r = ker[..., 0].astype(np.float64)
            g = ker[..., 1].astype(np.float64)
            b = ker[..., 2].astype(np.float64)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        return ker[..., 0].astype(np.float64, copy=False)
    raise ValueError(f"Unsupported kernel shape for luma mode: {ker.shape}")


def _broadcast_kernel_channels(kernel: ArrayLike, channels: int) -> Tuple[ArrayLike, bool]:
    """
    Normalize kernel shapes for per-channel convolution.

    Returns (kernel_ndarray, per_channel_flag).

    - If kernel is 2D, we apply the same kernel to all channels (per_channel_flag=False).
    - If kernel is 3D, we require kernel.shape[2] == channels and apply per-channel.
    """
    ker = np.asarray(kernel)
    if ker.ndim == 2:
        return ker.astype(np.float64, copy=False), False

    if ker.ndim == 3:
        if ker.shape[2] != channels:
            raise ValueError(
                f"Kernel channel count {ker.shape[2]} "
                f"does not match image channels {channels}"
            )
        return ker.astype(np.float64, copy=False), True

    raise ValueError(f"Unsupported kernel shape for per-channel mode: {ker.shape}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_convolve(
    img: ArrayLike,
    mode: Mode = "same-center",
    circular: bool = False,
    colorspace: ColorMode = "luma",
    normalize: str = "clip",  # "clip" | "rescale"
    gamma=None,
) -> ArrayLike:
    """
    Self-convolution of an image (auto-correlation-like blur) via 2D FFT.

    Parameters
    ----------
    img : ndarray
        Input image, 2D (H, W) or 3D (H, W, C).
    mode : {"full", "same-first", "same-center"}
        Spatial size policy for linear convolution. Ignored if circular=True.
    circular : bool
        If True, perform circular convolution (no padding, wrap-around).
    colorspace : {"luma", "channels"}
        - "luma": reduce to luminance and convolve only that 2D signal.
        - "channels": convolve each channel separately with its own channel
          from the image (auto-convolution per channel).

    Returns
    -------
    ndarray
        Convolved image:
        - For "luma": 2D array.
        - For "channels": same shape and dtype as `img`.
    """
    return pair_convolve(
        img,
        kernel=img,
        mode=mode,
        circular=circular,
        colorspace=colorspace,
        normalize=normalize,
        gamma=gamma,
    )

def pair_convolve(
    image: np.ndarray,
    kernel: np.ndarray,
    mode: str = "same-center",
    circular: bool = False,
    colorspace: str = "channels",
    normalize: str = "clip",
    gamma: float | None = None,
) -> np.ndarray:
    """
    Convolve an image with a kernel.

    Parameters
    ----------
    image : np.ndarray
        Input image, 2D (H, W) or 3D (H, W, C). Integer images are interpreted
        as [0, 1] after normalization.
    kernel : np.ndarray
        2D kernel or 3D kernel (for per-channel kernels in 'channels' mode).
    mode : {"full", "same-center", "same-first"}
        Convolution size mode.
    circular : bool
        If True, perform circular convolution. Otherwise linear with padding.
    colorspace : {"luma", "channels"}
        - "luma": convert image (and kernel if needed) to luma, convolve once.
          For RGB input, the luma result is expanded back to C channels, so the
          output shape matches the input shape.
        - "channels": convolve each channel separately (or with a broadcast
          kernel).
    normalize : {"none", "clip", "rescale"}
        Normalization applied after convolution. See `_postprocess_output`.
    gamma : float or None
        If not None, apply gamma correction after normalization.

    Returns
    -------
    y : np.ndarray
        Convolved image.
    """
    img_arr = np.asarray(image)

    if img_arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {img_arr.shape}")

    # --- Convert image to float for convolution, normalize integer ranges to [0,1] ---
    if np.issubdtype(img_arr.dtype, np.integer):
        # Treat e.g. uint8 as [0, 255] → [0, 1]
        info = np.iinfo(img_arr.dtype)
        scale = float(info.max)
        img_f = img_arr.astype(np.float64) / scale
    else:
        img_f = img_arr.astype(np.float64)

    # Reference for postprocessing (always float image in [0,1] or original float)
    ref_for_norm = img_f

    # ------------------------------------------------------------------
    # LUMA MODE: convolve in luma, return 2D for gray, C-channel for RGB
    # ------------------------------------------------------------------
    if colorspace == "luma":
        ker2d = _prepare_kernel_for_luma(kernel)

        if img_f.ndim == 2:
            # Already single-channel
            luma_in = img_f
            y_luma = _conv2d_real(luma_in, ker2d, mode=mode, circular=circular)
            y = _postprocess_output(y_luma, ref_for_norm, normalize=normalize)
            if gamma is not None:
                y = apply_gamma(y, gamma=gamma)
            return y

        if img_f.ndim == 3 and img_f.shape[2] in (1, 3, 4):
            # RGB / RGBA / single-channel in 3D
            luma_in = _rgb_to_luma(img_f)
            y_luma = _conv2d_real(luma_in, ker2d, mode=mode, circular=circular)

            # Normalize using luma scale (float in [0,1])
            y_luma = _postprocess_output(y_luma, ref_for_norm[..., 0], normalize=normalize)
            if gamma is not None:
                y_luma = apply_gamma(y_luma, gamma=gamma)

            C = img_f.shape[2]
            # Expand back to C channels so out.shape == image.shape
            y = np.repeat(y_luma[..., None], C, axis=-1)
            return y

        raise ValueError("colorspace='luma' expects 2D or 3D RGB/RGBA image")

    # ------------------------------------------------------------------
    # CHANNELS MODE: convolve each channel independently
    # ------------------------------------------------------------------
    if colorspace != "channels":
        raise ValueError(f"Unknown colorspace: {colorspace!r}")

    if img_f.ndim == 2:
        # Single-channel image, simple 2D conv
        img2d = img_f
        ker2d = _ensure_2d(kernel).astype(np.float64, copy=False)
        y = _conv2d_real(img2d, ker2d, mode=mode, circular=circular)
        y = _postprocess_output(y, ref_for_norm, normalize=normalize)
        if gamma is not None:
            y = apply_gamma(y, gamma=gamma)
        return y

    # 3D: per-channel routing
    H, W, C = img_f.shape
    ker_b, per_channel = _broadcast_kernel_channels(kernel, C)

    out = None  # allocate after first conv to get correct spatial shape

    for c in range(C):
        img_c = img_f[..., c]

        if per_channel:
            ker_c = _ensure_2d(ker_b[..., c])
        else:
            ker_c = _ensure_2d(ker_b)

        y_c = _conv2d_real(img_c, ker_c, mode=mode, circular=circular)

        if out is None:
            # y_c is (H_out, W_out) for this mode/kernel
            out = np.empty(y_c.shape + (C,), dtype=np.float64)

        out[..., c] = y_c

    # Post-process once after all channels are computed
    y = _postprocess_output(out, ref_for_norm, normalize=normalize)
    if gamma is not None:
        y = apply_gamma(y, gamma=gamma)

    return y


