# src/exconv/xmodal/sound2image.py
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from exconv.core import fftnd, ifftnd, radial_grid_2d
from exconv.dsp.normalize import normalize_range_01, normalize_chroma_midpoint, clip_01
from exconv.io import rgb_to_luma, luma_to_rgb, to_stereo

Mode = Literal["mono", "stereo", "mid-side"]
ColorMode = Literal["luma", "color"]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _rfft_magnitude(sig: np.ndarray) -> np.ndarray:
    """
    1D magnitude spectrum normalized to max=1.
    If the signal is silent, returns an all-ones curve (identity filter).
    """
    sig = np.asarray(sig, dtype=np.float64).ravel()
    spec = np.abs(np.fft.rfft(sig))
    m = float(spec.max()) if spec.size > 0 else 0.0
    if m > 0.0:
        spec = spec / m
    else:
        spec = np.ones_like(spec, dtype=np.float64)
    return spec.astype(np.float32)


def _radial_filter_from_curve(curve: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Map a 1D curve (frequency magnitude) radially onto a 2D grid.

    curve : (F,)
    returns H2 : (H,W)
    """
    curve = np.asarray(curve, dtype=np.float32)
    rho = radial_grid_2d(H, W)  # 0..1
    if curve.size <= 1:
        return np.ones((H, W), dtype=np.float32)

    idx = np.minimum((rho * (curve.size - 1)).astype(int), curve.size - 1)
    H2 = curve[idx]
    return H2.astype(np.float32)


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image in [0,1] to YCbCr (approx. BT.601).
    Cb/Cr are offset by +0.5 so neutral chroma sits at 0.5, which keeps
    intermediate filtering/clipping in image-space [0,1] well-behaved.
    Returns same shape (H,W,3).
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 0.564 * (b - Y) + 0.5
    Cr = 0.713 * (r - Y) + 0.5

    return np.stack([Y, Cb, Cr], axis=-1).astype(np.float32)


def _ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """
    Inverse YCbCr -> RGB (approx. BT.601).
    Input/Output float32, not clipped.
    """
    ycbcr = np.asarray(ycbcr, dtype=np.float32)
    Y, Cb, Cr = ycbcr[..., 0], ycbcr[..., 1] - 0.5, ycbcr[..., 2] - 0.5

    r = Y + 1.402 * Cr
    g = Y - 0.344136 * Cb - 0.714136 * Cr
    b = Y + 1.772 * Cb

    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _fft_filter_apply(img2d: np.ndarray, H2: np.ndarray) -> np.ndarray:
    """
    2D FFT filter: FFT2 -> multiply -> IFFT2 (real).
    img2d, H2 : (H,W)
    """
    img2d = np.asarray(img2d, dtype=np.float32)
    H2 = np.asarray(H2, dtype=np.complex64)

    F = fftnd(img2d, axes=(0, 1), real_input=False)
    Y = F * H2
    y = ifftnd(Y, axes=(0, 1), real_output=False)
    return y.real.astype(np.float32)


# ---------------------------------------------------------------------
# Audio -> curves helpers
# ---------------------------------------------------------------------

def _prepare_audio_channels(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given audio (N,) or (N,C), return:

    mono, L, R, mid, (sideL, sideR)

    - mono  = (L+R)/2
    - mid   = (L+R)/2
    - sideL = L - mid
    - sideR = R - mid

    If input is mono, L=R=mono, sideL=sideR=0.
    """
    x = np.asarray(audio, dtype=np.float32)

    # Mono
    if x.ndim == 1:
        mono = x
        L = mono
        R = mono
    elif x.ndim == 2:
        # Convert to stereo consistently (handles >2ch)
        stereo = to_stereo(x)  # (N,2)
        L = stereo[:, 0]
        R = stereo[:, 1]
        mono = 0.5 * (L + R)
    else:
        raise ValueError(f"audio must be 1D or 2D, got shape {x.shape}")

    mid = 0.5 * (L + R)
    sideL = L - mid
    sideR = R - mid

    return mono, L, R, mid, sideL, sideR


# ---------------------------------------------------------------------
# Public function: spectral_sculpt
# ---------------------------------------------------------------------

def spectral_sculpt(
    image: np.ndarray,
    audio: np.ndarray,
    sr: int,  # kept for future, not used in this simple v2
    *,
    mode: Mode = "mono",
    colorspace: ColorMode = "luma",
    normalize: bool = True,
    safe_color: bool = True,
    chroma_strength: float = 0.5,
    chroma_clip: float = 0.25,
) -> np.ndarray:
    """
    Sculpt an image using the spectrum of an audio signal.

    Modes
    -----
    mode = "mono"
        - Build a mono spectrum from the audio (downmix stereo).
        - If colorspace == "luma":
             Convert image -> luma
             Radial filter with mono curve
             Return luma (2D) or luma expanded back to RGB (if input was RGB).
        - If colorspace == "color":
             Convert image -> YCbCr
             Filter Y with mono curve
             Leave Cb, Cr unchanged
             Convert back to RGB.

    mode = "stereo"
        - Requires stereo (or will auto-make stereo from mono).
        - Build spectra:
             S_mono from mono = (L+R)/2
             S_L from left
             S_R from right
        - If colorspace == "luma":
             Same as "mono": apply only S_mono to luma, ignore color.
        - If colorspace == "color":
             Convert image -> YCbCr
             Y  filtered with S_mono (radial)
             Cb filtered with S_L (radial)
             Cr filtered with S_R (radial)
             Convert back to RGB.

    mode = "mid-side"
        - mid = (L+R)/2
        - sideL = L - mid, sideR = R - mid
        - Build spectra:
             S_mid   from mid
             S_sideL from sideL
             S_sideR from sideR
        - If colorspace == "luma":
             Convert image -> luma
             Filter with S_mid only (no color encodes).
        - If colorspace == "color":
             Convert image -> YCbCr
             Y  filtered with S_mid
             Cb filtered with S_sideL
             Cr filtered with S_sideR
             Convert back to RGB.

    Parameters
    ----------
    image : ndarray
        2D gray or 3D RGB image. Any dtype; converted to float32 0..1 internally.
    audio : ndarray
        1D mono or 2D multi-channel audio.
    sr : int
        Sample rate (currently unused but kept for future options).
    mode : {"mono","stereo","mid-side"}
    colorspace : {"luma","color"}
        - "luma": only luminance filtering (no color encoding).
        - "color": YCbCr filtering as described above.
    normalize : bool
        If True, normalize the final output to [0,1].
    safe_color : bool
        If True, applies chroma-safe normalization to reduce casts.
    chroma_strength : float
        Blend between original chroma (0.0) and fully filtered chroma (1.0).
    chroma_clip : float
        Maximum absolute chroma deviation from 0.5 when safe_color is True.

    Returns
    -------
    out : ndarray
        Sculpted image as float32 in [0,1] if normalize=True.
    """
    if mode not in ("mono", "stereo", "mid-side"):
        raise ValueError(f"Unknown mode {mode!r}")
    if colorspace not in ("luma", "color"):
        raise ValueError(f"Unknown colorspace {colorspace!r}")

    img = np.asarray(image)
    orig_ndim = img.ndim
    if orig_ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got {img.shape}")

    # Convert image to float32, roughly 0..1
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0

    H, W = img_f.shape[:2]

    # --- audio channel prep ---
    mono, L, R, mid, sideL, sideR = _prepare_audio_channels(audio)

    # --- build 1D spectra & 2D filters according to mode ---
    if mode == "mono":
        S_mono = _rfft_magnitude(mono)
        H_mono = _radial_filter_from_curve(S_mono, H, W)

    elif mode == "stereo":
        S_mono = _rfft_magnitude(mono)
        S_L = _rfft_magnitude(L)
        S_R = _rfft_magnitude(R)
        H_mono = _radial_filter_from_curve(S_mono, H, W)
        H_L = _radial_filter_from_curve(S_L, H, W)
        H_R = _radial_filter_from_curve(S_R, H, W)

    else:  # mode == "mid-side"
        S_mid = _rfft_magnitude(mid)
        S_sideL = _rfft_magnitude(sideL)
        S_sideR = _rfft_magnitude(sideR)
        H_mid = _radial_filter_from_curve(S_mid, H, W)
        H_sideL = _radial_filter_from_curve(S_sideL, H, W)
        H_sideR = _radial_filter_from_curve(S_sideR, H, W)

    # ------------------------------------------------------------------
    # LUMA MODE: only mono/mid applied, ignore YCbCr color encodings
    # ------------------------------------------------------------------
    if colorspace == "luma":
        if img_f.ndim == 2:
            luma = img_f
        else:
            luma = rgb_to_luma(img_f)

        if mode in ("mono", "stereo"):
            H_luma = H_mono
        else:  # "mid-side"
            H_luma = H_mid

        y = _fft_filter_apply(luma, H_luma)

        if normalize:
            y = normalize_range_01(y)

        if orig_ndim == 3:
            # expand back to RGB to match input shape
            y_rgb = luma_to_rgb(y, dtype=np.float32)
            return y_rgb
        else:
            return y

    # ------------------------------------------------------------------
    # COLOR MODE: YCbCr with separate filters for Y, Cb, Cr
    # ------------------------------------------------------------------
    # ensure 3-channel RGB representation; if input was 2D, treat as gray
    if img_f.ndim == 2:
        img_rgb = luma_to_rgb(img_f, dtype=np.float32)
    else:
        img_rgb = img_f

    ycbcr = _rgb_to_ycbcr(img_rgb)
    Y = ycbcr[..., 0]
    Cb = ycbcr[..., 1]
    Cr = ycbcr[..., 2]

    if mode == "mono":
        # Y gets mono; Cb, Cr unchanged
        Y_f = _fft_filter_apply(Y, H_mono)
        Cb_f = Cb
        Cr_f = Cr

    elif mode == "stereo":
        Y_f = _fft_filter_apply(Y, H_mono)
        Cb_f = _fft_filter_apply(Cb, H_L)
        Cr_f = _fft_filter_apply(Cr, H_R)

    else:  # "mid-side"
        Y_f = _fft_filter_apply(Y, H_mid)
        Cb_f = _fft_filter_apply(Cb, H_sideL)
        Cr_f = _fft_filter_apply(Cr, H_sideR)

    # blend chroma with original to reduce casts
    chroma_strength = float(np.clip(chroma_strength, 0.0, 1.0))
    if chroma_strength < 1.0:
        Cb_f = Cb * (1.0 - chroma_strength) + Cb_f * chroma_strength
        Cr_f = Cr * (1.0 - chroma_strength) + Cr_f * chroma_strength

    ycbcr_f = np.stack([Y_f, Cb_f, Cr_f], axis=-1)

    # soft clip YCbCr to avoid extreme values before RGB conversion
    ycbcr_f = np.clip(ycbcr_f, -1.0, 2.0)
    rgb_out = _ycbcr_to_rgb(ycbcr_f)

    if normalize:
        if safe_color:
            # Normalize Y globally, squash chroma, then clip RGB
            ycbcr_norm = np.empty_like(ycbcr_f)
            ycbcr_norm[..., 0] = normalize_range_01(ycbcr_f[..., 0])
            ycbcr_norm[..., 1] = normalize_chroma_midpoint(ycbcr_f[..., 1])
            ycbcr_norm[..., 2] = normalize_chroma_midpoint(ycbcr_f[..., 2])
            # hard clamp chroma around 0.5
            clip = max(0.0, float(chroma_clip))
            if clip > 0.0:
                ycbcr_norm[..., 1] = np.clip(ycbcr_norm[..., 1], 0.5 - clip, 0.5 + clip)
                ycbcr_norm[..., 2] = np.clip(ycbcr_norm[..., 2], 0.5 - clip, 0.5 + clip)
            rgb_out = _ycbcr_to_rgb(ycbcr_norm)
            rgb_out = clip_01(rgb_out)
        else:
            rgb_out = clip_01(rgb_out)

    return rgb_out
