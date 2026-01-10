"""
Image -> sound helpers:

- image2sound_flat:
    image -> grayscale -> flatten -> impulse -> conv

- image2sound_hist:
    image -> grayscale -> histogram -> impulse -> conv

- image2sound_radial:
    image -> fft2 -> radial unwrap -> impulse -> conv
"""

from __future__ import annotations

from typing import Literal, Tuple
import numpy as np

try:
    from scipy import fft as _fft
except Exception:
    import numpy.fft as _fft

from exconv.dsp.windows import hann
from exconv.dsp.normalize import normalize_impulse, normalize_to_reference

PadMode = Literal["full", "same-center", "same-first"]
ImpulseNorm = Literal["none", "peak", "energy"]
OutNorm = Literal["none", "match_peak", "match_rms"]
ColorMode = Literal["luma", "rgb-mean", "rgb-stereo", "ycbcr-mid-side"]
RadiusMode = Literal["linear", "log"]
PhaseMode = Literal["zero", "random", "image", "min-phase", "spiral"]
SmoothingMode = Literal["none", "hann"]


# ------------------------------
# Color helpers
# ------------------------------
def _rgb_to_luma(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB(A) image to luminance (Rec.709).

    Expects img in float32 [0, 1] or uint8 [0, 255].
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")

    # drop alpha if present
    if arr.shape[2] > 3:
        arr = arr[..., :3]

    # ensure float32 in 0..1
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    # Rec.709 coefficients
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y.astype(np.float32)


def _image_to_gray(img: np.ndarray, colorspace: ColorMode) -> np.ndarray:
    """Convert 2D/3D image to a float32 grayscale array in [0, 1]."""
    arr = np.asarray(img)

    if arr.ndim == 3:
        if colorspace == "luma":
            g = _rgb_to_luma(arr)
        elif colorspace == "rgb-mean":
            if arr.dtype not in (np.float32, np.float64):
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)
            g = arr.mean(axis=-1)
        else:
            raise ValueError(f"Unsupported colorspace: {colorspace}")
    elif arr.ndim == 2:
        if arr.dtype not in (np.float32, np.float64):
            g = arr.astype(np.float32) / 255.0
        else:
            g = arr.astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")

    return g.astype(np.float32)


def _to_float_image(img: np.ndarray) -> np.ndarray:
    """Return image as float32 in [0, 1], shape (H, W, C)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return arr


def _extract_channels(img: np.ndarray, colorspace: ColorMode) -> list[np.ndarray]:
    """
    Return a list of float32 channel arrays in [0, 1] for the requested colorspace.
    Supports mono (len=1) or stereo (len=2) impulses.
    """
    arr = _to_float_image(img)

    # handle grayscale by repeating if stereo is requested
    if arr.shape[2] == 1 and colorspace in ("rgb-stereo", "ycbcr-mid-side"):
        arr = np.repeat(arr, 3, axis=2)

    if colorspace == "luma":
        return [_rgb_to_luma(arr)]
    if colorspace == "rgb-mean":
        return [arr.mean(axis=-1)]
    if colorspace == "rgb-stereo":
        # map R -> left, B -> right (if no B, repeat R)
        r = arr[..., 0]
        b = arr[..., 2] if arr.shape[2] > 2 else arr[..., 0]
        return [r.astype(np.float32), b.astype(np.float32)]
    if colorspace == "ycbcr-mid-side":
        # Treat Y as mid (M), Cr/Cb offsets as side signals (S_L / S_R).
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2] if arr.shape[2] > 2 else arr[..., 0]
        mid = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - mid) + 0.5
        cr = 0.713 * (r - mid) + 0.5
        side_l = cr - 0.5
        side_r = cb - 0.5
        left = np.clip(mid + side_l, 0.0, 1.0)
        right = np.clip(mid + side_r, 0.0, 1.0)
        return [left.astype(np.float32), right.astype(np.float32)]

    raise ValueError(f"Unsupported colorspace: {colorspace}")


def _stack_impulses(impulses: list[np.ndarray]) -> np.ndarray:
    """Stack impulses; if mono, return 1D, else (L, C) float32."""
    if not impulses:
        return np.zeros(1, dtype=np.float32)
    if len(impulses) == 1:
        return impulses[0].astype(np.float32)

    lengths = [imp.size for imp in impulses]
    max_len = max(lengths)
    out = np.zeros((max_len, len(impulses)), dtype=np.float32)
    for idx, imp in enumerate(impulses):
        out[: imp.size, idx] = imp.astype(np.float32)
    return out


# ------------------------------
# Impulse constructors
# ------------------------------
def _image_to_impulse_flat(
    img: np.ndarray,
    *,
    colorspace: ColorMode = "luma",
    remove_dc: bool = True,
    impulse_norm: ImpulseNorm = "energy",
    max_len: int | None = None,
) -> np.ndarray:
    """
    Convert an image to an impulse by flattening grayscale values
    (supports mono or stereo mapping).
    """
    channels = _extract_channels(img, colorspace)
    impulses: list[np.ndarray] = []
    for ch in channels:
        h = ch.ravel().astype(np.float32)

        # optional length control (very simple)
        if max_len is not None and max_len > 0 and h.size > max_len:
            factor = int(np.ceil(h.size / max_len))
            h = h[::factor]

        # optional DC removal
        if remove_dc and h.size > 0:
            h = h - np.mean(h)

        h = normalize_impulse(h, impulse_norm)

        impulses.append(h.astype(np.float32))

    return _stack_impulses(impulses)


def _image_to_impulse_hist(
    img: np.ndarray,
    *,
    colorspace: ColorMode = "luma",
    n_bins: int = 256,
    impulse_norm: ImpulseNorm = "energy",
    remove_dc: bool = False,
) -> np.ndarray:
    """
    Convert an image to an impulse given by its histogram.

    - grayscale (or color-mapped) -> values in [0,1]
    - histogram over [0,1] with n_bins
    - use counts as impulse shape
    """
    channels = _extract_channels(img, colorspace)
    impulses: list[np.ndarray] = []
    for ch in channels:
        vals = ch.ravel().astype(np.float32)
        if vals.size == 0:
            impulses.append(np.zeros(1, dtype=np.float32))
            continue

        hist, _ = np.histogram(vals, bins=n_bins, range=(0.0, 1.0), density=False)
        h = hist.astype(np.float32)

        if remove_dc:
            h = h - np.mean(h)

        h = normalize_impulse(h, impulse_norm)

        impulses.append(h.astype(np.float32))

    return _stack_impulses(impulses)


def _radial_bins(shape: tuple[int, int], n_bins: int, mode: RadiusMode) -> np.ndarray:
    """Return integer bin indices per pixel for radial averaging."""
    h, w = shape
    cy = np.float32(h / 2.0)
    cx = np.float32(w / 2.0)
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r / (r.max() + np.float32(1e-8))

    if mode == "linear":
        bin_idx = np.minimum((r * (n_bins - 1)).astype(np.int32), n_bins - 1)
    elif mode == "log":
        r_log = np.log1p(r * np.float32(9.0)) / np.log1p(np.float32(9.0))  # denser near DC
        bin_idx = np.minimum((r_log * (n_bins - 1)).astype(np.int32), n_bins - 1)
    else:
        raise ValueError(f"Unknown radius mode: {mode}")
    return bin_idx


def _radial_profile_from_bins(
    values: np.ndarray, bin_idx: np.ndarray, n_bins: int
) -> np.ndarray:
    """Average values according to precomputed radial bins."""
    flat_bins = bin_idx.ravel()
    flat_vals = values.ravel()
    sums = np.bincount(flat_bins, weights=flat_vals, minlength=n_bins).astype(np.float32)
    counts = np.bincount(flat_bins, minlength=n_bins).astype(np.float32)
    counts[counts == 0] = 1.0
    return (sums / counts).astype(np.float32)


def _phase_profile_from_bins(
    phase_map: np.ndarray, bin_idx: np.ndarray, n_bins: int
) -> np.ndarray:
    """Circular-mean phase per radial bin."""
    flat_bins = bin_idx.ravel()
    phases = phase_map.ravel()
    out = np.zeros(n_bins, dtype=np.float32)
    for b in range(n_bins):
        mask = flat_bins == b
        if not np.any(mask):
            out[b] = 0.0
            continue
        out[b] = float(np.angle(np.exp(1j * phases[mask]).mean()))
    return out


def _spiral_phase_profile(phase_map: np.ndarray, n_bins: int) -> np.ndarray:
    """Deterministic spiral walk from center to edges to pick phases."""
    h, w = phase_map.shape
    cy = np.float32((h - 1) / 2.0)
    cx = np.float32((w - 1) / 2.0)
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r / (r.max() + np.float32(1e-8))
    pi = np.float32(np.pi)
    theta = (np.arctan2(y - cy, x - cx) + pi) / (np.float32(2.0) * pi)  # 0..1

    # Spiral-ish ordering: radius + a small angle term
    key = r + np.float32(0.25) * theta
    order = np.argsort(key.ravel(), kind="mergesort")
    phases_sorted = phase_map.ravel()[order]

    if phases_sorted.size >= n_bins:
        out = phases_sorted[:n_bins]
    else:
        reps = int(np.ceil(n_bins / phases_sorted.size))
        out = np.tile(phases_sorted, reps)[:n_bins]
    return out.astype(np.float32)


def _minimum_phase_spectrum(mag_profile: np.ndarray, n_fft: int) -> np.ndarray:
    """
    Build a minimum-phase half-spectrum from a magnitude profile.
    """
    eps = 1e-8
    mag = np.maximum(mag_profile.astype(np.float32), eps)
    log_mag = np.log(mag)
    cep = _fft.irfft(log_mag, n=n_fft).astype(np.float32)

    cep_min = np.zeros_like(cep)
    cep_min[0] = cep[0]
    half = n_fft // 2
    cep_min[1:half] = 2.0 * cep[1:half]
    if n_fft % 2 == 0:
        cep_min[half] = cep[half]

    spec_min = _fft.rfft(cep_min, n=n_fft)
    return np.exp(spec_min).astype(np.complex64)


def _image_to_impulse_radial(
    img: np.ndarray,
    *,
    colorspace: ColorMode = "luma",
    impulse_len: int = 8192,
    radius_mode: RadiusMode = "linear",
    phase_mode: PhaseMode = "zero",
    smoothing: SmoothingMode = "hann",
    remove_dc: bool = True,
    impulse_norm: ImpulseNorm = "energy",
) -> np.ndarray:
    """
    Derive an impulse from the radial average of the image's 2D FFT magnitude.
    """
    if impulse_len <= 0:
        raise ValueError("impulse_len must be positive")

    half_len = impulse_len // 2 + 1
    channels = _extract_channels(img, colorspace)
    rng = np.random.default_rng()
    impulses: list[np.ndarray] = []

    for ch in channels:
        channel = ch.astype(np.float32)
        if remove_dc:
            channel = channel - np.mean(channel)

        spec = _fft.fft2(channel)
        shifted = _fft.fftshift(spec)
        mag = np.abs(shifted)

        bin_idx = _radial_bins(channel.shape, n_bins=half_len, mode=radius_mode)
        profile = _radial_profile_from_bins(mag, bin_idx=bin_idx, n_bins=half_len)

        if smoothing == "hann":
            profile = profile * hann(profile.size).astype(np.float32)
        elif smoothing == "none":
            pass
        else:
            raise ValueError(f"Unknown smoothing: {smoothing}")

        if phase_mode == "min-phase":
            H_half = _minimum_phase_spectrum(profile, n_fft=impulse_len)
        else:
            if phase_mode == "zero":
                phase = np.zeros_like(profile, dtype=np.float32)
            elif phase_mode == "random":
                phase = rng.uniform(-np.pi, np.pi, size=profile.size).astype(np.float32)
            elif phase_mode == "image":
                phase_map = np.angle(shifted)
                phase = _phase_profile_from_bins(
                    phase_map, bin_idx=bin_idx, n_bins=half_len
                )
            elif phase_mode == "spiral":
                phase_map = np.angle(shifted)
                phase = _spiral_phase_profile(phase_map, n_bins=half_len)
            else:
                raise ValueError(f"Unknown phase_mode: {phase_mode}")

            H_half = profile.astype(np.float32) * np.exp(1j * phase)
        h = _fft.irfft(H_half, n=impulse_len).astype(np.float32)

        h = normalize_impulse(h, impulse_norm)

        impulses.append(h)

    return _stack_impulses(impulses)


# ------------------------------
# Shared convolution core
# ------------------------------
def _convolve_with_impulse(
    audio: np.ndarray,
    sr: int,
    h: np.ndarray,
    *,
    pad_mode: PadMode = "same-center",
    out_norm: OutNorm = "match_rms",
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Convolve mono/stereo audio with a given impulse h (mono or stereo).
    """
    x = np.asarray(audio, dtype=np.float32)

    # ensure 2D (N, C)
    if x.ndim == 1:
        x = x[:, None]
        mono_input = True
    elif x.ndim == 2:
        mono_input = False
    else:
        raise ValueError(f"audio must be 1D or 2D, got shape {x.shape}")

    n_samples, n_channels = x.shape
    h = np.asarray(h, dtype=np.float32)
    if h.ndim == 1:
        h = h[:, None]
    elif h.ndim != 2:
        raise ValueError(f"impulse must be 1D or 2D, got shape {h.shape}")

    if h.size == 0:
        # degenerate: no impulse -> passthrough
        y = x.copy()
        if mono_input:
            y = y[:, 0]
        return y, sr, h

    h_len, h_channels = h.shape
    out_channels = max(n_channels, h_channels)

    # convolution in frequency domain (channel-wise)
    x_len = n_samples
    n_full = x_len + h_len - 1

    # simple FFT size: next power of two
    n_fft = int(2 ** np.ceil(np.log2(max(n_full, 1))))

    # precompute impulse spectra per output channel (repeat if needed)
    H = []
    for c in range(out_channels):
        h_idx = min(c, h_channels - 1)
        H.append(_fft.rfft(h[:, h_idx], n=n_fft).astype(np.complex64))

    y_ch = []

    for c in range(out_channels):
        x_idx = min(c, n_channels - 1)
        X = _fft.rfft(x[:, x_idx], n=n_fft).astype(np.complex64)
        Y = X * H[c]
        y_full = _fft.irfft(Y, n=n_fft).astype(np.float32)
        y_ch.append(y_full[:n_full])

    y = np.stack(y_ch, axis=-1) if out_channels > 1 else y_ch[0][:, None]

    # crop according to pad_mode
    if pad_mode == "full":
        pass
    elif pad_mode == "same-center":
        if n_full <= x_len:
            y = y[:x_len]
        else:
            start = (n_full - x_len) // 2
            end = start + x_len
            y = y[start:end]
    elif pad_mode == "same-first":
        y = y[:x_len]
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")

    # loudness normalization relative to input
    y = normalize_to_reference(y, x, out_norm)

    # collapse back to mono if the output stayed mono
    if mono_input and y.shape[1] == 1:
        y = y[:, 0]

    h_return = h[:, 0] if h.shape[1] == 1 else h
    return y.astype(np.float32), sr, h_return.astype(np.float32)


# ------------------------------
# Public functions
# ------------------------------
def image2sound_flat(
    audio: np.ndarray,
    sr: int,
    img: np.ndarray,
    *,
    pad_mode: PadMode = "same-center",
    colorspace: ColorMode = "luma",
    impulse_norm: ImpulseNorm = "energy",
    out_norm: OutNorm = "match_rms",
    max_impulse_len: int | None = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Convolve audio with an impulse response derived from an image
    by flattening the grayscale values.
    """
    h = _image_to_impulse_flat(
        img,
        colorspace=colorspace,
        impulse_norm=impulse_norm,
        max_len=max_impulse_len,
        remove_dc=True,
    )
    return _convolve_with_impulse(audio, sr, h, pad_mode=pad_mode, out_norm=out_norm)


def image2sound_hist(
    audio: np.ndarray,
    sr: int,
    img: np.ndarray,
    *,
    pad_mode: PadMode = "same-center",
    colorspace: ColorMode = "luma",
    impulse_norm: ImpulseNorm = "energy",
    out_norm: OutNorm = "match_rms",
    n_bins: int = 256,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Convolve audio with an impulse response derived from the
    grayscale histogram of the image.

    The impulse length is `n_bins`. Shape encodes how many pixels
    live in each intensity bucket.
    """
    h = _image_to_impulse_hist(
        img,
        colorspace=colorspace,
        n_bins=n_bins,
        impulse_norm=impulse_norm,
        remove_dc=False,  # usually keep histogram all-positive
    )
    return _convolve_with_impulse(audio, sr, h, pad_mode=pad_mode, out_norm=out_norm)


def image2sound_radial(
    audio: np.ndarray,
    sr: int,
    img: np.ndarray,
    *,
    pad_mode: PadMode = "same-center",
    colorspace: ColorMode = "luma",
    impulse_len: int | Literal["auto"] = 8192,
    radius_mode: RadiusMode = "linear",
    phase_mode: PhaseMode = "zero",
    smoothing: SmoothingMode = "hann",
    impulse_norm: ImpulseNorm = "energy",
    out_norm: OutNorm = "match_rms",
    remove_dc: bool = True,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Convolve audio with an impulse derived from the radial average of the image FFT.
    """
    if impulse_len == "auto":
        impulse_len = int(np.asarray(audio).shape[0])

    h = _image_to_impulse_radial(
        img,
        colorspace=colorspace,
        impulse_len=impulse_len,
        radius_mode=radius_mode,
        phase_mode=phase_mode,
        smoothing=smoothing,
        remove_dc=remove_dc,
        impulse_norm=impulse_norm,
    )
    return _convolve_with_impulse(audio, sr, h, pad_mode=pad_mode, out_norm=out_norm)


__all__ = ["image2sound_flat", "image2sound_hist", "image2sound_radial"]
