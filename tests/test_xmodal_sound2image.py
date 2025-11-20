"""
Tests for v2 exconv.xmodal.sound2image.spectral_sculpt

We test the documented semantics of:

- mode: {"mono", "stereo", "mid-side"}
- colorspace: {"luma", "color"}

All tests use tiny in-memory arrays (no disk IO).
"""

from __future__ import annotations

import numpy as np
import pytest

from exconv.io import rgb_to_luma
from exconv.xmodal.sound2image import spectral_sculpt

SR = 44100  # sample rate placeholder for tests


# ---------------------------------------------------------------------
# Helpers: tiny synthetic images & audio
# ---------------------------------------------------------------------


def _dummy_image_rgb(H: int = 32, W: int = 32) -> np.ndarray:
    """
    Simple RGB test image: horizontal red, vertical green, diagonal blue.
    Values in [0,1].
    """
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]

    r = np.broadcast_to(x, (H, W))
    g = np.broadcast_to(y, (H, W))
    b = 0.5 * (r + g)

    img = np.stack([r, g, b], axis=-1)
    return img.astype(np.float32)


def _dummy_image_gray(H: int = 32, W: int = 32) -> np.ndarray:
    """
    Simple grayscale gradient image in [0,1].
    """
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    img = 0.7 * xx + 0.3 * yy
    return img.astype(np.float32)


def _audio_mono_constant(N: int = 4096) -> np.ndarray:
    """
    Constant mono signal [1,1,...,1].
    Its rFFT magnitude becomes a low-pass-ish curve after smoothing.
    """
    return np.ones(N, dtype=np.float32)


def _audio_mono_impulse(N: int = 4096) -> np.ndarray:
    """
    Impulse at n=0: [1,0,0,...].
    Its rFFT magnitude is all ones ⇒ identity radial filter.
    """
    x = np.zeros(N, dtype=np.float32)
    x[0] = 1.0
    return x


def _audio_stereo_equal(N: int = 4096) -> np.ndarray:
    """
    Stereo where L == R == constant.
    For this config, mono, mid, and L/R all have the same spectrum.
    """
    mono = _audio_mono_constant(N)
    return np.stack([mono, mono], axis=-1)


def _audio_stereo_asymmetric(N: int = 4096) -> np.ndarray:
    """
    Stereo with different left/right spectra so 'stereo' / 'mid-side'
    can diverge from 'mono'.
    """
    t = np.linspace(0.0, 1.0, N, endpoint=False, dtype=np.float32)
    L = np.sin(2 * np.pi * 3 * t).astype(np.float32)
    R = np.sin(2 * np.pi * 9 * t).astype(np.float32)
    return np.stack([L, R], axis=-1)


# ---------------------------------------------------------------------
# Basic shapes / range
# ---------------------------------------------------------------------


def test_mono_luma_gray_returns_2d():
    """
    Gray input + colorspace='luma' should return a 2D array
    with the same shape (no RGB expansion).
    """
    img = _dummy_image_gray()
    audio = _audio_mono_constant()

    out = spectral_sculpt(
        image=img,
        audio=audio,
        sr=SR,
        mode="mono",
        colorspace="luma",
        normalize=True,
    )

    assert out.ndim == 2
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_mono_color_rgb_returns_rgb():
    """
    RGB input + colorspace='color' should return RGB (H,W,3) in [0,1].
    """
    img = _dummy_image_rgb()
    audio = _audio_mono_constant()

    out = spectral_sculpt(
        image=img,
        audio=audio,
        sr=SR,
        mode="mono",
        colorspace="color",
        normalize=True,
    )

    assert out.ndim == 3
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------
# Mode semantics (mono / stereo / mid-side)
# ---------------------------------------------------------------------


def test_luma_modes_equivalent_when_LR_identical():
    """
    If L == R (no stereo width), then mono / stereo / mid-side
    should produce very similar luminance outputs.
    """
    img = _dummy_image_rgb()
    stereo = _audio_stereo_equal()

    out_mono = spectral_sculpt(
        image=img,
        audio=stereo[:, 0],
        sr=SR,
        mode="mono",
        colorspace="luma",
        normalize=True,
    )
    out_stereo = spectral_sculpt(
        image=img,
        audio=stereo,
        sr=SR,
        mode="stereo",
        colorspace="luma",
        normalize=True,
    )
    out_mid = spectral_sculpt(
        image=img,
        audio=stereo,
        sr=SR,
        mode="mid-side",
        colorspace="luma",
        normalize=True,
    )

    assert out_mono.shape == out_stereo.shape == out_mid.shape

    def max_abs_diff(a, b) -> float:
        return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))

    assert max_abs_diff(out_mono, out_stereo) < 1e-5
    assert max_abs_diff(out_mono, out_mid) < 1e-5


def test_color_stereo_differs_from_mono_when_LR_asymmetric():
    """
    With asymmetric L/R spectra and colorspace='color', mode='stereo'
    should give a different result than mode='mono'.
    """
    img = _dummy_image_rgb()
    audio_mono = _audio_mono_constant()
    audio_stereo = _audio_stereo_asymmetric()

    out_mono = spectral_sculpt(
        image=img,
        audio=audio_mono,
        sr=SR,
        mode="mono",
        colorspace="color",
        normalize=True,
    )

    out_stereo = spectral_sculpt(
        image=img,
        audio=audio_stereo,
        sr=SR,
        mode="stereo",
        colorspace="color",
        normalize=True,
    )

    diff = np.abs(out_mono.astype(np.float32) - out_stereo.astype(np.float32))
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())

    # Stereo-aware mapping should not collapse to mono behavior.
    assert mean_diff > 1e-6 or max_diff > 1e-4


# ---------------------------------------------------------------------
# Impulse → near-identity behavior
# ---------------------------------------------------------------------
def test_impulse_audio_identity_luma_mono():
    """
    Impulse audio (1,0,0,...) ⇒ flat magnitude spectrum.
    For colorspace='luma' and RGB input, result should be very close
    to the baseline luma of the image.
    """
    img = _dummy_image_rgb()
    audio = _audio_mono_impulse()

    baseline_luma = rgb_to_luma(img.astype(np.float32))

    out = spectral_sculpt(
        image=img,
        audio=audio,
        sr=SR,
        mode="mono",
        colorspace="luma",
        normalize=False,  # we compare directly to baseline_luma
    )

    # Output is RGB for RGB input, even in luma mode
    assert out.shape == img.shape

    out_luma = rgb_to_luma(out.astype(np.float32))
    diff = np.abs(out_luma - baseline_luma)
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())

    # Should be visually almost identical
    assert mean_diff < 1e-3
    assert max_diff < 5e-3

def test_impulse_audio_identity_color_mono():
    """
    Impulse audio for colorspace='color' should be close to identity
    on RGB: radial filter ~1 in all channels.
    """
    img = _dummy_image_rgb()
    audio = _audio_mono_impulse()

    out = spectral_sculpt(
        image=img,
        audio=audio,
        sr=SR,
        mode="mono",
        colorspace="color",
        normalize=False,
    )

    assert out.shape == img.shape
    diff = np.abs(out.astype(np.float32) - img.astype(np.float32))
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())

    assert mean_diff < 1e-3
    assert max_diff < 5e-3


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------


def test_invalid_mode_and_colorspace_raise():
    img = _dummy_image_gray()
    audio = _audio_mono_constant()

    with pytest.raises(ValueError):
        _ = spectral_sculpt(
            image=img,
            audio=audio,
            sr=SR,
            mode="definitely-not-a-mode",
            colorspace="luma",
        )

    with pytest.raises(ValueError):
        _ = spectral_sculpt(
            image=img,
            audio=audio,
            sr=SR,
            mode="mono",
            colorspace="not-a-colorspace",
        )
