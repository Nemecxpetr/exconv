"""
Tests for v2 exconv.xmodal.sound2image.spectral_sculpt

We test the documented semantics of:

- mode: {"mono", "stereo", "mid-side"}
- colorspace: {"luma", "color"}

without relying on disk assets.
"""

from __future__ import annotations

import numpy as np
import pytest


from exconv.io import rgb_to_luma 
from exconv.xmodal.sound2image import spectral_sculpt, _rgb_to_ycbcr


SR = 44100


# ---------------------------------------------------------------------------
# Small synthetic helpers
# ---------------------------------------------------------------------------

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
    Its rFFT magnitude is a strong low-pass curve after normalization.
    """
    return np.ones(N, dtype=np.float32)


def _audio_mono_impulse(N: int = 4096) -> np.ndarray:
    """
    Impulse at n=0: [1,0,0,...].
    Its rFFT magnitude is all ones → identity radial filter.
    """
    x = np.zeros(N, dtype=np.float32)
    x[0] = 1.0
    return x


def _audio_stereo_equal(N: int = 4096) -> np.ndarray:
    """
    Stereo where L == R == constant.
    For this config, mono, mid, and L/R are all the same curve.
    """
    mono = _audio_mono_constant(N)
    return np.stack([mono, mono], axis=-1)


def _audio_stereo_asymmetric(N: int = 4096) -> np.ndarray:
    """
    Stereo with different left/right spectra so 'stereo' / 'mid-side'
    can actually diverge from 'mono'.
    """
    t = np.linspace(0.0, 1.0, N, endpoint=False, dtype=np.float32)
    L = np.sin(2 * np.pi * 3 * t).astype(np.float32)
    R = np.sin(2 * np.pi * 9 * t).astype(np.float32)
    return np.stack([L, R], axis=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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
    assert np.all(np.isfinite(out))


def test_luma_modes_equivalent_when_channels_identical():
    """
    If L == R (no stereo width), then mono / stereo / mid-side
    should produce the same luminance result (up to tiny numerical noise).
    """
    img = _dummy_image_rgb()
    stereo = _audio_stereo_equal()

    # mono: feed one channel
    out_mono = spectral_sculpt(
        image=img,
        audio=stereo[:, 0],
        sr=SR,
        mode="mono",
        colorspace="luma",
        normalize=True,
    )

    # stereo: L/R but for luma only the mono curve is used
    out_stereo = spectral_sculpt(
        image=img,
        audio=stereo,
        sr=SR,
        mode="stereo",
        colorspace="luma",
        normalize=True,
    )

    # mid-side: mid == mono, side == 0 ⇒ same as mono for luma
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

    # We allow small numerical discrepancies but no visible differences.
    assert max_abs_diff(out_mono, out_stereo) < 1e-5
    assert max_abs_diff(out_mono, out_mid) < 1e-5


def test_color_mono_affects_luma_more_than_chroma():
    """
    For colorspace='color' and mode='mono', only Y should be filtered,
    while Cb/Cr remain unchanged (up to roundtrip conversion noise).
    We verify that Y changes much more than Cb/Cr.
    """
    img = _dummy_image_rgb()
    audio = _audio_mono_constant()

    # Baseline YCbCr of the input
    ycbcr_in = _rgb_to_ycbcr(img.astype(np.float32))
    Y0 = ycbcr_in[..., 0]
    Cb0 = ycbcr_in[..., 1]
    Cr0 = ycbcr_in[..., 2]

    out = spectral_sculpt(
        image=img,
        audio=audio,
        sr=SR,
        mode="mono",
        colorspace="color",
        normalize=False,  # avoid extra 0..1 clipping for this relational test
    )

    ycbcr_out = _rgb_to_ycbcr(out.astype(np.float32))
    Y1 = ycbcr_out[..., 0]
    Cb1 = ycbcr_out[..., 1]
    Cr1 = ycbcr_out[..., 2]

    Y_diff = np.mean(np.abs(Y1 - Y0))
    Cb_diff = np.mean(np.abs(Cb1 - Cb0))
    Cr_diff = np.mean(np.abs(Cr1 - Cr0))

    # Y should change noticeably (low-pass blur)
    assert Y_diff > 1e-3

    # Chroma should be much more stable than Y
    assert Cb_diff < Y_diff * 0.25
    assert Cr_diff < Y_diff * 0.25


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

    # The stereo-aware color mapping should not collapse to mono behavior.
    assert mean_diff > 1e-5 or max_diff > 1e-3

def test_impulse_audio_is_near_identity_filter():
    """
    Impulse audio (1,0,0,...) ⇒ flat magnitude spectrum ⇒ radial filter ~1.

    For colorspace='color':
        - The full RGB image should be very close to identity.

    For colorspace='luma':
        - The *luminance* of the output should be very close to the
          luminance of the input. RGB can legitimately change because
          v2 intentionally converts to luma and back (grayscale RGB).
    """
    img = _dummy_image_rgb()
    audio = _audio_mono_impulse()

    # Baseline luminance of the input for the luma checks
    baseline_luma = rgb_to_luma(img.astype(np.float32))

    for mode in ("mono", "stereo", "mid-side"):
        for colorspace in ("luma", "color"):
            out = spectral_sculpt(
                image=img,
                audio=audio,
                sr=SR,
                mode=mode,
                colorspace=colorspace,
                normalize=False,  # avoid extra global scaling
            )

            # For luma mode + RGB input, spectral_sculpt returns RGB
            # (luma expanded back). For 2D gray we would get 2D.
            assert out.shape[:2] == img.shape[:2]
            assert np.all(np.isfinite(out))

            if colorspace == "color":
                # Full RGB should be almost identical
                diff = np.abs(out.astype(np.float32) - img.astype(np.float32))
                mean_diff = float(diff.mean())
                max_diff = float(diff.max())
                assert mean_diff < 1e-3
                assert max_diff < 5e-3
            else:  # colorspace == "luma"
                # Compare only luminance, since v2 intentionally discards
                # original chroma and returns grayscale RGB.
                out_luma = rgb_to_luma(out.astype(np.float32))
                diff_luma = np.abs(out_luma - baseline_luma)
                mean_diff_luma = float(diff_luma.mean())
                max_diff_luma = float(diff_luma.max())
                assert mean_diff_luma < 1e-3
                assert max_diff_luma < 5e-3


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
