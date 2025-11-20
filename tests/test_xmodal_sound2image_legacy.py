# tests/test_xmodal_sound2image.py
from __future__ import annotations

import numpy as np
import pytest
pytestmark = pytest.mark.skip("Legacy v1 sound2image tests — API replaced by v2")


from exconv.xmodal.sound2image import spectral_sculpt


# ---------------------------------------------------------------------
# Fixtures / dummy data
# ---------------------------------------------------------------------

def _dummy_image_gray(H: int = 64, W: int = 64) -> np.ndarray:
    """Simple gradient image for deterministic tests."""
    y, x = np.mgrid[0:H, 0:W]
    img = (x + y).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return img


def _dummy_image_rgb(H: int = 64, W: int = 64) -> np.ndarray:
    """RGB variant with simple, nontrivial color structure."""
    g = _dummy_image_gray(H, W)
    return np.stack(
        [
            g,                 # R: gradient
            1.0 - g,           # G: inverted
            0.5 * np.ones_like(g),  # B: flat mid-gray
        ],
        axis=-1,
    ).astype(np.float32)


def _dummy_stereo_audio(N: int = 4096, sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Simple stereo audio:
    L: 440 Hz sine
    R: 660 Hz sine
    """
    t = np.arange(N, dtype=np.float64) / sr
    L = np.sin(2.0 * np.pi * 440.0 * t)
    R = np.sin(2.0 * np.pi * 660.0 * t)
    stereo = np.stack([L, R], axis=1).astype(np.float32)
    return stereo, sr


def _dummy_mono_audio(N: int = 4096, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Plain mono sine for fallback tests."""
    t = np.arange(N, dtype=np.float64) / sr
    x = np.sin(2.0 * np.pi * 440.0 * t)
    return x.astype(np.float32), sr


# ---------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------

@pytest.mark.parametrize("colorspace", ["luma", "channels"])
def test_spectral_sculpt_collapse_basic(colorspace: str):
    img = _dummy_image_rgb()
    audio, sr = _dummy_stereo_audio()
    out = spectral_sculpt(
        img,
        audio,
        sr,
        colorspace=colorspace,
        stereo_mode="collapse",
    )

    assert out.shape == img.shape
    assert np.isfinite(out).all()
    # normalized output should be in 0..1-ish
    assert out.min() >= -1e-6
    assert out.max() <= 1.0 + 1e-6


def test_mid_side_color_rgb_channels():
    img = _dummy_image_rgb()
    audio, sr = _dummy_stereo_audio()

    out = spectral_sculpt(
        img,
        audio,
        sr,
        colorspace="channels",
        stereo_mode="mid-side-color",
        beta=0.5,
    )

    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_lr_color_rgb_channels():
    img = _dummy_image_rgb()
    audio, sr = _dummy_stereo_audio()

    out = spectral_sculpt(
        img,
        audio,
        sr,
        colorspace="channels",
        stereo_mode="lr-color",
        beta=0.5,
    )

    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_mid_side_angular_luma_rgb_input():
    img = _dummy_image_rgb()
    audio, sr = _dummy_stereo_audio()

    out = spectral_sculpt(
        img,
        audio,
        sr,
        colorspace="luma",
        stereo_mode="mid-side-angular",
        alpha=0.8,
    )

    # luma mode on RGB → output RGB
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_mid_side_angular_gray_input():
    """Gray image + angular mode should keep 2D shape."""
    img = _dummy_image_gray()
    audio, sr = _dummy_stereo_audio()

    out = spectral_sculpt(
        img,
        audio,
        sr,
        colorspace="luma",
        stereo_mode="mid-side-angular",
        alpha=0.8,
    )

    assert out.shape == img.shape
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------
# Fallback behaviors
# ---------------------------------------------------------------------

def test_mono_fallback_matches_collapse():
    """
    When audio is mono and stereo_mode != "collapse",
    implementation should gracefully fall back to collapse.
    Here we check that explicitly for mid-side-color.
    """
    img = _dummy_image_rgb()
    mono, sr = _dummy_mono_audio()

    out_collapse = spectral_sculpt(
        img,
        mono,
        sr,
        colorspace="luma",
        stereo_mode="collapse",
    )
    out_mid_side = spectral_sculpt(
        img,
        mono,
        sr,
        colorspace="luma",
        stereo_mode="mid-side-color",
    )

    assert np.allclose(out_collapse, out_mid_side, atol=1e-5)


def test_invalid_stereo_mode_raises():
    img = _dummy_image_gray()
    audio, sr = _dummy_stereo_audio()

    with pytest.raises(ValueError):
        _ = spectral_sculpt(
            img,
            audio,
            sr,
            stereo_mode="definitely-not-a-mode",
        )
