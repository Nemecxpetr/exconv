import numpy as np

from exconv.io import rgb_to_luma
from exconv.xmodal.sound2image import (
    _rgb_to_ycbcr,
    _ycbcr_to_rgb,
    ImageSculptor,
    StereoFilterSpec,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _rand_rgb(shape=(16, 16, 3), seed=0):
    rng = np.random.default_rng(seed)
    # in [0,1]
    return rng.random(shape, dtype=np.float32)


def _gray_rgb(shape=(16, 16), seed=1):
    """
    Gray RGB where r = g = b (for testing chroma behaviour).
    """
    rng = np.random.default_rng(seed)
    g = rng.random(shape, dtype=np.float32)
    return np.stack([g, g, g], axis=-1)


# ---------------------------------------------------------------------
# 1. RGB <-> YCbCr conversion tests
# ---------------------------------------------------------------------

def test_rgb_ycbcr_roundtrip_close_to_identity():
    """
    _rgb_to_ycbcr followed by _ycbcr_to_rgb should be close
    to identity for RGB in [0,1].
    """
    rgb = _rand_rgb()
    ycbcr = _rgb_to_ycbcr(rgb)
    rgb_rec = _ycbcr_to_rgb(ycbcr)

    assert rgb_rec.shape == rgb.shape
    # Allow small numerical error
    err = np.max(np.abs(rgb - rgb_rec))
    assert err < 1e-5


def test_rgb_to_ycbcr_gray_has_neutral_chroma():
    """
    For gray RGB (r=g=b), Y should be ~that value and Cb/Cr ~ 0.5.
    """
    rgb = _gray_rgb()
    ycbcr = _rgb_to_ycbcr(rgb)

    Y = ycbcr[..., 0]
    Cb = ycbcr[..., 1]
    Cr = ycbcr[..., 2]

    # Y should match the gray level (up to numerical tolerance)
    gray = rgb[..., 0]
    assert np.allclose(Y, gray, atol=1e-5)

    # Neutral gray should give approximately 0.5 for both chroma channels
    assert np.allclose(Cb, 0.5, atol=1e-5)
    assert np.allclose(Cr, 0.5, atol=1e-5)


def test_ycbcr_to_rgb_roundtrip_close_to_identity():
    """
    Check that feeding back YCbCr produced from RGB gives us back the original
    (this is similar to the first test, but exercises the inverse direction explicitly).
    """
    rgb = _rand_rgb()
    # Build YCbCr manually then back
    ycbcr = _rgb_to_ycbcr(rgb)
    rgb2 = _ycbcr_to_rgb(ycbcr)

    err = np.max(np.abs(rgb - rgb2))
    assert err < 1e-5


# ---------------------------------------------------------------------
# 2. ImageSculptor + StereoFilterSpec: identity when H_luma == 1
# ---------------------------------------------------------------------

def test_image_sculptor_channels_identity_filter_rgb():
    """
    For colorspace='channels' and H_luma == 1 everywhere, with normalize=False,
    ImageSculptor should act as an identity (FFT * 1 in freq-domain).
    """
    rgb = _rand_rgb()
    H, W, _ = rgb.shape

    H_luma = np.ones((H, W), dtype=np.float32)
    filt = StereoFilterSpec(H_luma=H_luma)

    sculptor = ImageSculptor(colorspace="channels", normalize=False)
    out = sculptor.apply(rgb, filt)

    assert out.shape == rgb.shape
    # Since we disable normalize and multiply by 1 in freq domain,
    # we expect nearly perfect reconstruction.
    err = np.max(np.abs(out - rgb))
    assert err < 1e-5


def test_image_sculptor_channels_identity_filter_gray():
    """
    Same as above but for a 2D gray image.
    """
    gray = _rand_rgb(shape=(32, 32, 3)).mean(axis=-1).astype(np.float32)
    H, W = gray.shape
    H_luma = np.ones((H, W), dtype=np.float32)
    filt = StereoFilterSpec(H_luma=H_luma)

    sculptor = ImageSculptor(colorspace="channels", normalize=False)
    out = sculptor.apply(gray, filt)

    assert out.shape == gray.shape
    err = np.max(np.abs(out - gray))
    assert err < 1e-5


# ---------------------------------------------------------------------
# 3. Luma-mode behaviour: relationship to rgb_to_luma
# ---------------------------------------------------------------------

def test_image_sculptor_luma_mode_matches_rgb_to_luma_with_identity_filter():
    """
    In colorspace='luma', with H_luma == 1 and normalize=False:

    - luma = rgb_to_luma(img)
    - filtered luma is identical to luma
    - output should be that luma replicated to RGB.
    """
    rgb = _rand_rgb()
    H, W, _ = rgb.shape

    H_luma = np.ones((H, W), dtype=np.float32)
    filt = StereoFilterSpec(H_luma=H_luma)

    sculptor = ImageSculptor(colorspace="luma", normalize=False)
    out = sculptor.apply(rgb, filt)

    # Output should be RGB, but with all channels equal
    assert out.shape == rgb.shape
    r, g, b = out[..., 0], out[..., 1], out[..., 2]

    # All channels equal
    assert np.allclose(r, g, atol=1e-5)
    assert np.allclose(r, b, atol=1e-5)

    # And equal to rgb_to_luma result (up to numerical error)
    luma_ref = rgb_to_luma(rgb).astype(np.float32)
    assert np.allclose(r, luma_ref, atol=1e-5)


def test_image_sculptor_luma_mode_gray_input_stays_gray():
    """
    If input is already gray (2D) and H_luma == 1,
    output should be almost identical in luma-mode.
    """
    gray_rgb = _gray_rgb(shape=(32, 32))
    gray = gray_rgb[..., 0]  # 2D
    H, W = gray.shape

    H_luma = np.ones((H, W), dtype=np.float32)
    filt = StereoFilterSpec(H_luma=H_luma)

    sculptor = ImageSculptor(colorspace="luma", normalize=False)
    out = sculptor.apply(gray, filt)

    assert out.shape == gray.shape
    assert np.allclose(out, gray, atol=1e-5)


# ---------------------------------------------------------------------
# 4. YCbCr-based stereo color encodings: sanity checks
# ---------------------------------------------------------------------

def test_mid_side_color_uses_chroma_gains_without_exploding():
    """
    Smoke test: using non-trivial chroma gains doesn't explode the range
    after RGB conversion (before global normalization).
    Here we bypass spectral_sculpt and poke ImageSculptor directly.
    """
    rgb = _rand_rgb()
    H, W, _ = rgb.shape

    # Some arbitrary luma filter (identity) and non-trivial gains
    H_luma = np.ones((H, W), dtype=np.float32)
    cb_gain = np.linspace(0.1, 2.0, H * W, dtype=np.float32).reshape(H, W)
    cr_gain = np.linspace(2.0, 0.1, H * W, dtype=np.float32).reshape(H, W)

    filt = StereoFilterSpec(H_luma=H_luma, chroma_cb_gain=cb_gain, chroma_cr_gain=cr_gain)

    # normalize=False to inspect raw behaviour before final 0..1 squeeze
    sculptor = ImageSculptor(colorspace="channels", normalize=False)
    out = sculptor.apply(rgb, filt)

    # We don't care about exact values, but we want them finite and not insane
    assert np.isfinite(out).all()
    # Very loose bounds: should not go beyond, say, -5..5 in raw float
    assert out.min() > -5.0
    assert out.max() < 5.0
