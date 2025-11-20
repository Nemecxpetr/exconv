# tests/test_conv2d_image.py
import numpy as np

from exconv.conv2d.image import pair_convolve
from exconv.conv2d.kernels import (
    gaussian_separable,
    separable_conv2d,
)


def test_gaussian_blur_matches_separable_reference():
    rng = np.random.default_rng(42)

    # Random grayscale "image"
    H, W = 64, 48
    img = rng.random((H, W), dtype=np.float64)

    # Separable Gaussian kernels (1D)
    sigma = 1.5
    k_row, k_col = gaussian_separable(sigma=sigma, truncate=3.0, normalize=True)

    # Spatial reference: separable 2D convolution (same-center)
    ref = separable_conv2d(img, k_row, k_col)

    # Build equivalent 2D kernel
    kernel2d = np.outer(k_row, k_col)

    # Frequency-domain convolution: pair_convolve in luma mode (2D input)
    y_fft = pair_convolve(
        img,
        kernel2d,
        mode="same-center",
        circular=False,
        colorspace="luma",  # 2D input => treated as scalar field
    )

    assert y_fft.shape == img.shape == ref.shape

    # For a well-behaved Gaussian and same-center cropping,
    # FFT and spatial separable convolutions should match closely.
    max_err = np.max(np.abs(y_fft - ref))
    assert max_err < 1e-6, f"Max error too large: {max_err}"
