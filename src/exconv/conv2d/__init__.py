"""
exconv.conv2d
=============

2D convolution utilities for images and scalar fields.

Submodules
----------
- :mod:`exconv.conv2d.image`   : Image auto/pair convolution via 2D FFT.
- :mod:`exconv.conv2d.kernels` : Common 2D kernels (Gaussian, Laplacian, Gabor, ...).
- :mod:`exconv.conv2d.color`   : Simple color/gamma utilities.
"""

from .image import auto_convolve as image_auto_convolve, pair_convolve as image_pair_convolve
from .kernels import (
    gaussian_1d,
    gaussian_2d,
    gaussian_separable,
    laplacian_3x3,
    gabor_kernel,
    separable_conv2d,
)
from .color import apply_gamma

__all__ = [
    # high-level image conv
    "image_auto_convolve",
    "image_pair_convolve",
    # kernels
    "gaussian_1d",
    "gaussian_2d",
    "gaussian_separable",
    "laplacian_3x3",
    "gabor_kernel",
    "separable_conv2d",
    # color utilities
    "apply_gamma",
]
