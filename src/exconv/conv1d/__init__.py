"""
exconv.conv1d
=============

1D convolution utilities (primarily for audio).

Submodules
----------
- :mod:`exconv.conv1d.audio` : Audio container and 1D convolution helpers.
"""

from .audio import (
    Audio,
    AudioConvolutionResult,
    auto_convolve,
    pair_convolve,
    multi_convolve,
    convolution_family,
)

__all__ = [
    "Audio",
    "AudioConvolutionResult",
    "auto_convolve",
    "pair_convolve",
    "multi_convolve",
    "convolution_family",
]
