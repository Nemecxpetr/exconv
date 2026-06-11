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
    AudioSpectralProcessing,
    auto_convolve,
    convolution_family,
    multi_convolve,
    pair_convolve,
)

__all__ = [
    "Audio",
    "AudioConvolutionResult",
    "AudioSpectralProcessing",
    "auto_convolve",
    "pair_convolve",
    "multi_convolve",
    "convolution_family",
]
