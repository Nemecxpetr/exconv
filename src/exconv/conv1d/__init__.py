"""
exconv.conv1d
=============

1D convolution utilities (primarily for audio).

Submodules
----------
- :mod:`exconv.conv1d.audio` : Audio container and 1D convolution helpers.
"""

from .audio import Audio, auto_convolve, pair_convolve

__all__ = [
    "Audio",
    "auto_convolve",
    "pair_convolve",
]
