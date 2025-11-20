"""
exconv.io
=========

Unified IO helpers for audio and image data.

This module exposes:
- Audio read/write, channel conversion, and segment extraction
- Image read/write, dtype conversion, and RGB↔luminance transforms

Submodules:
- exconv.io.audio
- exconv.io.image
"""

from .audio import (
    read_audio,
    read_segment,
    write_audio,
    to_mono,
    to_stereo,
)

from .image import (
    read_image,
    write_image,
    as_float32,
    as_uint8,
    rgb_to_luma,
    luma_to_rgb,
)

__all__ = [
    # audio
    "read_audio",
    "read_segment",
    "write_audio",
    "to_mono",
    "to_stereo",
    # image
    "read_image",
    "write_image",
    "as_float32",
    "as_uint8",
    "rgb_to_luma",
    "luma_to_rgb",
]
