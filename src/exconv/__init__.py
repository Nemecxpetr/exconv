"""
exconv
Experimental convolution toolkit.
"""

# Robust version detection that works even when not installed
try:
    from importlib import metadata as _metadata  # 3.8+
except Exception:  # very old Pythons (not your case), or environment quirks
    _metadata = None  # type: ignore

try:
    __version__ = _metadata.version("exconv") if _metadata else "0.0.0.dev0"
except Exception:
    # Not installed (dev mode) or no metadata available
    __version__ = "0.0.0.dev0"

# Re-export core for convenience
from . import core, io, conv1d, conv2d  # noqa: E402

__all__ = ["core", "io", "conv1d", "conv2d", "__version__"]

