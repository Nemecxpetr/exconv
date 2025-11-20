"""
exconv.core
===========

Low-level computational primitives for experimental convolution.

Submodules
----------
- :mod:`exconv.core.fft`   : FFT utilities and frequency-domain convolution.
- :mod:`exconv.core.grids` : Frequency/spatial grids and windows.
- :mod:`exconv.core.norms` : Basic RMS / peak normalization helpers.
"""

from .fft import (
    next_fast_len_ge,
    pad_to_linear_shape,
    fftnd,
    ifftnd,
    linear_freq_multiply,
    make_hermitian_symmetric_unshifted,
)
from .grids import (
    freq_grid_2d,
    radial_grid_2d,
    hann,
    tukey,
)
from .norms import (
    rms_normalize,
    peak_normalize,
    apply_normalize,
)

__all__ = [
    # fft
    "next_fast_len_ge",
    "pad_to_linear_shape",
    "fftnd",
    "ifftnd",
    "linear_freq_multiply",
    "make_hermitian_symmetric_unshifted",
    # grids
    "freq_grid_2d",
    "radial_grid_2d",
    "hann",
    "tukey",
    # norms
    "rms_normalize",
    "peak_normalize",
    "apply_normalize",
]
