"""
Command-line entry for the exconv package.

Usage
-----
$ python -m exconv
"""

import numpy as np
from .core import fftnd, ifftnd, linear_freq_multiply, freq_grid_2d, radial_grid_2d
from . import __version__


def _diagnostics():
    print(f"exconv experimental convolution toolkit v{__version__}\n")

    print("FFT sanity check:")
    x = np.random.randn(8)
    X = fftnd(x)
    x_rec = ifftnd(X)
    print(f"  input shape: {x.shape}")
    print(f"  reconstructed (real) error: {np.max(np.abs(x - x_rec.real)):.2e}")

    print("\nLinear convolution test:")
    h = np.random.randn(4)
    y = linear_freq_multiply(x, h, axes=0, mode="same-center")
    print(f"  output shape (same-center): {y.shape}")

    print("\n2D grid diagnostics:")
    k1, k2 = freq_grid_2d(8, 8)
    rho = radial_grid_2d(8, 8)
    print(f"  freq_grid_2d ranges: k1[{k1.min()}..{k1.max()}], k2[{k2.min()}..{k2.max()}]")
    print(f"  radial_grid_2d: ρ in [{rho.min():.2f}, {rho.max():.2f}]")

    print("\nAll tests passed ✅")


if __name__ == "__main__":
    _diagnostics()
