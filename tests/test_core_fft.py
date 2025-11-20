# tests/test_core_fft.py
import numpy as np
import tests.conftest as conftest  # noqa: F401 to set up src/ in sys.path

from exconv.core.fft import (
    next_fast_len_ge,
    fftnd,
    ifftnd,
    linear_freq_multiply,
    make_hermitian_symmetric_unshifted,
)
from exconv.core.grids import freq_grid_2d, radial_grid_2d, hann, tukey


def test_next_fast_len_ge_monotonic():
    for n in [1, 2, 3, 7, 16, 31, 100, 257, 1024, 12345]:
        m = next_fast_len_ge(n)
        assert m >= n
        # should be non-decreasing for n+1
        m2 = next_fast_len_ge(n + 1)
        assert m2 >= m or m2 == m  # allow equal or larger


def _circular_reference(x, h, axes):
    # Reference circular convolution using FFT with size = x.shape along axes
    n = [x.shape[a] for a in axes]
    X = np.fft.fftn(x, s=n, axes=axes)
    H = np.fft.fftn(h, s=n, axes=axes)
    y = np.fft.ifftn(X * H, axes=axes)
    return y


def test_circular_matches_reference_1d():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128)
    h = rng.standard_normal(64)
    y = linear_freq_multiply(x, h, axes=0, mode="circular")
    y_ref = _circular_reference(x, h, axes=(0,))
    np.testing.assert_allclose(y, y_ref.real, rtol=1e-10, atol=1e-10)


def test_circular_matches_reference_2d():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 24))
    h = rng.standard_normal((32, 24))
    y = linear_freq_multiply(x, h, axes=(0, 1), mode="circular")
    y_ref = _circular_reference(x, h, axes=(0, 1))
    np.testing.assert_allclose(y, y_ref.real, rtol=1e-10, atol=1e-10)


def test_linear_full_vs_same_shapes():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((20, 10))
    h = rng.standard_normal((7, 5))
    y_full = linear_freq_multiply(x, h, axes=(0, 1), mode="full")
    assert y_full.shape == (20 + 7 - 1, 10 + 5 - 1)
    y_same_first = linear_freq_multiply(x, h, axes=(0, 1), mode="same-first")
    assert y_same_first.shape == x.shape
    y_same_center = linear_freq_multiply(x, h, axes=(0, 1), mode="same-center")
    assert y_same_center.shape == x.shape


def test_hermitian_symmetry_produces_real_ifft():
    rng = np.random.default_rng(3)
    F = rng.standard_normal((33, 20)) + 1j * rng.standard_normal((33, 20))
    F_sym = make_hermitian_symmetric_unshifted(F, axes=(0, 1))
    y = np.fft.ifftn(F_sym, axes=(0, 1))
    assert np.max(np.abs(y.imag)) < 1e-10  # numerically real
    # also ensure symmetry holds numerically (X[k] ~ conj(X[-k]))
    # also ensure symmetry holds numerically (X[k] ~ conj(X[-k mod N]))
    Xf = F_sym

    def modneg_conj(A, axes):
        B = A
        for ax in axes:
            N = B.shape[ax]
            idx = np.empty(N, dtype=int)
            idx[0] = 0
            if N > 1:
                idx[1:] = np.arange(N - 1, 0, -1)
            B = np.take(B, idx, axis=ax)
        return np.conj(B)

    X_modconj = modneg_conj(Xf, axes=(0, 1))
    np.testing.assert_allclose(Xf, X_modconj, rtol=1e-12, atol=1e-12)

    y = np.fft.ifftn(F_sym, axes=(0, 1))
    assert np.max(np.abs(y.imag)) < 1e-10



def test_grids_and_windows():
    H, W = 17, 12
    k1, k2 = freq_grid_2d(H, W)
    assert k1.shape == (H, W) and k2.shape == (H, W)
    rho = radial_grid_2d(H, W)
    assert rho.min() >= 0 and rho.max() <= 1 + 1e-12
    w_hann = hann(128)
    w_tuk = tukey(128, alpha=0.5)
    assert w_hann.shape == (128,)
    assert w_tuk.shape == (128,)
    # window edges are deterministic
    np.testing.assert_allclose(w_hann[[0, -1]], 0.0, atol=1e-12)
    assert w_tuk.min() >= 0.0
    assert w_tuk.max() <= 1.0
