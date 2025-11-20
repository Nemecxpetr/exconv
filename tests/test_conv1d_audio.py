# tests/test_conv1d_audio.py
import numpy as np
import pytest

from src.exconv.conv1d.audio import Audio, auto_convolve, pair_convolve


def test_impulse_self_convolution_full():
    # Impulse at index 3
    x = np.zeros(8, dtype=float)
    x[3] = 1.0
    a = Audio(samples=x, sr=44100)

    out = auto_convolve(a, mode="full", circular=False, normalize=None, order=2)
    expected = np.convolve(x, x)

    assert out.samples.shape == expected.shape
    np.testing.assert_allclose(out.samples, expected, rtol=1e-10, atol=1e-10)
    assert out.sr == a.sr


def test_impulse_self_convolution_same_modes():
    x = np.zeros(8, dtype=float)
    x[3] = 1.0
    a = Audio(samples=x, sr=48000)

    full = np.convolve(x, x)  # length 15

    # same-first
    out_first = auto_convolve(a, mode="same-first", circular=False, normalize=None, order=2)
    expected_first = full[: x.size]
    assert out_first.samples.shape == x.shape
    np.testing.assert_allclose(out_first.samples, expected_first, rtol=1e-10, atol=1e-10)

    # same-center
    out_center = auto_convolve(a, mode="same-center", circular=False, normalize=None, order=2)
    L = full.size
    N = x.size
    start = (L - N) // 2
    expected_center = full[start : start + N]
    assert out_center.samples.shape == x.shape
    np.testing.assert_allclose(out_center.samples, expected_center, rtol=1e-10, atol=1e-10)


def test_sine_burst_len_and_sr():
    sr = 32000
    N = 32
    t = np.arange(N) / sr
    freq = 440.0
    burst = np.sin(2 * np.pi * freq * t)
    # Window
    burst *= np.hanning(N)
    a = Audio(samples=burst, sr=sr)

    # auto_convolve (full)
    out_full = auto_convolve(a, mode="full", circular=False, normalize=None, order=2)
    assert out_full.n_samples == 2 * N - 1
    assert out_full.sr == sr
    assert np.max(np.abs(out_full.samples)) > 0.0

    # pair_convolve same signal (same-first)
    out_same = pair_convolve(a, a, mode="same-first", circular=False, normalize=None)
    assert out_same.n_samples == N
    assert out_same.sr == sr
    assert np.max(np.abs(out_same.samples)) > 0.0

    # circular: length = N
    out_circ = pair_convolve(a, a, mode="same-first", circular=True, normalize=None)
    assert out_circ.n_samples == N
    assert out_circ.sr == sr
    assert np.max(np.abs(out_circ.samples)) > 0.0


def test_stereo_per_channel_convolution():
    # Left impulse at 0, right impulse at 1
    N = 4
    left = np.array([1.0, 0.0, 0.0, 0.0])
    right = np.array([0.0, 1.0, 0.0, 0.0])
    stereo = np.stack([left, right], axis=1)  # (4, 2)
    a = Audio(samples=stereo, sr=44100)

    out = auto_convolve(a, mode="full", circular=False, normalize=None, order=2)

    # Expected per-channel conv
    expected_left = np.convolve(left, left)
    expected_right = np.convolve(right, right)
    assert out.samples.shape == (expected_left.size, 2)
    np.testing.assert_allclose(out.samples[:, 0], expected_left, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(out.samples[:, 1], expected_right, rtol=1e-10, atol=1e-10)


def test_mismatched_channels_downmix_to_mono():
    N = 16
    sr = 44100

    x = np.random.default_rng(0).standard_normal(N)
    h_stereo = np.random.default_rng(1).standard_normal((N, 2))

    a_x = Audio(samples=x, sr=sr)          # mono
    a_h = Audio(samples=h_stereo, sr=sr)   # stereo

    out = pair_convolve(a_x, a_h, mode="full", circular=False, normalize=None)

    # Output should be mono
    assert out.is_mono
    y = np.asarray(out.samples)

    # Manual reference: downmix inputs then convolve
    x_mono = x
    h_mono = h_stereo.mean(axis=1)
    expected = np.convolve(x_mono, h_mono)
    np.testing.assert_allclose(y, expected, rtol=1e-10, atol=1e-10)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x**2)))


def test_normalization_modes():
    sr = 44100
    x = np.array([1.0, -0.5, 0.25, -0.25])
    a = Audio(samples=x, sr=sr)

    # pair convolve with itself
    raw = pair_convolve(a, a, mode="full", circular=False, normalize=None)
    raw_rms = _rms(raw.samples)
    raw_peak = float(np.max(np.abs(raw.samples)))
    assert raw_peak > 0.0
    assert raw_rms > 0.0

    # RMS normalization
    target_rms = 0.1
    out_rms = pair_convolve(a, a, mode="full", circular=False, normalize="rms")
    out_rms_val = _rms(out_rms.samples)
    assert out_rms_val == pytest.approx(0.1, rel=1e-4, abs=1e-6)

    # peak normalization
    out_peak = pair_convolve(a, a, mode="full", circular=False, normalize="peak")
    out_peak_val = float(np.max(np.abs(out_peak.samples)))
    assert out_peak_val == pytest.approx(0.99, rel=1e-4, abs=1e-6)


def test_sample_rate_preservation_and_mismatch():
    sr = 44100
    sr2 = 48000
    x = np.random.default_rng(2).standard_normal(16)
    h = np.random.default_rng(3).standard_normal(8)

    a_x = Audio(samples=x, sr=sr)
    a_h = Audio(samples=h, sr=sr)

    out = pair_convolve(a_x, a_h, mode="full", circular=False, normalize=None)
    assert out.sr == sr

    # mismatched sample rates must raise
    a_h_bad = Audio(samples=h, sr=sr2)
    with pytest.raises(ValueError):
        _ = pair_convolve(a_x, a_h_bad, mode="full", circular=False, normalize=None)


def test_auto_convolve_order_1_returns_copy_with_optional_norm():
    x = np.array([0.5, -0.5, 0.5, -0.5], dtype=float)
    a = Audio(samples=x, sr=22050)

    out = auto_convolve(a, mode="full", circular=False, normalize=None, order=1)
    # Same samples, independent array
    np.testing.assert_allclose(out.samples, x)
    assert out.sr == a.sr
    assert out.samples is not a.samples

    out_rms = auto_convolve(a, mode="full", circular=False, normalize="rms", order=1)
    assert _rms(out_rms.samples) == pytest.approx(0.1, rel=1e-4, abs=1e-6)
