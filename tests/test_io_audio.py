import math
from pathlib import Path

import numpy as np

from exconv.io.audio import (
    read_audio,
    read_segment,
    write_audio,
    to_mono,
    to_stereo,
)


def test_to_mono_and_stereo_basic():
    # Simple deterministic test for mono/stereo conversion
    x = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)  # (2,2)
    mono = to_mono(x)
    assert mono.shape == (2,)
    # mean of each frame
    np.testing.assert_allclose(mono, np.array([2.0, 3.0], dtype=np.float32))

    stereo = to_stereo(mono)
    assert stereo.shape == (2, 2)
    np.testing.assert_allclose(stereo[:, 0], mono)
    np.testing.assert_allclose(stereo[:, 1], mono)


def test_audio_roundtrip_small(tmp_path: Path):
    sr = 8000
    t = np.linspace(0, 0.01, int(sr * 0.01), endpoint=False)
    x = 0.5 * np.sin(2 * math.pi * 440 * t).astype(np.float32)

    out_path = tmp_path / "test.wav"
    # Use FLOAT subtype for near-lossless round-trip
    write_audio(out_path, x, sr, subtype="FLOAT", dtype="float32")

    y, sr_out = read_audio(out_path, dtype="float32", always_2d=False)
    assert sr_out == sr
    assert y.shape == x.shape
    # allow tiny numerical differences
    np.testing.assert_allclose(y, x, rtol=1e-6, atol=1e-6)


def test_read_segment_samples(tmp_path: Path):
    sr = 16000
    n = 32
    # ramp so we can easily verify indices
    x = np.arange(n, dtype=np.float32)

    out_path = tmp_path / "seg.wav"
    write_audio(out_path, x, sr, subtype="FLOAT", dtype="float32")

    # read middle 10 samples
    start = 5
    stop = 15
    seg, sr_out = read_segment(
        out_path,
        start=start,
        stop=stop,
        unit="samples",
        dtype="float32",
        always_2d=False,
    )
    assert sr_out == sr
    assert seg.shape == (stop - start,)
    np.testing.assert_allclose(seg, x[start:stop])
