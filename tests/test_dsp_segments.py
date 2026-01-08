"""
Tests for exconv.dsp.segments helpers.
"""

import numpy as np

from exconv.dsp import segments


def test_slice_for_frame_clamps_to_audio_len():
    start, stop = segments.slice_for_frame(frame_idx=1, fps=2.0, sr=10, n_samples=7)
    assert (start, stop) == (5, 7)


def test_match_audio_length_center_zero_padding():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = segments.match_audio_length(audio, 5, mode="center-zero")
    expected = np.array([0.0, 1.0, 2.0, 3.0, 0.0], dtype=np.float32)[:, None]
    assert np.array_equal(out, expected)


def test_audio_chunk_for_interval_pad_loop():
    audio = np.arange(3, dtype=np.float32)
    rng = np.random.default_rng(0)
    out = segments.audio_chunk_for_interval(
        audio,
        start=0,
        stop=7,
        mode="pad-loop",
        rng=rng,
        noise_scale=0.0,
    )
    expected = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0], dtype=np.float32)[
        :, None
    ]
    assert np.array_equal(out, expected)
