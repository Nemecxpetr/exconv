# tests/test_examples_audio_audio.py
import numpy as np
import pytest
from pathlib import Path

from exconv.io import read_audio
from exconv.conv1d import Audio, auto_convolve, pair_convolve

# AUDIO_FILES imported from conftest; but we can simply re-declare:
AUDIO_FILES = [
    "audio_long_sines.wav",
    "audio_plucks.wav",
]


def _asset_path(root: Path, name: str) -> Path:
    return root / name


def _load_audio(root: Path, name: str) -> Audio:
    path = _asset_path(root, name)
    samples, sr = read_audio(path, dtype="float32", always_2d=False)
    return Audio(samples=samples, sr=sr)


@pytest.mark.parametrize("fname_x", AUDIO_FILES)
@pytest.mark.parametrize("fname_h", AUDIO_FILES)
@pytest.mark.parametrize("mode", ["full", "same-first", "same-center"])
@pytest.mark.parametrize("circular", [False, True])
def test_pair_convolve_shapes_and_finiteness(test_assets_dir: Path, fname_x, fname_h, mode, circular):
    x = _load_audio(test_assets_dir, fname_x)
    h = _load_audio(test_assets_dir, fname_h)

    y = pair_convolve(
        x,
        h,
        mode=mode,
        circular=circular,
        normalize="none",
    )

    assert y.sr == x.sr == h.sr

    n_x = x.n_samples
    n_h = h.n_samples
    n_y = y.n_samples

    if circular:
        assert n_y == n_x
    else:
        if mode == "full":
            assert n_y == n_x + n_h - 1
        else:
            assert n_y == n_x

    assert np.all(np.isfinite(y.samples))
    assert np.any(np.abs(y.samples) > 0.0)


@pytest.mark.parametrize("fname", AUDIO_FILES)
@pytest.mark.parametrize("mode", ["same-first", "same-center"])
@pytest.mark.parametrize("circular", [False, True])
def test_auto_convolve_reasonable_energy(test_assets_dir: Path, fname, mode, circular):
    x = _load_audio(test_assets_dir, fname)

    y = auto_convolve(
        x,
        mode=mode,
        circular=circular,
        normalize="rms",
        order=2,
    )

    assert y.sr == x.sr
    assert y.n_samples == x.n_samples

    in_rms = np.sqrt(np.mean(x.samples.astype(np.float64) ** 2))
    out_rms = np.sqrt(np.mean(y.samples.astype(np.float64) ** 2))

    assert np.isfinite(in_rms) and np.isfinite(out_rms)
    if in_rms > 0:
        assert out_rms > 0
