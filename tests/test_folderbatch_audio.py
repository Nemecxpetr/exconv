from pathlib import Path

import numpy as np

from exconv.cli.folderbatch import _balanced_random_pairs, process_audio_batch
from exconv.io import write_audio


def _write_test_wav(path: Path, value: float) -> None:
    sr = 8000
    samples = np.array([value, 0.0, -value, 0.0], dtype=np.float32)
    write_audio(path, samples, sr, subtype="FLOAT")


def test_process_audio_batch_multi_is_opt_in(tmp_path: Path):
    audio_dir = tmp_path / "input"
    audio_dir.mkdir()
    _write_test_wav(audio_dir / "a.wav", 0.25)
    _write_test_wav(audio_dir / "b.wav", 0.5)

    out_self = tmp_path / "self"
    out_pair = tmp_path / "pair"
    out_multi = tmp_path / "multi"

    process_audio_batch(
        audio_dir,
        out_self,
        out_pair,
        out_multi,
        mode="same-center",
        normalize="none",
        subtype="FLOAT",
    )

    assert out_self.exists()
    assert out_pair.exists()
    assert not out_multi.exists()


def test_process_audio_batch_multi_circular_implies_multi(tmp_path: Path):
    audio_dir = tmp_path / "input"
    audio_dir.mkdir()
    _write_test_wav(audio_dir / "a.wav", 0.25)
    _write_test_wav(audio_dir / "b.wav", 0.5)

    out_self = tmp_path / "self"
    out_pair = tmp_path / "pair"
    out_multi = tmp_path / "multi"

    process_audio_batch(
        audio_dir,
        out_self,
        out_pair,
        out_multi,
        mode="same-center",
        include_multi=True,
        multi_circular=True,
        normalize="none",
        subtype="FLOAT",
    )

    assert out_multi.exists()
    assert len(list(out_multi.glob("*.wav"))) == 1


def test_process_audio_batch_recursive_reads_subfolders(tmp_path: Path):
    audio_dir = tmp_path / "input"
    nested_dir = audio_dir / "nested"
    nested_dir.mkdir(parents=True)
    _write_test_wav(audio_dir / "a.wav", 0.25)
    _write_test_wav(nested_dir / "a.wav", 0.5)

    out_self = tmp_path / "self"
    out_pair = tmp_path / "pair"
    out_multi = tmp_path / "multi"

    process_audio_batch(
        audio_dir,
        out_self,
        out_pair,
        out_multi,
        mode="same-center",
        normalize="none",
        subtype="FLOAT",
        recursive=True,
    )

    assert (out_self / "a__SELF_o2.wav").exists()
    assert (out_self / "nested__a__SELF_o2.wav").exists()
    assert len(list(out_pair.glob("*.wav"))) == 1


def test_balanced_random_pairs_samples_near_regular_subset():
    pairs = _balanced_random_pairs(100, 110, seed=123)

    assert len(pairs) == 110
    assert len(set(pairs)) == 110

    degrees = np.zeros(100, dtype=int)
    for i, j in pairs:
        assert i != j
        degrees[i] += 1
        degrees[j] += 1

    assert degrees.min() >= 2
    assert degrees.max() <= 3


def test_process_audio_batch_samples_pairs_above_limit(tmp_path: Path):
    audio_dir = tmp_path / "input"
    audio_dir.mkdir()
    for i in range(4):
        _write_test_wav(audio_dir / f"{i}.wav", 0.1 + i * 0.1)

    out_self = tmp_path / "self"
    out_pair = tmp_path / "pair"
    out_multi = tmp_path / "multi"

    process_audio_batch(
        audio_dir,
        out_self,
        out_pair,
        out_multi,
        mode="same-center",
        normalize="none",
        subtype="FLOAT",
        max_pairs=3,
    )

    assert len(list(out_self.glob("*.wav"))) == 4
    assert len(list(out_pair.glob("*.wav"))) == 3


def test_process_audio_batch_high_sample_factor_allows_all_pairs(tmp_path: Path):
    audio_dir = tmp_path / "input"
    audio_dir.mkdir()
    for i in range(4):
        _write_test_wav(audio_dir / f"{i}.wav", 0.1 + i * 0.1)

    out_self = tmp_path / "self"
    out_pair = tmp_path / "pair"
    out_multi = tmp_path / "multi"

    process_audio_batch(
        audio_dir,
        out_self,
        out_pair,
        out_multi,
        mode="same-center",
        normalize="none",
        subtype="FLOAT",
        max_pairs=0,
        pair_sample_factor=10.0,
    )

    assert len(list(out_pair.glob("*.wav"))) == 6
