from pathlib import Path

import numpy as np

from exconv.cli.folderbatch import process_audio_batch
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
