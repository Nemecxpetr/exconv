from __future__ import annotations

from pathlib import Path

import numpy as np

from exconv.cli.exconv_cli import main
from exconv.io import write_audio


def test_spectrogram_cli_writes_png(tmp_path: Path) -> None:
    sr = 8000
    t = np.linspace(0.0, 0.25, int(sr * 0.25), endpoint=False)
    audio = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    audio_path = tmp_path / "tone.wav"
    out_path = tmp_path / "tone_spectrogram.png"
    write_audio(audio_path, audio, sr, subtype="FLOAT", dtype="float32")

    rc = main(
        [
            "spectrogram",
            "--audio",
            str(audio_path),
            "--out",
            str(out_path),
            "--n-fft",
            "512",
            "--hop-length",
            "128",
            "--max-freq",
            "2000",
            "--max-seconds",
            "0.1",
            "--dpi",
            "80",
        ]
    )

    assert rc == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0
