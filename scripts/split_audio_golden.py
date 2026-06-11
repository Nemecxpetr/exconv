# scripts/split_audio_golden.py

from __future__ import annotations

import math
from pathlib import Path

import click
import soundfile as sf

from exconv.io import read_segment, write_audio


def _golden_ratio_inv() -> float:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    return 1.0 / phi


def _allocate_lengths(total_frames: int, parts: int, ratio: float) -> list[int]:
    weights = [ratio**i for i in range(parts)]
    sum_w = sum(weights)
    raw = [total_frames * w / sum_w for w in weights]
    base = [int(x) for x in raw]

    remainder = total_frames - sum(base)
    if remainder > 0:
        frac = [x - int(x) for x in raw]
        order = sorted(range(parts), key=lambda i: frac[i], reverse=True)
        for i in range(remainder):
            base[order[i]] += 1

    return base


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, dir_okay=False),
    default="samples/input/audio/original.wav",
)
@click.option(
    "--parts",
    type=int,
    default=5,
    show_default=True,
    help="Number of golden-ratio parts to create.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Output directory for the split files (defaults to <input>_golden).",
)
@click.option(
    "--prefix",
    default=None,
    help="Filename prefix (defaults to input stem).",
)
@click.option(
    "--subtype",
    default="PCM_16",
    show_default=True,
    help="Output subtype passed to libsndfile (e.g. PCM_16, PCM_24, FLOAT).",
)
def main(
    input_path: str, parts: int, out_dir: str | None, prefix: str | None, subtype: str
) -> None:
    """
    Split a single audio file into N golden-ratio parts and write them out.
    """
    if parts < 1:
        raise click.UsageError("--parts must be >= 1")

    input_path = Path(input_path)
    info = sf.info(str(input_path))
    total_frames = int(info.frames)
    sr = int(info.samplerate)

    if total_frames < parts:
        raise click.UsageError("Audio is too short for the requested part count.")

    ratio = _golden_ratio_inv()
    lengths = _allocate_lengths(total_frames, parts, ratio)

    if any(n <= 0 for n in lengths):
        raise click.UsageError("Some parts would be empty; reduce --parts.")

    if out_dir is None:
        out_dir_path = input_path.parent / f"{input_path.stem}_golden"
    else:
        out_dir_path = Path(out_dir)

    if prefix is None:
        prefix = input_path.stem

    ext = input_path.suffix if input_path.suffix else ".wav"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    start = 0
    for idx, n_frames in enumerate(lengths, start=1):
        stop = start + n_frames
        data, _ = read_segment(
            input_path,
            start=start,
            stop=stop,
            unit="samples",
            dtype="float32",
            always_2d=False,
        )
        out_name = f"{prefix}_part_{idx:02d}{ext}"
        write_audio(out_dir_path / out_name, data, sr, subtype=subtype)
        start = stop

    click.echo(f"Wrote {parts} parts to {out_dir_path}")


if __name__ == "__main__":
    main()
