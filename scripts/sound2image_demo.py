# scripts/sound2image_demo.py

from pathlib import Path

import click
import numpy as np

from exconv.io import read_image, read_audio, write_image
from exconv.xmodal.sound2image import spectral_sculpt


@click.command()
@click.argument(
    "image_path", 
    type=click.Path(exists=True, dir_okay=False),
    default="samples/input/test_assets/img_checker.png",
    show_default=True,
)
@click.argument(
    "audio_path", 
    type=click.Path(exists=True, dir_okay=False),
    default="samples/input/test_assets/audio_long_sines.wav",
    show_default=True,
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False),
    default="samples/output/img/sound2image_demo/demo.png",
    show_default=True,
)
@click.option(
    "--mode",
    type=click.Choice(["mono", "stereo", "mid-side"]),
    default="mono",
    show_default=True,
    help="Sound→image mapping mode.",
)
@click.option(
    "--colorspace",
    type=click.Choice(["luma", "color"]),
    default="luma",
    show_default=True,
    help="Luma-only filtering or full YCbCr color mapping.",
)
@click.option(
    "--no-normalize",
    is_flag=True,
    help="Disable final [0,1] normalization (keep raw float output).",
)
def main(image_path, audio_path, output_path, mode, colorspace, no_normalize):
    """
    Simple sound→image demo.

    IMAGE_PATH : input image (JPG/PNG/...)
    AUDIO_PATH : input audio (WAV/FLAC/...)
    OUTPUT_PATH: output image path
    """
    image_path = Path(image_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # --- load data ---
    img = read_image(image_path, mode="RGB", dtype="uint8")
    audio, sr = read_audio(audio_path, dtype="float32", always_2d=False)

    # --- process ---
    out = spectral_sculpt(
        img,
        audio,
        sr,
        mode=mode,
        colorspace=colorspace,
        normalize=True,
    )

    # --- save result ---
    out_vis = out
    if np.issubdtype(out_vis.dtype, np.floating):
        # For demo/visualization: always map to 0..1
        vmin = float(out_vis.min())
        vmax = float(out_vis.max())
        if vmax > vmin:
            out_vis = (out_vis - vmin) / (vmax - vmin)
        out_vis = np.clip(out_vis, 0.0, 1.0).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_path, out_vis)


if __name__ == "__main__":
    main()
