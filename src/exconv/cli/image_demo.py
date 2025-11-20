# src/exconv/cli/image_demo.py
"""Command-line demo: Image auto- and pair-convolution.
Usage:
    python -m exconv.cli.image_demo INPUT_PATH OUTPUT_DIR [--kernel_path KERNEL_PATH] [OPTIONS...]
Saves results to OUTPUT_DIR/<mode>/ with filenames indicating the method used.  
"""
import os
import click
from PIL import Image
import numpy as np
import imageio.v3 as iio

from exconv.conv2d.image import pair_convolve, auto_convolve

@click.command()
@click.argument("input_path")
@click.argument("output_dir")
@click.option("--kernel_path", default=None, help="Path to kernel image for pair convolution.")
@click.option("--mode", default="same-center", type=click.Choice(["full", "same-first", "same-center"]), show_default=True)
@click.option("--circular", is_flag=True, default=False, help="Use circular convolution.")
@click.option("--colorspace", default="channels", type=click.Choice(["luma", "channels"]), show_default=True)
@click.option("--normalize", default="rescale", type=click.Choice(["clip", "rescale", "none"]), show_default=True)
def main(input_path, output_dir, kernel_path, mode, circular, colorspace, normalize):

    # --- Load input image (keep uint8 → safe) ---
    img = np.asarray(Image.open(input_path).convert("RGB"))

    # --- Prepare output folders ---
    # Output structure:  output_dir/<mode>/auto_*.jpg , pair_*.jpg
    mode_dir = os.path.join(output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)

    # Useful base name
    base = os.path.splitext(os.path.basename(input_path))[0]

    # =============== AUTO-CONVOLUTION ===============
    out_auto = auto_convolve(
        img,
        mode=mode,
        circular=circular,
        colorspace=colorspace,
        normalize=normalize,
    )

    # Force float → uint8 for JPEG
    if np.issubdtype(out_auto.dtype, np.floating):
        out_auto = np.clip(out_auto, 0, 255).astype(np.uint8)

    auto_path = os.path.join(mode_dir, f"{base}_auto.jpg")
    iio.imwrite(auto_path, out_auto)
    print(f"Saved auto convolution → {auto_path}")

    # =============== PAIR-CONVOLUTION (optional) ===============
    if kernel_path is not None:
        kernel = np.asarray(Image.open(kernel_path).convert("RGB"))

        out_pair = pair_convolve(
            img,
            kernel=kernel,
            mode=mode,
            circular=circular,
            colorspace=colorspace,
            normalize=normalize,
        )

        # JPEG-safe
        if np.issubdtype(out_pair.dtype, np.floating):
            out_pair = np.clip(out_pair, 0, 255).astype(np.uint8)

        # Name uses both base names
        ker_base = os.path.splitext(os.path.basename(kernel_path))[0]
        pair_path = os.path.join(mode_dir, f"{base}_PAIR_{ker_base}.jpg")

        iio.imwrite(pair_path, out_pair)
        print(f"Saved pair convolution → {pair_path}")

    print(f"All results stored in: {mode_dir}")


if __name__ == "__main__":
    main()
