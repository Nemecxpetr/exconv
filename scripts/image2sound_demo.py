# scripts/image2sound_demo.py

from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt

from exconv.io import read_audio, write_audio, read_image
from exconv.xmodal.image2sound import (
    image2sound_flat,
    image2sound_hist,
    image2sound_radial,
)


@click.command()
@click.option(
    "--audio",
    "audio_path",
    type=click.Path(exists=True, dir_okay=False),
    default="samples/input/audio/test_kernel.wav",
    show_default=True,
    help="Input audio file.",
)
@click.option(
    "--image",
    "image_path",
    type=click.Path(exists=True, dir_okay=False),
    default="samples/input/test_assets/img_gradients.png",
    show_default=True,
    help="Input image file.",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default="samples/output/examples/img2sound_demo",
    show_default=True,
    help="Directory for the output file.",
)
@click.option(
    "--name",
    "name",
    type=str,
    default="img2sound_radial",
    show_default=True,
    help="Name tag for the output file.",
)
@click.option(
    "--mode",
    type=click.Choice(["flat", "hist", "radial"]),
    default="radial",
    show_default=True,
    help="Impulse derivation mode.",
)
@click.option(
    "--pad-mode",
    type=click.Choice(["full", "same-center", "same-first"]),
    default="same-first",
    show_default=True,
)
@click.option(
    "--colorspace",
    type=click.Choice(["luma", "rgb-mean", "rgb-stereo", "ycbcr-mid-side"]),
    default="luma",
    show_default=True,
)
@click.option(
    "--impulse-norm",
    type=click.Choice(["energy", "peak", "none"]),
    default="energy",
    show_default=True,
)
@click.option(
    "--out-norm",
    type=click.Choice(["match_rms", "match_peak", "none"]),
    default="match_rms",
    show_default=True,
)
@click.option(
    "--impulse-len",
    type=str,
    default="8192",
    show_default=True,
    help="Target impulse length for radial mode (integer or 'auto' to match audio length).",
)
@click.option(
    "--radius-mode",
    type=click.Choice(["linear", "log"]),
    default="linear",
    show_default=True,
    help="Radius binning mode for radial FFT mapping.",
)
@click.option(
    "--phase-mode",
    type=click.Choice(["zero", "random", "image", "min-phase", "spiral"]),
    default="zero",
    show_default=True,
    help="Phase strategy for radial mode.",
)
@click.option(
    "--smoothing",
    type=click.Choice(["none", "hann"]),
    default="hann",
    show_default=True,
    help="Optional smoothing on radial profile.",
)
def main(
    audio_path: str,
    image_path: str,
    out_dir: str,
    name: str,
    mode: str,
    pad_mode: str,
    colorspace: str,
    impulse_norm: str,
    out_norm: str,
    impulse_len: str,
    radius_mode: str,
    phase_mode: str,
    smoothing: str,
):
    """
    Simple image->sound demo.

    Uses default small test assets, unless overridden with options.
    """
    audio_path = Path(audio_path)
    image_path = Path(image_path)
    out_dir = Path(out_dir)

    # Auto-generate output file name:
    # e.g. audio_long_sines__img_checker__img2sound_radial.wav
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{audio_path.stem}__{image_path.stem}__{name}.wav"

    # --- Load inputs ---
    print(f"[load] audio: {audio_path}")
    samples, sr = read_audio(audio_path, dtype="float32", always_2d=False)
    print(f"       shape={samples.shape}, sr={sr}")

    print(f"[load] image: {image_path}")
    img = read_image(image_path, mode="RGB", dtype="uint8")
    print(f"       shape={img.shape}, dtype={img.dtype}")

    # Resolve impulse length (supports "auto")
    if impulse_len == "auto":
        resolved_impulse_len = samples.shape[0]
    else:
        try:
            resolved_impulse_len = int(impulse_len)
        except ValueError:
            raise click.BadParameter(
                f"impulse-len must be integer or 'auto', got {impulse_len}"
            )
    if resolved_impulse_len <= 0:
        raise click.BadParameter("impulse-len must be positive")

    # Normalize to 0..1 float for nicer behavior in the impulse mapping
    img_f = img.astype(np.float32) / 255.0
    # --- Run image->sound ---
    print("[proc] deriving impulse from image and convolving...")
    if mode == "flat":
        y, sr_out, h = image2sound_flat(
            audio=samples,
            sr=sr,
            img=img_f,
            pad_mode=pad_mode,
            colorspace=colorspace,
            impulse_norm=impulse_norm,
            out_norm=out_norm,
        )
    elif mode == "hist":
        y, sr_out, h = image2sound_hist(
            audio=samples,
            sr=sr,
            img=img_f,
            pad_mode=pad_mode,
            colorspace=colorspace,
            impulse_norm=impulse_norm,
            out_norm=out_norm,
        )
    elif mode == "radial":
        y, sr_out, h = image2sound_radial(
            audio=samples,
            sr=sr,
            img=img_f,
            pad_mode=pad_mode,
            colorspace=colorspace,
            impulse_len=resolved_impulse_len,
            radius_mode=radius_mode,
            phase_mode=phase_mode,
            smoothing=smoothing,
            impulse_norm=impulse_norm,
            out_norm=out_norm,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    impulse_len_out = h.shape[0] if h.ndim == 1 else h.shape[0]
    impulse_channels = 1 if h.ndim == 1 else h.shape[1]

    print(
        f"[info] impulse length = {impulse_len_out} samples, "
        f"channels = {impulse_channels}"
    )
    print(f"[info] output shape   = {y.shape}, sr_out={sr_out}")
    print(
        f"[info] output peak    = {float(np.max(np.abs(y))):.3f}, "
        f"rms = {float(np.sqrt(np.mean(y**2))):.3f}"
    )

    # --- Save impulse visualization ---
    # Downsample for plotting if impulse is very long
    h_plot = h
    max_points = 5000
    if h_plot.shape[0] > max_points:
        step = int(np.ceil(h_plot.shape[0] / max_points))
        h_plot = h_plot[::step]

    impulse_plot_path = output_path.with_suffix(".impulse.png")
    print(f"[plot] saving impulse visualization: {impulse_plot_path}")

    plt.figure(figsize=(10, 3))
    if h_plot.ndim == 1:
        plt.plot(h_plot, label="impulse")
    else:
        for ch in range(h_plot.shape[1]):
            plt.plot(h_plot[:, ch], label=f"impulse ch{ch+1}")
        plt.legend()
    plt.title("Image-derived impulse response")
    plt.xlabel(
        "Sample index (downsampled)" if h.shape[0] > max_points else "Sample index"
    )
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(impulse_plot_path, dpi=150)
    plt.close()

    # --- Save output ---
    print(f"[save] writing: {output_path}")
    write_audio(output_path, y, sr_out, subtype="PCM_16")

    print("[done]")


if __name__ == "__main__":
    main()
