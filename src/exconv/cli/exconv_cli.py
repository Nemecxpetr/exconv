from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from exconv import __version__  # version helper :contentReference[oaicite:0]{index=0}
from exconv.cli.folderbatch import register_folderbatch_subcommand
from exconv.cli.video_biconv import register_video_biconv_subcommand
from exconv.cli.video_folderbatch import register_video_folderbatch_subcommand
from exconv.io import (
    read_audio,
    write_audio,
    read_image,
    write_image,
    as_uint8,
)  # audio/image IO 
from exconv.conv1d import Audio, auto_convolve as audio_auto_convolve  # :contentReference[oaicite:2]{index=2}
from exconv.conv2d import (
    image_auto_convolve,
    image_pair_convolve,
    gaussian_2d,
)  # 2D conv + kernels 
from exconv.xmodal.sound2image import spectral_sculpt  # :contentReference[oaicite:4]{index=4}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _parse_gaussian_kernel_spec(spec: str) -> np.ndarray:
    """
    Parse strings like:
        "gaussian"
        "gaussian:sigma=2.0"
        "gaussian:σ=2.0"
        "gaussian:sigma=2.0,radius=5"
    and return a 2D Gaussian kernel via exconv.conv2d.gaussian_2d.
    """
    if not spec:
        raise ValueError("Empty kernel spec.")

    kind, _, param_str = spec.partition(":")
    kind = kind.strip().lower()

    if kind != "gaussian":
        raise ValueError(f"Unsupported kernel kind {kind!r}; only 'gaussian' is supported for now.")

    params: Dict[str, str] = {}
    if param_str:
        for item in param_str.split(","):
            item = item.strip()
            if not item:
                continue
            key, eq, val = item.partition("=")
            if not eq:
                continue
            key = key.strip().lower()
            # allow unicode sigma
            key = key.replace("σ", "sigma")
            params[key] = val.strip()

    sigma_str = params.get("sigma")
    if sigma_str is None:
        raise ValueError("Gaussian kernel requires 'sigma' (e.g. 'gaussian:sigma=2.0').")

    sigma = float(sigma_str)
    radius = params.get("radius")
    truncate = params.get("truncate")

    radius_val: Optional[int] = int(radius) if radius is not None else None
    truncate_val: float = float(truncate) if truncate is not None else 3.0

    return gaussian_2d(sigma=sigma, radius=radius_val, truncate=truncate_val, normalize=True)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def _cmd_audio_auto(args: argparse.Namespace) -> int:
    in_path = _path(args.in_path)
    out_path = _path(args.out_path)

    samples, sr = read_audio(in_path, dtype="float32", always_2d=False)
    audio = Audio(samples=samples, sr=sr)
    out_audio = audio_auto_convolve(
        audio,
        mode=args.mode,
        circular=args.circular,
        normalize=args.normalize,
        order=args.order,
    )
    write_audio(
        out_path,
        out_audio.samples,
        out_audio.sr,
        subtype=args.subtype,
    )
    return 0


def _cmd_img_auto(args: argparse.Namespace) -> int:
    in_path = _path(args.in_path)
    out_path = _path(args.out_path)

    img = read_image(in_path, mode="RGB", dtype="uint8")

    if args.kernel is None:
        # pure auto-convolution
        out = image_auto_convolve(
            img,
            mode=args.mode,
            circular=args.circular,
            colorspace=args.colorspace,
            normalize=args.normalize,
        )
    else:
        kernel = _parse_gaussian_kernel_spec(args.kernel)
        out = image_pair_convolve(
            img,
            kernel=kernel,
            mode=args.mode,
            circular=args.circular,
            colorspace=args.colorspace,
            normalize=args.normalize,
        )

    out_u8 = as_uint8(out)
    write_image(out_path, out_u8)
    return 0

def _cmd_sound2image(args: argparse.Namespace) -> int:
    img_path = _path(args.img)
    audio_path = _path(args.audio)
    out_path = _path(args.out)

    # make sure output dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = read_image(img_path, mode="RGB", dtype="uint8")
    audio, sr = read_audio(audio_path, dtype="float32", always_2d=True)

    out = spectral_sculpt(
        image=img,
        audio=audio,
        sr=sr,
        mode=args.mode,
        colorspace=args.colorspace,
        normalize=args.normalize,
    )

    out_u8 = as_uint8(out)
    write_image(out_path, out_u8)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exconv",
        description="Minimal CLI for exconv demos (audio, image, and sound→image).",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- audio-auto ----
    p_audio = subparsers.add_parser(
        "audio-auto",
        help="Self-convolve an audio file (auto-convolution).",
    )
    p_audio.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input audio file (e.g. WAV/FLAC).",
    )
    p_audio.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output audio file.",
    )
    p_audio.add_argument(
        "--mode",
        choices=["full", "same-first", "same-center"],
        default="same-center",
        help="Linear convolution size policy.",
    )
    p_audio.add_argument(
        "--order",
        type=int,
        default=2,
        help="Self-convolution order (2 = x*x, 3 = x*x*x, ...).",
    )
    p_audio.add_argument(
        "--circular",
        action="store_true",
        help="Use circular convolution (length preserved).",
    )
    p_audio.add_argument(
        "--normalize",
        choices=["rms", "peak", "none"],
        default="rms",
        help="Output normalization mode.",
    )
    p_audio.add_argument(
        "--subtype",
        default="PCM_16",
        help='libsndfile subtype for writing (e.g. "PCM_16", "PCM_24", "FLOAT").',
    )
    p_audio.set_defaults(func=_cmd_audio_auto)

    # ---- img-auto ----
    p_img = subparsers.add_parser(
        "img-auto",
        help="Image auto-convolution or Gaussian pair-convolution.",
    )
    p_img.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input image (PNG/JPEG...).",
    )
    p_img.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output image.",
    )
    p_img.add_argument(
        "--mode",
        choices=["full", "same-first", "same-center"],
        default="same-center",
        help="Spatial size policy for linear convolution.",
    )
    p_img.add_argument(
        "--circular",
        action="store_true",
        help="Use circular convolution (wrap-around).",
    )
    p_img.add_argument(
        "--colorspace",
        choices=["luma", "channels"],
        default="channels",
        help="Convolution in luminance or per-channel RGB space.",
    )
    p_img.add_argument(
        "--normalize",
        choices=["clip", "rescale", "none"],
        default="rescale",
        help="Output normalization mode.",
    )
    p_img.add_argument(
        "--kernel",
        metavar="SPEC",
        help=(
            "Optional Gaussian kernel spec, e.g. "
            "'gaussian:sigma=2.0' or 'gaussian:σ=3.0,radius=7'. "
            "If omitted, pure auto-convolution is used."
        ),
    )
    p_img.set_defaults(func=_cmd_img_auto)

    # ---- sound2image ----
    p_s2i = subparsers.add_parser(
        "sound2image",
        help="Spectrally sculpt an image using an audio file.",
    )
    p_s2i.add_argument(
        "--img",
        required=True,
        help="Input image (PNG/JPEG...).",
    )
    p_s2i.add_argument(
        "--audio",
        required=True,
        help="Input audio file (mono or stereo).",
    )
    p_s2i.add_argument(
        "--out",
        required=True,
        help="Output image path.",
    )
    p_s2i.add_argument(
        "--mode",
        choices=["mono", "stereo", "mid-side"],
        default="mono",
        help="Audio → filter mapping: mono, stereo, or mid-side.",
    )
    p_s2i.add_argument(
        "--colorspace",
        choices=["luma", "color"],
        default="luma",
        help="Process only luminance (luma) or full YCbCr color.",
    )
    p_s2i.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable output normalization/clipping to [0,1].",
    )
    p_s2i.set_defaults(func=_cmd_sound2image, normalize=True)

    # ---- folderbatch ----
    register_folderbatch_subcommand(subparsers)

    # ---- video-biconv ----
    register_video_biconv_subcommand(subparsers)

    # ---- video-folderbatch ----
    register_video_folderbatch_subcommand(subparsers)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
