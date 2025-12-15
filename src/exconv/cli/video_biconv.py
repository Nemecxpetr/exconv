from __future__ import annotations

from pathlib import Path
import argparse

from exconv.xmodal import (
    biconv_video_from_files,
    DualSerialMode,
    AudioLengthMode,
)


def _path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _parse_impulse_len(val: str) -> str:
    if val == "auto":
        return val
    try:
        int(val)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("i2s-impulse-len must be integer or 'auto'") from exc
    return val


def _coerce_impulse_len_runtime(val: str) -> int | str:
    if val == "auto":
        return "auto"
    try:
        return int(val)
    except ValueError as exc:
        raise ValueError("i2s-impulse-len must be integer or 'auto'") from exc


def run_video_biconv(
    *,
    video_path: Path,
    audio_path: Path | None,
    out_video: Path,
    out_audio: Path | None,
    fps: float | None,
    mux: bool,
    serial_mode: DualSerialMode,
    audio_length_mode: AudioLengthMode,
    block_size: int,
    s2i_mode: str,
    s2i_colorspace: str,
    i2s_mode: str,
    i2s_colorspace: str,
    i2s_pad_mode: str,
    i2s_impulse_len: str,
    i2s_radius_mode: str,
    i2s_phase_mode: str,
    i2s_smoothing: str,
    i2s_impulse_norm: str,
    i2s_out_norm: str,
    i2s_n_bins: int,
) -> int:
    """
    Execute bi-directional video convolution and print a short summary.
    """
    impulse_len_resolved = _coerce_impulse_len_runtime(i2s_impulse_len)

    frames_out, audio_out, fps_used, sr = biconv_video_from_files(
        video_path=video_path,
        audio_path=audio_path,
        fps=fps,
        s2i_mode=s2i_mode,  # type: ignore[arg-type]
        s2i_colorspace=s2i_colorspace,  # type: ignore[arg-type]
        i2s_mode=i2s_mode,  # type: ignore[arg-type]
        i2s_colorspace=i2s_colorspace,  # type: ignore[arg-type]
        i2s_pad_mode=i2s_pad_mode,  # type: ignore[arg-type]
        i2s_impulse_len=impulse_len_resolved,  # type: ignore[arg-type]
        i2s_radius_mode=i2s_radius_mode,  # type: ignore[arg-type]
        i2s_phase_mode=i2s_phase_mode,  # type: ignore[arg-type]
        i2s_smoothing=i2s_smoothing,  # type: ignore[arg-type]
        i2s_impulse_norm=i2s_impulse_norm,  # type: ignore[arg-type]
        i2s_out_norm=i2s_out_norm,  # type: ignore[arg-type]
        i2s_n_bins=i2s_n_bins,
        serial_mode=serial_mode,
        audio_length_mode=audio_length_mode,
        block_size=block_size,
        out_video=out_video,
        out_audio=out_audio,
        mux_output=mux,
    )

    print(f"[done] wrote video {out_video} @ {fps_used:.3f} fps ({len(frames_out)} frames)")
    if out_audio:
        print(f"[done] wrote audio {out_audio} (shape={audio_out.shape}, sr={sr})")
    elif mux:
        print(f"[done] audio muxed into {out_video} (shape={audio_out.shape}, sr={sr})")

    return 0


def register_video_biconv_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the video-biconv subcommand on the shared argparse CLI.
    """
    p = subparsers.add_parser(
        "video-biconv",
        help="Bi-directional video convolution: sound <-> image per frame.",
    )
    p.add_argument(
        "--video",
        dest="video_path",
        required=True,
        help="Input video file (frames only; audio supplied separately or extracted).",
    )
    p.add_argument(
        "--audio",
        dest="audio_path",
        required=False,
        help="Audio file to drive both directions. If omitted, use audio from the video (ffmpeg required).",
    )
    p.add_argument(
        "--out-video",
        dest="out_video",
        required=True,
        help="Output video path.",
    )
    p.add_argument(
        "--out-audio",
        dest="out_audio",
        required=False,
        help="Output audio path (optional if muxing into video).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS if metadata missing/incorrect.",
    )
    p.add_argument(
        "--mux",
        dest="mux",
        action="store_true",
        default=True,
        help="Mux processed audio into the output video (requires ffmpeg).",
    )
    p.add_argument(
        "--no-mux",
        dest="mux",
        action="store_false",
        help="Disable muxing (write audio separately).",
    )
    p.add_argument(
        "--serial-mode",
        choices=["parallel", "serial-image-first", "serial-sound-first"],
        default="parallel",
        help="How to chain image<->sound per frame.",
    )
    p.add_argument(
        "--audio-length-mode",
        choices=["trim", "pad-zero", "pad-loop", "pad-noise", "center-zero"],
        default="pad-zero",
        help="How to match audio length to video duration.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Process frames in blocks of this size (e.g. 12, 24, 50, 120...) using the same audio chunk.",
    )
    # sound->image
    p.add_argument(
        "--s2i-mode",
        choices=["mono", "stereo", "mid-side"],
        default="mono",
        help="Sound->image mode.",
    )
    p.add_argument(
        "--s2i-colorspace",
        choices=["luma", "color"],
        default="luma",
        help="Sound->image colorspace.",
    )
    # image->sound
    p.add_argument(
        "--i2s-mode",
        choices=["flat", "hist", "radial"],
        default="radial",
        help="Image->sound impulse mode.",
    )
    p.add_argument(
        "--i2s-colorspace",
        choices=["luma", "rgb-mean", "rgb-stereo", "ycbcr-mid-side"],
        default="luma",
        help="Image->sound colorspace.",
    )
    p.add_argument(
        "--i2s-pad-mode",
        choices=["full", "same-center", "same-first"],
        default="same-center",
        help="Image->sound convolution pad mode.",
    )
    p.add_argument(
        "--i2s-impulse-len",
        type=_parse_impulse_len,
        default="auto",
        help="Impulse length (int or 'auto'=match audio chunk).",
    )
    p.add_argument(
        "--i2s-radius-mode",
        choices=["linear", "log"],
        default="linear",
        help="Radial binning (radial mode).",
    )
    p.add_argument(
        "--i2s-phase-mode",
        choices=["zero", "random", "image", "min-phase", "spiral"],
        default="zero",
        help="Phase strategy (radial mode).",
    )
    p.add_argument(
        "--i2s-smoothing",
        choices=["none", "hann"],
        default="hann",
        help="Smoothing on radial profile.",
    )
    p.add_argument(
        "--i2s-impulse-norm",
        choices=["energy", "peak", "none"],
        default="energy",
        help="Impulse normalization.",
    )
    p.add_argument(
        "--i2s-out-norm",
        choices=["match_rms", "match_peak", "none"],
        default="match_rms",
        help="Output normalization for convolved audio.",
    )
    p.add_argument(
        "--i2s-n-bins",
        type=int,
        default=256,
        help="Histogram bins (hist mode).",
    )

    p.set_defaults(func=_cmd_video_biconv)


def _cmd_video_biconv(args: argparse.Namespace) -> int:
    video_path = _path(args.video_path)
    audio_path = _path(args.audio_path) if args.audio_path else None
    out_video = _path(args.out_video)
    out_audio = _path(args.out_audio) if args.out_audio else None

    return run_video_biconv(
        video_path=video_path,
        audio_path=audio_path,
        out_video=out_video,
        out_audio=out_audio,
        fps=args.fps,
        mux=args.mux,
        serial_mode=args.serial_mode,
        audio_length_mode=args.audio_length_mode,
        block_size=args.block_size,
        s2i_mode=args.s2i_mode,
        s2i_colorspace=args.s2i_colorspace,
        i2s_mode=args.i2s_mode,
        i2s_colorspace=args.i2s_colorspace,
        i2s_pad_mode=args.i2s_pad_mode,
        i2s_impulse_len=args.i2s_impulse_len,
        i2s_radius_mode=args.i2s_radius_mode,
        i2s_phase_mode=args.i2s_phase_mode,
        i2s_smoothing=args.i2s_smoothing,
        i2s_impulse_norm=args.i2s_impulse_norm,
        i2s_out_norm=args.i2s_out_norm,
        i2s_n_bins=args.i2s_n_bins,
    )
