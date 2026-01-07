from __future__ import annotations

from pathlib import Path
import argparse

from exconv.xmodal import (
    biconv_video_to_files_stream,
    DualSerialMode,
    AudioLengthMode,
)


def _path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _parse_impulse_len(val: str) -> str:
    if val in ("auto", "frame"):
        return val
    try:
        int(val)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "i2s-impulse-len must be integer, 'auto', or 'frame'"
        ) from exc
    return val


def _coerce_impulse_len_runtime(val: str) -> int | str:
    if val in ("auto", "frame"):
        return val
    try:
        return int(val)
    except ValueError as exc:
        raise ValueError("i2s-impulse-len must be integer, 'auto', or 'frame'") from exc


def run_video_biconv(
    *,
    video_path: Path,
    audio_path: Path | None,
    out_video: Path,
    out_audio: Path | None,
    fps: float | None,
    fps_policy: str = "auto",
    mux: bool,
    serial_mode: DualSerialMode,
    audio_length_mode: AudioLengthMode,
    block_size: int,
    block_size_div: int | None,
    block_strategy: str,
    block_min_frames: int,
    block_max_frames: int | None,
    block_beats_per: int,
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
    s2i_safe_color: bool,
    s2i_chroma_strength: float,
    s2i_chroma_clip: float,
) -> int:
    """
    Execute bi-directional video convolution and print a short summary.
    """
    impulse_len_resolved = _coerce_impulse_len_runtime(i2s_impulse_len)

    n_frames, audio_shape, fps_used, sr = biconv_video_to_files_stream(
        video_path=video_path,
        audio_path=audio_path,
        fps=fps,
        fps_policy=fps_policy,
        s2i_mode=s2i_mode,  # type: ignore[arg-type]
        s2i_colorspace=s2i_colorspace,  # type: ignore[arg-type]
        s2i_safe_color=s2i_safe_color,
        s2i_chroma_strength=s2i_chroma_strength,
        s2i_chroma_clip=s2i_chroma_clip,
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
        block_size_div=block_size_div,
        block_strategy=block_strategy,  # type: ignore[arg-type]
        block_min_frames=block_min_frames,
        block_max_frames=block_max_frames,
        block_beats_per=block_beats_per,
        out_video=out_video,
        out_audio=out_audio,
        mux_output=mux,
    )

    print(f"[done] wrote video {out_video} @ {fps_used:.3f} fps ({n_frames} frames)")
    if out_audio:
        print(f"[done] wrote audio {out_audio} (shape={audio_shape}, sr={sr})")
    elif mux:
        print(f"[done] audio muxed into {out_video} (shape={audio_shape}, sr={sr})")

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
        "--fps-policy",
        choices=["auto", "metadata", "avg_frame_rate", "r_frame_rate"],
        default="auto",
        help=(
            "FPS selection policy when --fps is unset. "
            "auto prefers r_frame_rate when avg/r differ significantly."
        ),
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
        help=(
            "Process frames in blocks of this size (e.g. 12, 24, 50, 120...) "
            "using the same audio chunk (fixed strategy only)."
        ),
    )
    p.add_argument(
        "--block-size-div",
        type=int,
        default=None,
        help=(
            "Alternative: split the video into N blocks (divisor). "
            "1 = whole video as one block, 2 = halves, etc. "
            "Overrides --block-size when set (fixed strategy only)."
        ),
    )
    p.add_argument(
        "--block-strategy",
        choices=["fixed", "beats", "novelty", "structure"],
        default="fixed",
        help=(
            "Block segmentation strategy: fixed (frame-count) or "
            "audio-driven (beats/novelty/structure)."
        ),
    )
    p.add_argument(
        "--block-min-frames",
        type=int,
        default=1,
        help="Minimum block length (in frames) when using audio-driven strategies.",
    )
    p.add_argument(
        "--block-max-frames",
        type=int,
        default=None,
        help="Maximum block length (in frames) when using audio-driven strategies.",
    )
    p.add_argument(
        "--beats-per-block",
        type=int,
        default=1,
        help="Group this many beats into a single block for --block-strategy beats.",
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
    p.add_argument(
        "--s2i-safe-color",
        dest="s2i_safe_color",
        action="store_true",
        default=True,
        help="Enable chroma-safe normalization in color mode.",
    )
    p.add_argument(
        "--s2i-unsafe-color",
        dest="s2i_safe_color",
        action="store_false",
        help="Disable chroma-safe normalization in color mode.",
    )
    p.add_argument(
        "--s2i-chroma-strength",
        type=float,
        default=0.5,
        help="Blend between original chroma (0.0) and fully filtered chroma (1.0) in color mode.",
    )
    p.add_argument(
        "--s2i-chroma-clip",
        type=float,
        default=0.25,
        help="Max chroma deviation around 0.5 when safe-color is enabled.",
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
        help="Impulse length (int, 'auto'=match audio chunk, 'frame'=one frame's worth of samples).",
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
        fps_policy=args.fps_policy,
        mux=args.mux,
        serial_mode=args.serial_mode,
        audio_length_mode=args.audio_length_mode,
        block_size=args.block_size,
        block_size_div=args.block_size_div,
        block_strategy=args.block_strategy,
        block_min_frames=args.block_min_frames,
        block_max_frames=args.block_max_frames,
        block_beats_per=args.beats_per_block,
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
        s2i_safe_color=args.s2i_safe_color,
        s2i_chroma_strength=args.s2i_chroma_strength,
        s2i_chroma_clip=args.s2i_chroma_clip,
    )
