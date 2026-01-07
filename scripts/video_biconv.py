# scripts/video_biconv.py
from __future__ import annotations
from pathlib import Path

import click

from exconv.xmodal import DualSerialMode, AudioLengthMode
from exconv.cli.video_biconv import run_video_biconv


@click.command()
@click.option(
    "--video",
    "video_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input video file (frames only; audio supplied separately).",
)
@click.option(
    "--audio",
    "audio_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Audio file to drive both directions. If omitted, use audio from the video (ffmpeg required).",
)
@click.option(
    "--out-video",
    "out_video",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Output video path.",
)
@click.option(
    "--out-audio",
    "out_audio",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=False,
    help="Output audio path (optional if muxing into video).",
)
@click.option(
    "--fps",
    type=float,
    default=None,
    show_default=True,
    help="Override FPS if metadata missing/incorrect.",
)
@click.option(
    "--fps-policy",
    type=click.Choice(["auto", "metadata", "avg_frame_rate", "r_frame_rate"]),
    default="auto",
    show_default=True,
    help="FPS selection policy when --fps is unset.",
)
@click.option(
    "--mux/--no-mux",
    default=True,
    show_default=True,
    help="Mux processed audio into the output video (requires ffmpeg).",
)
@click.option(
    "--serial-mode",
    type=click.Choice(["parallel", "serial-image-first", "serial-sound-first"]),
    default="parallel",
    show_default=True,
    help="How to chain image<->sound per frame.",
)
@click.option(
    "--audio-length-mode",
    type=click.Choice(["trim", "pad-zero", "pad-loop", "pad-noise", "center-zero"]),
    default="pad-zero",
    show_default=True,
    help="How to match audio length to video duration.",
)
@click.option(
    "--block-size",
    type=int,
    default=1,
    show_default=True,
    help="Process frames in blocks of this size (e.g. 12, 24, 50, 120...) using the same audio chunk.",
)
# sound->image
@click.option(
    "--s2i-mode",
    type=click.Choice(["mono", "stereo", "mid-side"]),
    default="mono",
    show_default=True,
    help="Sound->image mode.",
)
@click.option(
    "--s2i-colorspace",
    type=click.Choice(["luma", "color"]),
    default="luma",
    show_default=True,
    help="Sound->image colorspace.",
)
# image->sound
@click.option(
    "--i2s-mode",
    type=click.Choice(["flat", "hist", "radial"]),
    default="radial",
    show_default=True,
    help="Image->sound impulse mode.",
)
@click.option(
    "--i2s-colorspace",
    type=click.Choice(["luma", "rgb-mean", "rgb-stereo", "ycbcr-mid-side"]),
    default="luma",
    show_default=True,
    help="Image->sound colorspace.",
)
@click.option(
    "--i2s-pad-mode",
    type=click.Choice(["full", "same-center", "same-first"]),
    default="same-center",
    show_default=True,
    help="Image->sound convolution pad mode.",
)
@click.option(
    "--i2s-impulse-len",
    type=str,
    default="auto",
    show_default=True,
    help="Impulse length (int or 'auto'=match audio chunk).",
)
@click.option(
    "--i2s-radius-mode",
    type=click.Choice(["linear", "log"]),
    default="linear",
    show_default=True,
    help="Radial binning (radial mode).",
)
@click.option(
    "--i2s-phase-mode",
    type=click.Choice(["zero", "random", "image", "min-phase", "spiral"]),
    default="zero",
    show_default=True,
    help="Phase strategy (radial mode).",
)
@click.option(
    "--i2s-smoothing",
    type=click.Choice(["none", "hann"]),
    default="hann",
    show_default=True,
    help="Smoothing on radial profile.",
)
@click.option(
    "--i2s-impulse-norm",
    type=click.Choice(["energy", "peak", "none"]),
    default="energy",
    show_default=True,
    help="Impulse normalization.",
)
@click.option(
    "--i2s-out-norm",
    type=click.Choice(["match_rms", "match_peak", "none"]),
    default="match_rms",
    show_default=True,
    help="Output normalization for convolved audio.",
)
@click.option(
    "--i2s-n-bins",
    type=int,
    default=256,
    show_default=True,
    help="Histogram bins (hist mode).",
)
def main(
    video_path: Path,
    audio_path: Path | None,
    out_video: Path,
    out_audio: Path | None,
    fps: float | None,
    fps_policy: str,
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
):
    """
    Bi-directional video convolution: sound->image and image->sound per frame.
    """
    try:
        return run_video_biconv(
            video_path=video_path,
            audio_path=audio_path,
            out_video=out_video,
            out_audio=out_audio,
            fps=fps,
            fps_policy=fps_policy,
            mux=mux,
            serial_mode=serial_mode,
            audio_length_mode=audio_length_mode,
            block_size=block_size,
            s2i_mode=s2i_mode,
            s2i_colorspace=s2i_colorspace,
            i2s_mode=i2s_mode,
            i2s_colorspace=i2s_colorspace,
            i2s_pad_mode=i2s_pad_mode,
            i2s_impulse_len=i2s_impulse_len,
            i2s_radius_mode=i2s_radius_mode,
            i2s_phase_mode=i2s_phase_mode,
            i2s_smoothing=i2s_smoothing,
            i2s_impulse_norm=i2s_impulse_norm,
            i2s_out_norm=i2s_out_norm,
            i2s_n_bins=i2s_n_bins,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    main()
