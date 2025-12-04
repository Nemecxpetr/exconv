# scripts/video_biconv.py
from __future__ import annotations
from pathlib import Path

import click

from exconv.xmodal import (
    biconv_video_from_files,
    DualSerialMode,
    AudioLengthMode,
)


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
    mux: bool,
    serial_mode: DualSerialMode,
    audio_length_mode: AudioLengthMode,
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
    # resolve impulse len
    impulse_len_resolved: int | str
    if i2s_impulse_len == "auto":
        impulse_len_resolved = "auto"
    else:
        try:
            impulse_len_resolved = int(i2s_impulse_len)
        except ValueError:
            raise click.BadParameter("i2s-impulse-len must be integer or 'auto'")

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
        out_video=out_video,
        out_audio=out_audio,
        mux_output=mux,
    )

    print(f"[done] wrote video {out_video} @ {fps_used:.3f} fps")
    if out_audio:
        print(f"[done] wrote audio {out_audio} (shape={audio_out.shape}, sr={sr})")
    elif mux:
        print(f"[done] audio muxed into {out_video} (shape={audio_out.shape}, sr={sr})")


if __name__ == "__main__":
    main()
