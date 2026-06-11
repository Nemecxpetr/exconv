from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Sequence


def _command_to_text(cmd: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(cmd))
    return " ".join(cmd)


def _probe_video_size(ffprobe: str, path: Path) -> tuple[int, int]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    value = result.stdout.strip()
    if "x" not in value:
        raise RuntimeError(f"Unexpected ffprobe output for {path}: {value}")
    width_str, height_str = value.split("x", maxsplit=1)
    return int(width_str), int(height_str)


def _choose_target_size(
    sizes: Sequence[tuple[int, int]],
    *,
    prefer_horizontal: bool,
) -> tuple[int, int]:
    if prefer_horizontal:
        horizontal = [size for size in sizes if size[0] >= size[1]]
        if horizontal:
            return max(horizontal, key=lambda size: size[0] * size[1])
    return max(sizes, key=lambda size: size[0] * size[1])


def _fit_within(
    width: int,
    height: int,
    max_width: int | None,
    max_height: int | None,
) -> tuple[int, int]:
    if not max_width or not max_height:
        return width, height
    scale = min(max_width / width, max_height / height, 1.0)
    return int(width * scale), int(height * scale)


def _ensure_even(width: int, height: int) -> tuple[int, int]:
    return max(2, width - width % 2), max(2, height - height % 2)


def _resolve_target_size(
    args: argparse.Namespace, inputs: Sequence[Path]
) -> tuple[int, int]:
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            raise SystemExit("--width and --height must be provided together")
        return _ensure_even(args.width, args.height)

    sizes = [_probe_video_size(args.ffprobe, path) for path in inputs]
    width, height = _choose_target_size(
        sizes,
        prefer_horizontal=args.prefer_horizontal,
    )
    width, height = _fit_within(width, height, args.max_width, args.max_height)
    return _ensure_even(width, height)


def _video_filter_chain(args: argparse.Namespace, width: int, height: int) -> str:
    scale_expr = f"min(min({width}/iw,{height}/ih),1)"
    width_expr = f"trunc(iw*{scale_expr}/2)*2"
    height_expr = f"trunc(ih*{scale_expr}/2)*2"

    if args.vulkan_scale:
        upload = "hwdownload,format=nv12,hwupload" if args.hw_decode else "hwupload"
        return (
            f"{upload},"
            f"scale_vulkan=w='{width_expr}':h='{height_expr}',"
            "hwdownload,format=nv12,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1,setpts=PTS-STARTPTS"
        )

    prefix = "hwdownload,format=nv12," if args.hw_decode else ""
    return (
        f"{prefix}"
        f"scale=w='{width_expr}':h='{height_expr}':flags={args.scale_flags},"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "setsar=1,setpts=PTS-STARTPTS"
    )


def _build_filter_complex(
    args: argparse.Namespace,
    width: int,
    height: int,
) -> str:
    video_chain = _video_filter_chain(args, width, height)
    parts = [f"[{idx}:v]{video_chain}[v{idx}]" for idx, _ in enumerate(args.inputs)]

    if args.include_audio:
        audio_chain = (
            "aformat=sample_rates=48000:channel_layouts=stereo," "asetpts=PTS-STARTPTS"
        )
        parts.extend(
            f"[{idx}:a]{audio_chain}[a{idx}]" for idx, _ in enumerate(args.inputs)
        )
        labels = "".join(f"[v{idx}][a{idx}]" for idx, _ in enumerate(args.inputs))
        parts.append(f"{labels}concat=n={len(args.inputs)}:v=1:a=1[v][a]")
    else:
        labels = "".join(f"[v{idx}]" for idx, _ in enumerate(args.inputs))
        parts.append(f"{labels}concat=n={len(args.inputs)}:v=1:a=0[v]")

    return ";".join(parts)


def _build_cmd(
    args: argparse.Namespace, inputs: Sequence[Path], output: Path
) -> list[str]:
    width, height = _resolve_target_size(args, inputs)
    filter_complex = _build_filter_complex(args, width, height)

    cmd = [args.ffmpeg]
    if args.overwrite:
        cmd.append("-y")
    if args.vulkan_scale:
        cmd += [
            "-init_hw_device",
            f"vulkan={args.gpu_filter_device}",
            "-filter_hw_device",
            args.gpu_filter_device,
        ]
    for input_path in inputs:
        if args.hw_decode:
            cmd += [
                "-hwaccel",
                args.hw_decoder,
                "-hwaccel_output_format",
                args.hw_output_format,
            ]
        cmd += ["-i", str(input_path)]

    cmd += ["-filter_complex", filter_complex, "-map", "[v]", "-c:v", args.video_codec]
    if args.video_codec == "libx264":
        cmd += ["-crf", str(args.crf), "-preset", args.preset]
    if args.include_audio:
        cmd += ["-map", "[a]", "-c:a", args.audio_codec, "-b:a", args.audio_bitrate]
    else:
        cmd.append("-an")
    cmd.append(str(output))
    return cmd


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate videos after scaling/padding them to one frame size.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input videos in order.")
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output video."
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable.")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable.")
    parser.add_argument("--width", type=int, default=None, help="Manual output width.")
    parser.add_argument(
        "--height", type=int, default=None, help="Manual output height."
    )
    parser.add_argument(
        "--max-width", type=int, default=640, help="Auto target max width."
    )
    parser.add_argument(
        "--max-height", type=int, default=360, help="Auto target max height."
    )
    parser.add_argument(
        "--prefer-vertical",
        dest="prefer_horizontal",
        action="store_false",
        help="When choosing auto size, do not prefer horizontal sources.",
    )
    parser.set_defaults(prefer_horizontal=True)
    parser.add_argument(
        "--no-audio",
        dest="include_audio",
        action="store_false",
        help="Write video-only output.",
    )
    parser.set_defaults(include_audio=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output.")
    parser.add_argument(
        "--scale-flags", default="fast_bilinear", help="ffmpeg scale flags."
    )
    parser.add_argument("--video-codec", default="libx264", help="Output video codec.")
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF.")
    parser.add_argument("--preset", default="medium", help="libx264 preset.")
    parser.add_argument("--audio-codec", default="aac", help="Output audio codec.")
    parser.add_argument("--audio-bitrate", default="192k", help="Output audio bitrate.")
    parser.add_argument(
        "--hw-decode",
        action="store_true",
        help="Enable ffmpeg hardware decode for each input.",
    )
    parser.add_argument("--hw-decoder", default="d3d11va", help="ffmpeg hwaccel name.")
    parser.add_argument(
        "--hw-output-format",
        default="d3d11",
        help="ffmpeg hwaccel output format.",
    )
    parser.add_argument(
        "--vulkan-scale",
        action="store_true",
        help="Use scale_vulkan instead of the CPU scale filter.",
    )
    parser.add_argument(
        "--gpu-filter-device",
        default="vk",
        help="Name for the initialized Vulkan filter device.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    inputs = [path.expanduser().resolve() for path in args.inputs]
    if len(inputs) < 2:
        raise SystemExit("At least two input videos are required")
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input videos:\n" + "\n".join(missing))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(args, inputs, output)
    print(_command_to_text(cmd))
    if args.dry_run:
        return 0
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
