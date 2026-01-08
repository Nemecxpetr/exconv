from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, List

DEFAULT_PREFIX = r".\samples\output\video\shu_closeup"
DEFAULT_MAX_FRAMES = [8, 12, 24, 48]
DEFAULT_STRATEGIES = ["fixed", "beats", "novelty", "structure"]
DEFAULT_AUDIO_SOURCE = "mix"


def _parse_csv_list(value: str) -> List[str]:
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def _parse_int_list(values: Iterable[str]) -> List[int]:
    items: List[int] = []
    for value in values:
        items.extend(int(v) for v in _parse_csv_list(value))
    return items


def _command_to_text(cmd: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return " ".join(cmd)


def _strategy_suffix(strategy: str, max_frames: int) -> str:
    if strategy == "fixed":
        return f"{strategy}-frames{max_frames}.mp4"
    return f"{strategy}-max{max_frames}.mp4"


def _build_input_paths(prefix: str, max_frames: int) -> List[Path]:
    paths = []
    for strategy in DEFAULT_STRATEGIES:
        suffix = _strategy_suffix(strategy, max_frames)
        paths.append(Path(f"{prefix}-{suffix}"))
    return paths


def _build_ffmpeg_cmd(
    *,
    ffmpeg: str,
    inputs: List[Path],
    output: Path,
    audio_source: str,
) -> List[str]:
    scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    filter_parts = []
    for idx in range(4):
        filter_parts.append(f"[{idx}:v]{scale_filter}[v{idx}]")
    layout = "0_0|w0_0|0_h0|w0_h0"
    filter_parts.append("[v0][v1][v2][v3]xstack=inputs=4:layout=" + layout + "[v]")
    if audio_source == "mix":
        left_chain = (
            "aformat=channel_layouts=stereo,"
            "pan=mono|c0=0.5*(c0+c1),"
            "volume=0.5,"
            "pan=stereo|c0=c0|c1=0"
        )
        right_chain = (
            "aformat=channel_layouts=stereo,"
            "pan=mono|c0=0.5*(c0+c1),"
            "volume=0.5,"
            "pan=stereo|c0=0|c1=c0"
        )
        filter_parts.append(f"[0:a]{left_chain}[a0]")
        filter_parts.append(f"[1:a]{right_chain}[a1]")
        filter_parts.append(f"[2:a]{left_chain}[a2]")
        filter_parts.append(f"[3:a]{right_chain}[a3]")
        filter_parts.append("[a0][a1][a2][a3]amix=inputs=4:normalize=0[a]")
    filter_complex = ";".join(filter_parts)

    cmd = [ffmpeg, "-y"]
    for path in inputs:
        cmd += ["-i", str(path)]
    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
    ]

    if audio_source == "none":
        cmd += ["-an"]
    elif audio_source == "mix":
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    else:
        audio_idx = DEFAULT_STRATEGIES.index(audio_source)
        cmd += ["-map", f"{audio_idx}:a?", "-c:a", "aac", "-b:a", "192k"]

    cmd.append(str(output))
    return cmd


def _run_command(cmd: List[str], *, dry_run: bool) -> None:
    print(_command_to_text(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a 2x2 grid comparison for fixed/beats/novelty/structure outputs."
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Output prefix used by test_block_strategies.py.",
    )
    parser.add_argument(
        "--max-frames",
        action="append",
        default=[],
        help="Comma-separated list of max frames (repeatable).",
    )
    parser.add_argument(
        "--audio-source",
        choices=["mix", "none", *DEFAULT_STRATEGIES],
        default=DEFAULT_AUDIO_SOURCE,
        help="Audio mode: mix (panned), none, or keep a single input's audio.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable to use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    max_frames_list = _parse_int_list(args.max_frames) or DEFAULT_MAX_FRAMES
    prefix = str(args.prefix)

    for max_frames in max_frames_list:
        inputs = _build_input_paths(prefix, max_frames)
        missing = [str(p) for p in inputs if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing input videos:\n" + "\n".join(missing)
            )

        out_path = Path(f"{prefix}-grid-max{max_frames}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = _build_ffmpeg_cmd(
            ffmpeg=args.ffmpeg,
            inputs=inputs,
            output=out_path,
            audio_source=args.audio_source,
        )
        _run_command(cmd, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
