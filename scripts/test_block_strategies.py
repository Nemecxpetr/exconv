from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable, List


DEFAULT_VIDEO = (
    r".\samples\input\video\synestezie\document_5974044649671629709.mp4"
)
DEFAULT_OUT_PREFIX = ".\\samples\\output\\video\\shu_closeup"
DEFAULT_STRATEGIES = ["fixed", "beats", "novelty", "structure"]
DEFAULT_MAX_FRAMES = [8, 12, 24, 48]
DEFAULT_BLOCK_MIN_FRAMES = 1
DEFAULT_COMMON_ARGS = [
    "--mux",
    "--serial-mode",
    "parallel",
    "--s2i-mode",
    "stereo",
    "--s2i-colorspace",
    "color",
    "--i2s-mode",
    "radial",
    "--i2s-colorspace",
    "ycbcr-mid-side",
    "--i2s-phase-mode",
    "spiral",
    "--i2s-impulse-len",
    "auto",
    "--s2i-chroma-strength",
    "0.01",
    "--s2i-chroma-clip",
    "0.012",
]


def _parse_csv_list(value: str) -> List[str]:
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def _parse_int_list(values: Iterable[str]) -> List[int]:
    items: List[int] = []
    for value in values:
        items.extend(int(v) for v in _parse_csv_list(value))
    return items


def _parse_str_list(values: Iterable[str]) -> List[str]:
    items: List[str] = []
    for value in values:
        items.extend(_parse_csv_list(value))
    return items


def _command_to_text(cmd: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return " ".join(cmd)


def _build_out_path(prefix: str, strategy: str, max_frames: int, ext: str = ".mp4") -> str:
    if strategy == "fixed":
        suffix = f"{strategy}-frames{max_frames}{ext}"
    else:
        suffix = f"{strategy}-max{max_frames}{ext}"
    return f"{prefix}-{suffix}"


def _run_command(cmd: List[str], *, dry_run: bool) -> None:
    print(_command_to_text(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch test video-biconv block strategies with default settings."
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help="Input video path.",
    )
    parser.add_argument(
        "--out-prefix",
        default=DEFAULT_OUT_PREFIX,
        help="Output prefix (path without suffix/extension).",
    )
    parser.add_argument(
        "--strategies",
        action="append",
        default=[],
        help="Comma-separated list (repeatable).",
    )
    parser.add_argument(
        "--max-frames",
        action="append",
        default=[],
        help="Comma-separated list of max frames (repeatable).",
    )
    parser.add_argument(
        "--block-min-frames",
        type=int,
        default=DEFAULT_BLOCK_MIN_FRAMES,
        help="Minimum block length for audio-driven strategies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    strategies = _parse_str_list(args.strategies) or DEFAULT_STRATEGIES
    max_frames = _parse_int_list(args.max_frames) or DEFAULT_MAX_FRAMES

    for strategy in strategies:
        for max_frame in max_frames:
            out_path = _build_out_path(args.out_prefix, strategy, max_frame)
            cmd = [
                "exconv",
                "video-biconv",
                "--video",
                args.video,
                "--out-video",
                out_path,
                *DEFAULT_COMMON_ARGS,
                "--block-strategy",
                strategy,
            ]
            if strategy == "fixed":
                cmd += ["--block-size", str(max_frame)]
            else:
                cmd += [
                    "--block-min-frames",
                    str(args.block_min_frames),
                    "--block-max-frames",
                    str(max_frame),
                ]
            _run_command(cmd, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
