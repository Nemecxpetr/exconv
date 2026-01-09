from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List

FFMPEG_EXE = r"C:\ffmpeg\bin\ffmpeg.exe"
VIDEO_1 = r"E:\Documents\Kompozice\2025\AUDIOKEZKOUSKAM\video1.mp4"
VIDEO_2 = r"E:\Documents\Kompozice\2025\AUDIOKEZKOUSKAM\video2.mp4"
OUTPUT = r"E:\Documents\Kompozice\2025\AUDIOKEZKOUSKAM\joined.mp4"
METHOD = "filter"  # "filter" (re-encode) or "concat" (stream copy).
INCLUDE_AUDIO = True
OVERWRITE = True
TARGET_WIDTH = 640
TARGET_HEIGHT = 360


def _command_to_text(cmd: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return " ".join(cmd)


def _build_filter_cmd(
    ffmpeg: str,
    input_a: Path,
    input_b: Path,
    output: Path,
    *,
    include_audio: bool,
    overwrite: bool,
) -> List[str]:
    video_chain = (
        f"scale=w={TARGET_WIDTH}:h={TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        "setsar=1,setpts=PTS-STARTPTS"
    )
    audio_chain = (
        "aformat=sample_rates=48000:channel_layouts=stereo,asetpts=PTS-STARTPTS"
    )

    if include_audio:
        filter_complex = (
            f"[0:v]{video_chain}[v0];"
            f"[1:v]{video_chain}[v1];"
            f"[0:a]{audio_chain}[a0];"
            f"[1:a]{audio_chain}[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )
    else:
        filter_complex = (
            f"[0:v]{video_chain}[v0];"
            f"[1:v]{video_chain}[v1];"
            "[v0][v1]concat=n=2:v=1:a=0[v]"
        )

    cmd = [ffmpeg]
    if overwrite:
        cmd.append("-y")
    cmd += [
        "-i",
        str(input_a),
        "-i",
        str(input_b),
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
    ]

    if include_audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd.append("-an")

    cmd.append(str(output))
    return cmd


def _write_concat_list(path: Path, inputs: List[Path]) -> None:
    lines = []
    for item in inputs:
        normalized = item.resolve().as_posix()
        lines.append(f"file '{normalized}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_concat_cmd(
    ffmpeg: str,
    list_path: Path,
    output: Path,
    *,
    overwrite: bool,
) -> List[str]:
    cmd = [ffmpeg]
    if overwrite:
        cmd.append("-y")
    cmd += [
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(output),
    ]
    return cmd


def _run_command(cmd: List[str]) -> None:
    print(_command_to_text(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    inputs = [Path(VIDEO_1), Path(VIDEO_2)]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input videos:\n" + "\n".join(missing))

    output_path = Path(OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if METHOD == "filter":
        cmd = _build_filter_cmd(
            FFMPEG_EXE,
            inputs[0],
            inputs[1],
            output_path,
            include_audio=INCLUDE_AUDIO,
            overwrite=OVERWRITE,
        )
    elif METHOD == "concat":
        list_path = output_path.with_suffix(".concat.txt")
        _write_concat_list(list_path, inputs)
        cmd = _build_concat_cmd(
            FFMPEG_EXE,
            list_path,
            output_path,
            overwrite=OVERWRITE,
        )
    else:
        raise ValueError("METHOD must be 'filter' or 'concat'")

    _run_command(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
