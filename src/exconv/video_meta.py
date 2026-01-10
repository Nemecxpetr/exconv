from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import subprocess

__all__ = [
    "ffprobe_available",
    "parse_fraction",
    "parse_float",
    "ffprobe_fps_info",
    "build_exconv_metadata",
    "ffmpeg_metadata_args",
]

_FFPROBE_AVAILABLE: bool | None = None


def ffprobe_available() -> bool:
    global _FFPROBE_AVAILABLE
    if _FFPROBE_AVAILABLE is not None:
        return _FFPROBE_AVAILABLE
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        _FFPROBE_AVAILABLE = True
    except (OSError, FileNotFoundError):
        _FFPROBE_AVAILABLE = False
    return _FFPROBE_AVAILABLE


def parse_fraction(val: object) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text or text.lower() == "n/a":
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
        except ValueError:
            return None
        if den_f == 0:
            return None
        return num_f / den_f
    try:
        return float(text)
    except ValueError:
        return None


def parse_float(val: object) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text or text.lower() == "n/a":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def ffprobe_fps_info(
    video_path: str | Path,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not ffprobe_available():
        return None, None, None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        res = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, FileNotFoundError):
        return None, None, None
    if res.returncode != 0 or not res.stdout:
        return None, None, None
    try:
        data = json.loads(res.stdout.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return None, None, None

    streams = data.get("streams") or []
    if not streams:
        return None, None, None
    stream = streams[0] or {}

    avg = parse_fraction(stream.get("avg_frame_rate"))
    r = parse_fraction(stream.get("r_frame_rate"))
    duration = parse_float(stream.get("duration"))
    if duration is None:
        duration = parse_float((data.get("format") or {}).get("duration"))
    return avg, r, duration


def build_exconv_metadata(
    process: str,
    variant: str,
    settings: dict[str, object],
) -> dict[str, str]:
    payload = {
        "process": process,
        "variant": variant,
        "settings": settings,
    }
    payload_json = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return {
        "exconv_process": str(process),
        "exconv_variant": str(variant),
        "exconv_settings": payload_json,
    }


def ffmpeg_metadata_args(metadata: dict[str, str] | None) -> list[str]:
    if not metadata:
        return []
    args: list[str] = []
    for key in sorted(metadata):
        value = metadata[key]
        if value is None:
            continue
        args += ["-metadata", f"{key}={value}"]
    return args
