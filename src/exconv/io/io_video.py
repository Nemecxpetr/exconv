# src/exconv/io_video.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import imageio.v3 as iio


def _as_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def read_video_frames(
    path: str | Path,
) -> Tuple[List[np.ndarray], float]:
    """
    Read a video into a list of frames and return (frames, fps).

    Frames are HxWxC uint8 arrays.

    If FPS metadata is missing, the returned fps will be 0.0 and callers
    are expected to override it.
    """
    path = _as_path(path)
    meta = iio.immeta(path)
    fps_meta = meta.get("fps", None)

    try:
        fps = float(fps_meta) if fps_meta is not None else 0.0
    except (TypeError, ValueError):
        fps = 0.0

    frames = list(iio.imiter(path))
    # Ensure uint8 frames for consistency
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]

    return frames, fps


def write_video_frames(
    path: str | Path,
    frames: Iterable[np.ndarray],
    fps: float,
) -> None:
    """
    Write a sequence of frames (HxWxC) to a video file at given fps.

    Frames are converted to uint8 if needed.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")

    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames_list = [np.asarray(f, dtype=np.uint8) for f in frames]
    if not frames_list:
        raise ValueError("No frames to write.")

    iio.imwrite(path, frames_list, fps=fps)
