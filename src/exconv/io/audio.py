# exconv/io/audio.py
"""
Audio I/O utilities using soundfile."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple, Union, Optional

import numpy as np
import soundfile as sf

ArrayLike = np.ndarray
PathLike = Union[str, Path]
TimeUnit = Literal["seconds", "samples"]

__all__ = [
    "read_audio",
    "read_segment",
    "write_audio",
    "to_mono",
    "to_stereo",
]


def _pathify(path: PathLike) -> str:
    return str(Path(path))


def read_audio(
    path: PathLike,
    *,
    dtype: str = "float32",
    always_2d: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Read an audio file (WAV/FLAC/anything libsndfile supports) via soundfile.

    Parameters
    ----------
    path : str or Path
        Input file.
    dtype : str, default="float32"
        Data type passed to soundfile. Typical options: "float32", "float64".
    always_2d : bool, default=False
        If True, always return shape (N, C). If False and C == 1, returns (N,).

    Returns
    -------
    data : ndarray
        Audio samples, shape (N,) or (N, C).
    sr : int
        Sample rate in Hz.
    """
    data, sr = sf.read(
        _pathify(path),
        dtype=dtype,
        always_2d=True,
    )

    if not always_2d and data.shape[1] == 1:
        data = data[:, 0]

    return np.asarray(data), int(sr)


def read_segment(
    path: PathLike,
    *,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    unit: TimeUnit = "seconds",
    dtype: str = "float32",
    always_2d: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Read a time segment from an audio file without loading the whole file.

    Parameters
    ----------
    path : str or Path
        Input file.
    start, stop : float or None
        Segment boundaries. Interpretation depends on `unit`:
        - "seconds": time in seconds.
        - "samples": sample indices.
        If None, defaults to file start or end.
    unit : {"seconds", "samples"}, default="seconds"
        Unit for start/stop.
    dtype : str, default="float32"
        Data type passed to soundfile.
    always_2d : bool, default=False
        If True, always return shape (N, C). If False and C == 1, returns (N,).

    Returns
    -------
    data : ndarray
        Audio segment, shape (N,) or (N, C).
    sr : int
        Sample rate in Hz.
    """
    path_str = _pathify(path)

    with sf.SoundFile(path_str, mode="r") as f:
        sr = f.samplerate
        n_frames = len(f)

        if start is None and stop is None:
            # whole file
            frames = f.read(frames=n_frames, dtype=dtype, always_2d=True)
            # Fix for mono + seek stride issue
            frames = np.array(frames, copy=True)
        else:
            if unit == "seconds":
                start_frame = 0 if start is None else int(round(start * sr))
                stop_frame = n_frames if stop is None else int(round(stop * sr))
            elif unit == "samples":
                start_frame = 0 if start is None else int(start)
                stop_frame = n_frames if stop is None else int(stop)
            else:
                raise ValueError(f"Unknown unit {unit!r}")

            # Clamp to valid range and ensure stop >= start
            start_frame = max(0, min(start_frame, n_frames))
            stop_frame = max(start_frame, min(stop_frame, n_frames))
            n_read = stop_frame - start_frame

            f.seek(start_frame)
            frames = f.read(frames=n_read, dtype=dtype, always_2d=True)

    if not always_2d and frames.shape[1] == 1:
        frames = frames[:, 0]

    return np.asarray(frames), int(sr)


def to_mono(x: ArrayLike) -> np.ndarray:
    """
    Convert multi-channel audio to mono by averaging channels.

    Parameters
    ----------
    x : ndarray, shape (N,) or (N, C)

    Returns
    -------
    mono : ndarray, shape (N,)
    """
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {arr.shape}")

    if arr.shape[1] == 1:
        return arr[:, 0]

    return arr.mean(axis=1)


def to_stereo(x: ArrayLike) -> np.ndarray:
    """
    Convert audio to stereo.

    Rules
    -----
    - (N,) or (N,1): duplicated into two identical channels → (N, 2)
    - (N,2): returned unchanged
    - (N,C>2): first two channels are kept

    Parameters
    ----------
    x : ndarray, shape (N,), (N,1), (N,2) or (N,C>2)

    Returns
    -------
    stereo : ndarray, shape (N, 2)
    """
    arr = np.asarray(x)

    if arr.ndim == 1:
        mono = arr
        return np.stack([mono, mono], axis=1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {arr.shape}")

    n, c = arr.shape
    if c == 1:
        mono = arr[:, 0]
        return np.stack([mono, mono], axis=1)
    if c == 2:
        return arr

    # C > 2: take first two channels
    return arr[:, :2]

def write_audio(
    path: PathLike,
    data: ArrayLike,
    sr: int,
    *,
    subtype: str = "PCM_16",
    clip: Optional[bool] = None,
    dtype: Optional[str] = None,
) -> None:
    """
    Write an audio file (WAV/FLAC/...) via soundfile.

    Parameters
    ----------
    path : str or Path
        Output file path; format is inferred from extension.
    data : ndarray, shape (N,) or (N, C)
        Audio samples. If 1D, treated as mono.
    sr : int
        Sample rate in Hz.
    subtype : str, default="PCM_16"
        libsndfile subtype, e.g. "PCM_16", "PCM_24", "FLOAT".
        For deterministic float round-tripping, use "FLOAT".
    clip : bool or None, default=None
        Clipping behavior for floating-point data:
        - None: clip only for non-FLOAT subtypes (PCM), no clip for "FLOAT".
        - True: always clip to [-1, 1] for float data.
        - False: never clip.
    dtype : str or None
        Optional dtype cast before writing, e.g. "float32".
    """
    arr = np.asarray(data)

    # Ensure (N, C)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {arr.shape}")

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    # Decide clipping for float data
    if np.issubdtype(arr.dtype, np.floating):
        if clip is None:
            # Default: clip for integer PCM, not for FLOAT subtype
            do_clip = (subtype.upper() != "FLOAT")
        else:
            do_clip = bool(clip)

        if do_clip:
            arr = np.clip(arr, -1.0, 1.0)

    sf.write(
        _pathify(path),
        arr,
        int(sr),
        subtype=subtype,
    )
