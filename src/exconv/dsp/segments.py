from __future__ import annotations

from typing import Literal

import numpy as np

AudioLengthMode = Literal["trim", "pad-zero", "pad-loop", "pad-noise", "center-zero"]

__all__ = [
    "AudioLengthMode",
    "slice_for_frame",
    "frame_range_samples",
    "match_audio_length",
    "audio_chunk_for_interval",
]


def slice_for_frame(
    frame_idx: int,
    fps: float,
    sr: int,
    n_samples: int,
) -> tuple[int, int]:
    """
    Map a frame index to [start, stop) sample indices.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")

    t_start = frame_idx / fps
    t_end = (frame_idx + 1) / fps

    start = int(round(t_start * sr))
    stop = int(round(t_end * sr))

    start = max(0, min(start, n_samples))
    stop = max(start, min(stop, n_samples))

    return start, stop


def frame_range_samples(
    start_frame: int,
    end_frame: int,
    *,
    sr: int,
    fps: float,
    n_samples: int | None,
) -> tuple[int, int]:
    if end_frame < start_frame:
        return 0, 0
    start_s = int(round(start_frame * sr / fps))
    stop_s = int(round((end_frame + 1) * sr / fps))
    if n_samples is not None:
        start_s = max(0, min(start_s, n_samples))
        stop_s = max(start_s, min(stop_s, n_samples))
    return start_s, stop_s


def match_audio_length(
    audio: np.ndarray,
    target_len: int,
    *,
    mode: AudioLengthMode = "pad-zero",
) -> np.ndarray:
    """
    Trim or pad audio to target_len samples.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    n, c = audio.shape

    if mode == "trim":
        if n >= target_len:
            return audio[:target_len]
        pad = np.zeros((target_len - n, c), dtype=np.float32)
        return np.concatenate([audio, pad], axis=0)

    if mode == "pad-zero":
        if n >= target_len:
            return audio[:target_len]
        pad = np.zeros((target_len - n, c), dtype=np.float32)
        return np.concatenate([audio, pad], axis=0)

    if mode == "center-zero":
        if n >= target_len:
            trim = (n - target_len) // 2
            return audio[trim : trim + target_len]
        pad_front = (target_len - n) // 2
        pad_back = target_len - n - pad_front
        front = np.zeros((pad_front, c), dtype=np.float32)
        back = np.zeros((pad_back, c), dtype=np.float32)
        return np.concatenate([front, audio, back], axis=0)

    if mode == "pad-loop":
        if n >= target_len:
            return audio[:target_len]
        reps = int(np.ceil(target_len / n))
        tiled = np.tile(audio, (reps, 1))
        return tiled[:target_len]

    if mode == "pad-noise":
        if n >= target_len:
            return audio[:target_len]
        pad_len = target_len - n
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size > 0 else 1.0
        scale = 0.01 * rms if rms > 0 else 0.01
        noise = np.random.default_rng().normal(0.0, scale, size=(pad_len, c)).astype(
            np.float32
        )
        return np.concatenate([audio, noise], axis=0)

    raise ValueError(f"Unknown audio length mode: {mode}")


def audio_chunk_for_interval(
    audio: np.ndarray,
    start: int,
    stop: int,
    *,
    mode: AudioLengthMode,
    rng: np.random.Generator,
    noise_scale: float,
    center_target_len: int | None = None,
) -> np.ndarray:
    """
    Return audio[start:stop] for a logical timeline, applying padding strategy.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]

    if stop <= start:
        return np.zeros((0, audio.shape[1]), dtype=np.float32)

    n_samples, n_ch = audio.shape
    need = int(stop - start)

    if mode in ("trim", "pad-zero"):
        if n_samples <= 0:
            return np.zeros((need, n_ch), dtype=np.float32)
        if start >= n_samples:
            return np.zeros((need, n_ch), dtype=np.float32)
        take_end = min(stop, n_samples)
        head = audio[start:take_end, :]
        if take_end >= stop:
            return head.astype(np.float32, copy=False)
        pad = np.zeros((stop - take_end, n_ch), dtype=np.float32)
        return np.concatenate([head, pad], axis=0)

    if mode == "pad-loop":
        if n_samples <= 0:
            return np.zeros((need, n_ch), dtype=np.float32)
        out = np.empty((need, n_ch), dtype=np.float32)
        pos = 0
        cur = int(start)
        while pos < need:
            idx = cur % n_samples
            take = min(need - pos, n_samples - idx)
            out[pos : pos + take, :] = audio[idx : idx + take, :]
            pos += take
            cur += take
        return out

    if mode == "pad-noise":
        if n_samples <= 0:
            return rng.normal(0.0, noise_scale, size=(need, n_ch)).astype(np.float32)
        if start >= n_samples:
            return rng.normal(0.0, noise_scale, size=(need, n_ch)).astype(np.float32)
        take_end = min(stop, n_samples)
        head = audio[start:take_end, :]
        if take_end >= stop:
            return head.astype(np.float32, copy=False)
        tail = rng.normal(0.0, noise_scale, size=(stop - take_end, n_ch)).astype(
            np.float32
        )
        return np.concatenate([head, tail], axis=0)

    if mode == "center-zero":
        if center_target_len is None or center_target_len <= 0:
            raise ValueError("center-zero requires a valid target duration")

        target = int(center_target_len)
        if need <= 0:
            return np.zeros((0, n_ch), dtype=np.float32)

        if n_samples >= target:
            trim = (n_samples - target) // 2
            src_start = trim + start
            src_stop = trim + stop
            src_start = max(0, min(src_start, n_samples))
            src_stop = max(src_start, min(src_stop, n_samples))
            head = audio[src_start:src_stop, :]
            if head.shape[0] >= need:
                return head[:need].astype(np.float32, copy=False)
            pad = np.zeros((need - head.shape[0], n_ch), dtype=np.float32)
            return np.concatenate([head, pad], axis=0)

        pad_front = (target - n_samples) // 2
        out = np.zeros((need, n_ch), dtype=np.float32)
        src_start = start - pad_front
        src_stop = stop - pad_front
        a0 = max(0, src_start)
        a1 = min(n_samples, src_stop)
        if a1 <= a0:
            return out
        o0 = a0 - src_start
        o1 = o0 + (a1 - a0)
        out[o0:o1, :] = audio[a0:a1, :]
        return out

    raise ValueError(f"Unknown audio length mode: {mode}")
