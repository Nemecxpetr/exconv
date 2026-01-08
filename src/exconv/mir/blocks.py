from __future__ import annotations

from typing import Literal, Sequence
import math

import numpy as np

from exconv.dsp.windows import hann
from scipy.signal import fftconvolve, find_peaks, stft

BlockStrategy = Literal["fixed", "beats", "novelty", "structure"]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] > 1:
        return np.mean(arr, axis=1, dtype=np.float32)
    return np.squeeze(arr).astype(np.float32, copy=False)


def _stft_mag(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    nperseg = int(n_fft)
    noverlap = int(max(0, n_fft - hop_length))
    _, _, z = stft(
        audio,
        fs=float(sr),
        nperseg=nperseg,
        noverlap=noverlap,
        window=hann(nperseg),
        boundary=None,
        padded=False,
    )
    return np.abs(z).astype(np.float32)


def spectral_flux_novelty(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    log_compress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Fast onset/novelty curve based on positive spectral flux.
    """
    mono = _to_mono(audio)
    mag = _stft_mag(mono, sr, n_fft=n_fft, hop_length=hop_length)
    if log_compress:
        mag = np.log1p(mag)
    if mag.shape[1] < 2:
        novelty = np.zeros((mag.shape[1],), dtype=np.float32)
    else:
        diff = np.diff(mag, axis=1)
        diff = np.maximum(diff, 0.0)
        novelty = np.concatenate([np.zeros((1,), dtype=np.float32), diff.sum(axis=0)])
    novelty = novelty.astype(np.float32, copy=False)
    if normalize and novelty.size > 0:
        peak = float(np.max(novelty))
        if peak > 0:
            novelty = novelty / peak
    return novelty


def peak_pick(
    novelty: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    threshold: float = 0.3,
    min_distance_s: float = 0.2,
    smooth: int = 5,
    adaptive: bool = False,
    adaptive_window: int = 32,
) -> np.ndarray:
    """
    Simple peak picker with optional smoothing and distance in seconds.
    """
    x = np.asarray(novelty, dtype=np.float32)
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if smooth > 1:
        kernel = np.ones((smooth,), dtype=np.float32) / float(smooth)
        x = np.convolve(x, kernel, mode="same")
    if adaptive:
        win = int(adaptive_window)
        if win <= 1:
            win = 3
        if win % 2 == 0:
            win += 1
        kernel = np.ones((win,), dtype=np.float32) / float(win)
        baseline = np.convolve(x, kernel, mode="same")
        x = np.maximum(x - baseline, 0.0)
        peak = float(np.max(x)) if x.size > 0 else 0.0
        if peak > 0:
            x = x / peak
    if min_distance_s is not None:
        distance = max(1, int(round(min_distance_s * sr / hop_length)))
    else:
        distance = 1
    peaks, _ = find_peaks(x, height=threshold, distance=distance)
    return peaks.astype(np.int64, copy=False)


def estimate_tempo(
    novelty: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    bpm_min: float = 60.0,
    bpm_max: float = 200.0,
) -> float | None:
    """
    Estimate tempo from novelty autocorrelation.
    """
    x = np.asarray(novelty, dtype=np.float32)
    if x.size < 4:
        return None
    x = x - float(np.mean(x))
    ac = fftconvolve(x, x[::-1], mode="full")
    ac = ac[len(x) - 1 :]
    if ac.size < 2:
        return None
    ac[0] = 0.0
    lag_min = int(round((60.0 * sr) / (bpm_max * hop_length)))
    lag_max = int(round((60.0 * sr) / (bpm_min * hop_length)))
    lag_min = max(1, lag_min)
    lag_max = min(int(ac.size - 1), lag_max)
    if lag_max <= lag_min:
        return None
    idx = int(np.argmax(ac[lag_min : lag_max + 1])) + lag_min
    if idx <= 0:
        return None
    bpm = (60.0 * sr) / (idx * hop_length)
    return float(bpm)


def _structure_features(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    mag = _stft_mag(_to_mono(audio), sr, n_fft=n_fft, hop_length=hop_length)
    if mag.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    feat = np.log1p(mag).T
    return feat.astype(np.float32, copy=False)


def structure_novelty(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    kernel_size: int = 16,
    max_frames: int = 600,
) -> tuple[np.ndarray, int]:
    """
    Novelty curve via self-similarity + checkerboard kernel.

    Returns (novelty, stride), where stride maps novelty frames to original
    feature frames when downsampling is applied.
    """
    features = _structure_features(audio, sr, n_fft=n_fft, hop_length=hop_length)
    n_frames = int(features.shape[0])
    if n_frames == 0:
        return np.zeros((0,), dtype=np.float32), 1
    stride = 1
    if max_frames and n_frames > max_frames:
        stride = int(math.ceil(n_frames / float(max_frames)))
        features = features[::stride]
        n_frames = int(features.shape[0])

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    feats = features / norms
    ssm = feats @ feats.T

    k = int(kernel_size)
    if k <= 1 or n_frames < (2 * k + 1):
        novelty = np.zeros((n_frames,), dtype=np.float32)
        return novelty, stride

    win = hann(2 * k)
    weight = np.outer(win, win).astype(np.float32)
    kernel = np.ones((2 * k, 2 * k), dtype=np.float32)
    kernel[:k, :k] = 1.0
    kernel[k:, k:] = 1.0
    kernel[:k, k:] = -1.0
    kernel[k:, :k] = -1.0
    kernel *= weight

    novelty = np.zeros((n_frames,), dtype=np.float32)
    for t in range(k, n_frames - k):
        sub = ssm[t - k : t + k, t - k : t + k]
        novelty[t] = float(np.sum(sub * kernel))
    peak = float(np.max(novelty)) if novelty.size > 0 else 0.0
    if peak > 0:
        novelty = novelty / peak
    return novelty.astype(np.float32, copy=False), stride


def _boundaries_to_segments(
    boundaries: Sequence[int],
    n_frames: int,
    *,
    min_block_frames: int,
    max_block_frames: int | None,
) -> list[tuple[int, int]]:
    if n_frames <= 0:
        return []
    if min_block_frames <= 0:
        raise ValueError("min_block_frames must be > 0")
    if max_block_frames is not None and max_block_frames <= 0:
        raise ValueError("max_block_frames must be > 0")

    starts = sorted({int(b) for b in boundaries if 0 <= int(b) < n_frames})
    if not starts or starts[0] != 0:
        starts = [0] + starts
    if starts[-1] != n_frames:
        starts.append(n_frames)

    segments = [(s, e - 1) for s, e in zip(starts, starts[1:]) if e > s]
    if not segments:
        return [(0, n_frames - 1)]

    merged: list[tuple[int, int]] = []
    for seg in segments:
        seg_len = seg[1] - seg[0] + 1
        if merged and seg_len < min_block_frames:
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1])
        else:
            merged.append(seg)

    if merged and (merged[0][1] - merged[0][0] + 1) < min_block_frames and len(merged) > 1:
        first = merged.pop(0)
        merged[0] = (first[0], merged[0][1])

    if max_block_frames is None:
        return merged

    split: list[tuple[int, int]] = []
    for seg in merged:
        start, end = seg
        length = end - start + 1
        if length <= max_block_frames:
            split.append(seg)
            continue
        cur = start
        while cur <= end:
            sub_end = min(cur + max_block_frames - 1, end)
            split.append((cur, sub_end))
            cur = sub_end + 1
    return split


def audio_to_frame_blocks(
    audio: np.ndarray,
    sr: int,
    fps: float,
    *,
    n_frames: int | None,
    strategy: BlockStrategy,
    min_block_frames: int = 1,
    max_block_frames: int | None = None,
    beats_per_block: int = 1,
    hop_length: int = 512,
    n_fft: int = 2048,
    peak_threshold: float = 0.3,
    peak_distance_s: float = 0.2,
    structure_kernel: int = 16,
    structure_max_frames: int = 600,
    bpm_min: float = 60.0,
    bpm_max: float = 200.0,
) -> list[tuple[int, int]]:
    """
    Resolve audio-driven frame blocks for a video.

    Returns a list of (start_frame, end_frame) pairs (inclusive).
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if beats_per_block <= 0:
        raise ValueError("beats_per_block must be > 0")

    if n_frames is None:
        n_frames = int(round(len(audio) * fps / float(sr)))

    if n_frames <= 0:
        return []

    if strategy == "fixed":
        return [(0, n_frames - 1)]

    if strategy == "structure":
        novelty, stride = structure_novelty(
            audio,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            kernel_size=structure_kernel,
            max_frames=structure_max_frames,
        )
        peaks = peak_pick(
            novelty,
            sr,
            hop_length * stride,
            threshold=peak_threshold,
            min_distance_s=peak_distance_s,
        )
        times = (peaks * stride * hop_length) / float(sr)
        boundaries = np.round(times * fps).astype(np.int64)
        return _boundaries_to_segments(
            boundaries,
            n_frames,
            min_block_frames=min_block_frames,
            max_block_frames=max_block_frames,
        )

    novelty = spectral_flux_novelty(
        audio,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    if strategy == "novelty":
        peaks = peak_pick(
            novelty,
            sr,
            hop_length,
            threshold=peak_threshold,
            min_distance_s=peak_distance_s,
        )
        times = (peaks * hop_length) / float(sr)
        boundaries = np.round(times * fps).astype(np.int64)
        return _boundaries_to_segments(
            boundaries,
            n_frames,
            min_block_frames=min_block_frames,
            max_block_frames=max_block_frames,
        )

    if strategy == "beats":
        bpm = estimate_tempo(
            novelty,
            sr,
            hop_length,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
        )
        if bpm is None or bpm <= 0:
            bpm = (bpm_min + bpm_max) * 0.5
        period = int(round((60.0 * sr) / (bpm * hop_length)))
        period = max(1, period)
        min_dist_s = 0.5 * (period * hop_length) / float(sr)
        adaptive_window = max(9, min(256, int(round(period * 2))))
        peaks = peak_pick(
            novelty,
            sr,
            hop_length,
            threshold=peak_threshold,
            min_distance_s=min_dist_s,
            adaptive=True,
            adaptive_window=adaptive_window,
        )
        if peaks.size == 0:
            peaks = np.arange(0, novelty.shape[0], period, dtype=np.int64)
        beats = peaks[::beats_per_block]
        if beats.size < 2:
            step = max(1, period * beats_per_block)
            beats = np.arange(0, novelty.shape[0], step, dtype=np.int64)
        times = (beats * hop_length) / float(sr)
        boundaries = np.round(times * fps).astype(np.int64)
        return _boundaries_to_segments(
            boundaries,
            n_frames,
            min_block_frames=min_block_frames,
            max_block_frames=max_block_frames,
        )

    raise ValueError(f"Unknown block strategy: {strategy}")


def quick_dtw_path(
    x: np.ndarray,
    y: np.ndarray,
    *,
    band: int | None = None,
    downsample: int = 1,
) -> tuple[np.ndarray, float]:
    """
    Lightweight DTW with optional band constraint and downsampling.

    Returns (path, cost) where path is an array of (i, j) index pairs.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]

    if downsample > 1:
        x_arr = x_arr[::downsample]
        y_arr = y_arr[::downsample]

    n = int(x_arr.shape[0])
    m = int(y_arr.shape[0])
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=np.int64), float("inf")

    if band is None:
        band = max(n, m)
    band = max(int(band), abs(n - m))

    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        xi = x_arr[i - 1]
        for j in range(j_start, j_end + 1):
            yj = y_arr[j - 1]
            cost = float(np.linalg.norm(xi - yj))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        diag = dtw[i - 1, j - 1]
        up = dtw[i - 1, j]
        left = dtw[i, j - 1]
        if diag <= up and diag <= left:
            i -= 1
            j -= 1
        elif up <= left:
            i -= 1
        else:
            j -= 1

    path.reverse()
    path_arr = np.asarray(path, dtype=np.int64)
    if downsample > 1 and path_arr.size > 0:
        path_arr = path_arr * int(downsample)
    return path_arr, float(dtw[n, m])
