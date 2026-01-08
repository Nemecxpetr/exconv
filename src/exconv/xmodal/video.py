# src/exconv/xmodal/video.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Literal, overload, Optional
import json
import subprocess
import tempfile
from io import BytesIO
import math

import numpy as np
import soundfile as sf
from tqdm import tqdm

from exconv.io import read_audio, write_audio, read_video_frames, write_video_frames
from exconv.conv1d.audio import Audio, auto_convolve as audio_auto_convolve
from .sound2image import spectral_sculpt, Mode, ColorMode
from .image2sound import (
    image2sound_flat,
    image2sound_hist,
    image2sound_radial,
    PadMode as Img2SoundPadMode,
    ImpulseNorm,
    OutNorm,
    ColorMode as Img2SoundColorMode,
    RadiusMode,
    PhaseMode,
    SmoothingMode,
)


AudioVideoMode = Literal["per-buffer-auto", "original"]
BlockStrategy = Literal["fixed", "beats", "novelty", "structure"]

__all__ = [
    "AudioVideoMode",
    "BlockStrategy",
    "sound2image_video_arrays",
    "sound2image_video_from_files",
    "biconv_video_arrays",
    "biconv_video_from_files",
    "biconv_video_to_files_stream",
]


def _as_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except (OSError, FileNotFoundError):
        return False


_FFPROBE_AVAILABLE: bool | None = None
_FPS_GUARD_RATIO = 1.2

FPSPolicy = Literal["auto", "metadata", "avg_frame_rate", "r_frame_rate"]


def _ffprobe_available() -> bool:
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


def _parse_fraction(val: object) -> Optional[float]:
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


def _parse_float(val: object) -> Optional[float]:
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


def _ffprobe_fps_info(video_path: Path) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not _ffprobe_available():
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

    avg = _parse_fraction(stream.get("avg_frame_rate"))
    r = _parse_fraction(stream.get("r_frame_rate"))
    duration = _parse_float(stream.get("duration"))
    if duration is None:
        duration = _parse_float((data.get("format") or {}).get("duration"))
    return avg, r, duration


def _resolve_fps_for_video(
    video_path: Path,
    *,
    meta: dict,
    fps_override: float | None,
    fps_policy: FPSPolicy,
    n_frames: int | None = None,
) -> float:
    if fps_override is not None:
        fps_val = float(fps_override)
        if fps_val <= 0:
            raise ValueError("fps must be > 0")
        return fps_val

    meta_fps = _parse_float(meta.get("fps"))
    meta_duration = _parse_float(meta.get("duration"))

    ff_avg: Optional[float] = None
    ff_r: Optional[float] = None
    ff_duration: Optional[float] = None
    if fps_policy != "metadata":
        ff_avg, ff_r, ff_duration = _ffprobe_fps_info(video_path)

    duration = meta_duration if meta_duration and meta_duration > 0 else ff_duration
    fps_from_frames: Optional[float] = None
    if n_frames is not None and duration and duration > 0:
        fps_from_frames = float(n_frames) / float(duration)

    def _valid(val: Optional[float]) -> bool:
        return val is not None and val > 0

    if fps_policy == "metadata":
        if _valid(meta_fps):
            return float(meta_fps)
        raise RuntimeError("Could not determine FPS; pass fps explicitly.")

    if fps_policy == "avg_frame_rate":
        for candidate in (ff_avg, meta_fps, fps_from_frames, ff_r):
            if _valid(candidate):
                return float(candidate)
        raise RuntimeError("Could not determine FPS; pass fps explicitly.")

    if fps_policy == "r_frame_rate":
        for candidate in (ff_r, meta_fps, fps_from_frames, ff_avg):
            if _valid(candidate):
                return float(candidate)
        raise RuntimeError("Could not determine FPS; pass fps explicitly.")

    if _valid(ff_avg) and _valid(ff_r):
        ratio = max(ff_r / ff_avg, ff_avg / ff_r)
        if ratio >= _FPS_GUARD_RATIO:
            return float(ff_r)

    if _valid(meta_fps) and _valid(fps_from_frames):
        ratio = max(meta_fps / fps_from_frames, fps_from_frames / meta_fps)
        if ratio >= _FPS_GUARD_RATIO:
            return float(fps_from_frames)

    for candidate in (meta_fps, ff_avg, ff_r, fps_from_frames):
        if _valid(candidate):
            return float(candidate)

    raise RuntimeError("Could not determine FPS; pass fps explicitly.")

_VIDEO_EXTS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".ogv",
}


def _looks_like_video(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_EXTS


def _extract_audio_from_video(
    video_path: Path, target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Try to extract audio from a video. Prefers ffmpeg pipe; falls back to read_audio.
    """
    if _ffmpeg_available():
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
        ]
        if target_sr:
            cmd += ["-ar", str(target_sr)]
        cmd += ["-f", "wav", "-"]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            data = BytesIO(res.stdout)
            audio, sr = sf.read(data, dtype="float32", always_2d=True)
            return audio, sr
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode(errors="ignore") if exc.stderr else ""
            raise RuntimeError(
                f"ffmpeg failed to extract audio from {video_path}: {err}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"ffmpeg succeeded but audio decode failed for {video_path}"
            ) from exc

    # fallback: try to read directly (may fail for mp4)
    try:
        audio, sr = read_audio(video_path, dtype="float32", always_2d=True)
        return audio, sr
    except Exception as exc:
        raise RuntimeError(
            "Could not extract audio from video. Install ffmpeg or pass --audio explicitly."
        ) from exc


def _slice_for_frame(
    frame_idx: int,
    fps: float,
    sr: int,
    n_samples: int,
) -> Tuple[int, int]:
    """
    Map a frame index to [start, stop) sample indices.

    The mapping is based on continuous time:

        t_start = frame_idx / fps
        t_end   = (frame_idx + 1) / fps

        start = round(t_start * sr)
        stop  = round(t_end * sr)

    and then clamped to [0, n_samples].
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


def _match_audio_length(
    audio: np.ndarray,
    target_len: int,
    *,
    mode: Literal["trim", "pad-zero", "pad-loop", "pad-noise", "center-zero"] = "pad-zero",
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
        # noise scaled to 1% rms of existing audio or unity if silent
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size > 0 else 1.0
        scale = 0.01 * rms if rms > 0 else 0.01
        noise = np.random.default_rng().normal(0.0, scale, size=(pad_len, c)).astype(
            np.float32
        )
        return np.concatenate([audio, noise], axis=0)

    raise ValueError(f"Unknown audio length mode: {mode}")


def _estimate_total_frames_from_meta(meta: dict, fps: float) -> int | None:
    """
    Best-effort estimate of total frames from imageio metadata.

    We intentionally bias upward (ceil) to reduce the risk of under-estimating,
    which could otherwise truncate audio too early if callers use the estimate.
    """
    nframes = meta.get("nframes", None)
    if isinstance(nframes, (int, np.integer)) and int(nframes) > 0:
        return int(nframes)

    try:
        nframes_f = float(nframes)  # may be inf/NaN
        if math.isfinite(nframes_f) and nframes_f > 0:
            return int(math.ceil(nframes_f))
    except (TypeError, ValueError):
        pass

    dur = meta.get("duration", None)
    try:
        dur_f = float(dur)
    except (TypeError, ValueError):
        return None

    if dur_f > 0 and fps > 0:
        return int(max(1, math.ceil(dur_f * fps)))
    return None


def _read_video_meta(path: Path) -> dict:
    try:
        import imageio.v3 as iio  # local import to keep import surface small

        meta = iio.immeta(path)
        return dict(meta) if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _audio_chunk_for_interval(
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

    Unlike `_slice_for_frame`, `start/stop` are NOT clamped to the source audio
    length. This lets us implement pad/loop/noise without pre-materializing a
    full-length padded array.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]

    if stop <= start:
        return np.zeros((0, audio.shape[1]), dtype=np.float32)

    n_samples, n_ch = audio.shape
    need = int(stop - start)

    if mode in ("trim", "pad-zero"):
        # For our use (only requesting within video timeline), trim behaves like
        # pad-zero: take what exists, and zero-pad past EOF.
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
        # Video timeline maps to audio index = t - pad_front
        out = np.zeros((need, n_ch), dtype=np.float32)
        src_start = start - pad_front
        src_stop = stop - pad_front
        # overlap with audio indices [0, n_samples)
        a0 = max(0, src_start)
        a1 = min(n_samples, src_stop)
        if a1 <= a0:
            return out
        o0 = a0 - src_start
        o1 = o0 + (a1 - a0)
        out[o0:o1, :] = audio[a0:a1, :]
        return out

    raise ValueError(f"Unknown audio length mode: {mode}")


def _segments_from_block_size(n_frames: int, block_size: int) -> list[tuple[int, int]]:
    if n_frames <= 0:
        return []
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    segments: list[tuple[int, int]] = []
    for start in range(0, n_frames, block_size):
        end = min(start + block_size, n_frames) - 1
        segments.append((start, end))
    return segments


def _resolve_block_segments(
    *,
    audio: np.ndarray,
    sr: int,
    fps: float,
    n_frames: int | None,
    block_size: int,
    block_size_div: int | None,
    block_strategy: BlockStrategy,
    block_min_frames: int,
    block_max_frames: int | None,
    block_beats_per: int,
    block_hop_length: int,
    block_n_fft: int,
    block_peak_threshold: float,
    block_peak_distance_s: float,
    block_structure_kernel: int,
    block_structure_max_frames: int,
    block_bpm_min: float,
    block_bpm_max: float,
) -> list[tuple[int, int]]:
    if block_strategy == "fixed":
        if n_frames is None:
            return []
        if block_size_div is not None:
            if block_size_div <= 0:
                raise ValueError("block_size_div must be positive")
            block_size = max(1, int(math.ceil(n_frames / float(block_size_div))))
        return _segments_from_block_size(n_frames, block_size)

    from exconv.mir.blocks import audio_to_frame_blocks

    return audio_to_frame_blocks(
        audio=audio,
        sr=sr,
        fps=fps,
        n_frames=n_frames,
        strategy=block_strategy,
        min_block_frames=block_min_frames,
        max_block_frames=block_max_frames,
        beats_per_block=block_beats_per,
        hop_length=block_hop_length,
        n_fft=block_n_fft,
        peak_threshold=block_peak_threshold,
        peak_distance_s=block_peak_distance_s,
        structure_kernel=block_structure_kernel,
        structure_max_frames=block_structure_max_frames,
        bpm_min=block_bpm_min,
        bpm_max=block_bpm_max,
    )


def _image2sound_apply(
    img: np.ndarray,
    audio: np.ndarray,
    sr: int,
    *,
    mode: Literal["flat", "hist", "radial"] = "radial",
    pad_mode: Img2SoundPadMode = "same-center",
    colorspace: Img2SoundColorMode = "luma",
    impulse_norm: ImpulseNorm = "energy",
    out_norm: OutNorm = "match_rms",
    impulse_len: int | Literal["auto"] = "auto",
    n_bins: int = 256,
    radius_mode: RadiusMode = "linear",
    phase_mode: PhaseMode = "zero",
    smoothing: SmoothingMode = "hann",
) -> np.ndarray:
    """
    Dispatch helper to apply image->sound using selected mode.
    """
    img = np.asarray(img)
    audio = np.asarray(audio, dtype=np.float32)
    if impulse_len == "auto":
        impulse_len_resolved = audio.shape[0]
    else:
        impulse_len_resolved = int(impulse_len)

    if mode == "flat":
        y, _, _ = image2sound_flat(
            audio=audio,
            sr=sr,
            img=img,
            pad_mode=pad_mode,
            colorspace=colorspace,
            impulse_norm=impulse_norm,
            out_norm=out_norm,
        )
    elif mode == "hist":
        y, _, _ = image2sound_hist(
            audio=audio,
            sr=sr,
            img=img,
            pad_mode=pad_mode,
            colorspace=colorspace,
            impulse_norm=impulse_norm,
            out_norm=out_norm,
            n_bins=n_bins,
        )
    elif mode == "radial":
        y, _, _ = image2sound_radial(
            audio=audio,
            sr=sr,
            img=img,
            pad_mode=pad_mode,
            colorspace=colorspace,
            impulse_len=impulse_len_resolved,
            radius_mode=radius_mode,
            phase_mode=phase_mode,
            smoothing=smoothing,
            impulse_norm=impulse_norm,
            out_norm=out_norm,
        )
    else:
        raise ValueError(f"Unknown image2sound mode: {mode}")

    if y.ndim == 1:
        y = y[:, None]
    return y.astype(np.float32)


@overload
def sound2image_video_arrays(
    frames: Sequence[np.ndarray],
    audio: np.ndarray,
    sr: int,
    fps: float,
    *,
    mode: Mode = "mono",
    colorspace: ColorMode = "luma",
    audio_out_mode: AudioVideoMode = "per-buffer-auto",
) -> Tuple[List[np.ndarray], np.ndarray]:
    ...


def sound2image_video_arrays(
    frames,
    audio,
    sr,
    fps,
    *,
    mode="mono",
    colorspace="luma",
    audio_out_mode="per-buffer-auto",
):
    """
    Apply sound->image spectral sculpting per frame, and build an audio track.

    Parameters
    ----------
    frames
        Sequence of image frames as HxWxC uint8 or float arrays.
    audio
        1D or 2D array of audio samples, shape (N,) or (N, C).
    sr
        Audio sample rate in Hz.
    fps
        Video frame rate.
    mode
        Audio->image mapping mode ("mono", "stereo", "mid-side").
        See `exconv.xmodal.sound2image.spectral_sculpt`.
    colorspace
        "luma" or "color" - see `spectral_sculpt`.
    audio_out_mode
        - "per-buffer-auto": auto-convolve each frame's chunk and concatenate.
        - "original": reuse original audio, trimmed / padded to video duration.

    Returns
    -------
    processed_frames, audio_out
        processed_frames : list of frames (uint8, HxWxC)
        audio_out        : 2D array of samples (N_out, C), float32
    """
    frames = list(frames)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim != 2:
        raise ValueError("audio must be 1D (N,) or 2D (N, C)")

    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr!r}")

    n_samples, n_channels = audio.shape
    n_frames = len(frames)

    processed_frames: List[np.ndarray] = []
    audio_out_chunks: List[np.ndarray] = []

    frame_iter = tqdm(frames, total=n_frames, desc="sound->image video", unit="frame")
    for idx, frame in enumerate(frame_iter):
        frame = np.asarray(frame)

        # --- slice audio for this frame ---
        start, stop = _slice_for_frame(idx, fps, sr, n_samples)

        if stop > start:
            chunk = audio[start:stop, :]
        else:
            # if audio shorter than video, use a small silence chunk so that
            # spectral_sculpt still receives something meaningful
            chunk_len = max(1, int(round(sr / fps)))
            chunk = np.zeros((chunk_len, n_channels), dtype=audio.dtype)

        # --- image processing: per-frame spectral sculpt ---
        out_frame = spectral_sculpt(
            image=frame,
            audio=chunk,
            sr=sr,
            mode=mode,           # type: ignore[arg-type]
            colorspace=colorspace,  # type: ignore[arg-type]
            normalize=True,
        )

        if np.issubdtype(out_frame.dtype, np.floating):
            frame_u8 = np.clip(out_frame * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            frame_u8 = np.clip(out_frame, 0, 255).astype(np.uint8)

        processed_frames.append(frame_u8)

        # --- audio processing: per-buffer auto-convolution or original ---
        if audio_out_mode == "per-buffer-auto":
            chunk_audio = Audio(samples=chunk, sr=sr)
            chunk_proc = audio_auto_convolve(
                chunk_audio,
                mode="same-center",
                circular=False,
                normalize="rms",
                order=2,
            )
            samples = np.asarray(chunk_proc.samples, dtype=np.float32)
            if samples.ndim == 1:
                samples = samples[:, None]
            audio_out_chunks.append(samples)

    # --- assemble output audio ---
    if audio_out_mode == "per-buffer-auto":
        if audio_out_chunks:
            audio_out = np.concatenate(audio_out_chunks, axis=0)
        else:
            audio_out = np.zeros_like(audio, dtype=np.float32)
    elif audio_out_mode == "original":
        # Match original audio length to video duration
        ideal_samples = int(round(n_frames * sr / fps))
        if ideal_samples <= n_samples:
            audio_out = audio[:ideal_samples, :]
        else:
            pad_len = ideal_samples - n_samples
            pad = np.zeros((pad_len, n_channels), dtype=np.float32)
            audio_out = np.concatenate([audio, pad], axis=0)
    else:
        raise ValueError(f"Unknown audio_out_mode: {audio_out_mode!r}")

    # ensure float32 output
    audio_out = np.asarray(audio_out, dtype=np.float32)

    return processed_frames, audio_out


def sound2image_video_from_files(
    video_path: str | Path,
    audio_path: str | Path,
    *,
    fps: float | None = None,
    fps_policy: FPSPolicy = "auto",
    mode: Mode = "mono",
    colorspace: ColorMode = "luma",
    audio_out_mode: AudioVideoMode = "per-buffer-auto",
    out_video: str | Path | None = None,
    out_audio: str | Path | None = None,
) -> Tuple[List[np.ndarray], np.ndarray, float, int]:
    """
    High-level helper that works on files (video + audio).

    Parameters
    ----------
    video_path, audio_path
        Input paths.
    fps
        Override FPS if metadata is missing/incorrect. If None, the FPS is
        selected according to fps_policy.
    fps_policy
        FPS selection policy when fps is None.
    mode, colorspace, audio_out_mode
        Passed through to `sound2image_video_arrays`.
    out_video, out_audio
        Optional output paths. If given, frames/audio are written using
        `write_video_frames` (video) and `write_audio` (audio).

    Returns
    -------
    frames_out, audio_out, fps_used, sr
        frames_out : list of processed frames (uint8, HxWxC)
        audio_out  : processed audio (N_out, C), float32
        fps_used   : FPS actually used
        sr         : audio sample rate
    """
    video_path = _as_path(video_path)
    audio_path = _as_path(audio_path)

    # --- audio ---
    audio, sr = read_audio(audio_path, dtype="float32", always_2d=True)

    # --- video frames + metadata ---
    frames, meta_fps = read_video_frames(video_path)
    meta = _read_video_meta(video_path)
    if meta_fps and not meta.get("fps"):
        meta["fps"] = float(meta_fps)
    fps_used = _resolve_fps_for_video(
        video_path,
        meta=meta,
        fps_override=fps,
        fps_policy=fps_policy,
        n_frames=len(frames),
    )

    frames_out, audio_out = sound2image_video_arrays(
        frames=frames,
        audio=audio,
        sr=sr,
        fps=fps_used,
        mode=mode,
        colorspace=colorspace,
        audio_out_mode=audio_out_mode,
    )

    # --- optional writing ---
    if out_video is not None:
        out_video = _as_path(out_video)
        write_video_frames(out_video, frames_out, fps=fps_used)

    if out_audio is not None:
        out_audio = _as_path(out_audio)
        write_audio(out_audio, audio_out, sr)

    return frames_out, audio_out, fps_used, sr


# ----------------------------------------------------------------------
# Bi-directional video processing (image<->sound per frame)
# ----------------------------------------------------------------------
DualSerialMode = Literal["parallel", "serial-image-first", "serial-sound-first"]
AudioLengthMode = Literal["trim", "pad-zero", "pad-loop", "pad-noise", "center-zero"]


def biconv_video_arrays(
    frames: Sequence[np.ndarray],
    audio: np.ndarray,
    sr: int,
    fps: float,
    *,
    # sound->image options
    s2i_mode: Mode = "mono",
    s2i_colorspace: ColorMode = "luma",
    # image->sound options
    i2s_mode: Literal["flat", "hist", "radial"] = "radial",
    i2s_colorspace: Img2SoundColorMode = "luma",
    i2s_pad_mode: Img2SoundPadMode = "same-center",
    i2s_impulse_len: int | Literal["auto", "frame"] = "auto",
    i2s_radius_mode: RadiusMode = "linear",
    i2s_phase_mode: PhaseMode = "zero",
    i2s_smoothing: SmoothingMode = "hann",
    i2s_impulse_norm: ImpulseNorm = "energy",
    i2s_out_norm: OutNorm = "match_rms",
    i2s_n_bins: int = 256,
    # serial/parallel behavior
    serial_mode: DualSerialMode = "parallel",
    audio_length_mode: AudioLengthMode = "pad-zero",
    block_size: int = 1,
    block_strategy: BlockStrategy = "fixed",
    block_min_frames: int = 1,
    block_max_frames: int | None = None,
    block_beats_per: int = 1,
    block_hop_length: int = 512,
    block_n_fft: int = 2048,
    block_peak_threshold: float = 0.3,
    block_peak_distance_s: float = 0.2,
    block_structure_kernel: int = 16,
    block_structure_max_frames: int = 600,
    block_bpm_min: float = 60.0,
    block_bpm_max: float = 200.0,
    # s2i color controls
    s2i_safe_color: bool = True,
    s2i_chroma_strength: float = 0.5,
    s2i_chroma_clip: float = 0.25,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Convolve video frames and audio with each other per frame.

    Modes
    -----
    - parallel: image_out = sound->image(original), audio_out = image->sound(original)
    - serial-image-first: image_out = sound->image(original), audio_out uses image_out
    - serial-sound-first: audio_out = image->sound(original), image_out uses audio_out
    """
    frames = list(frames)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]

    if fps <= 0:
        raise ValueError("fps must be > 0")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if block_strategy == "fixed" and block_size <= 0:
        raise ValueError("block_size must be positive")

    # If requested, resolve impulse length to the duration of a single frame.
    # This keeps memory stable even when block_size grows, and better matches an
    # "instantaneous per-frame" interpretation.
    if i2s_impulse_len == "frame":
        i2s_impulse_len = max(1, int(round(sr / fps)))

    def _to_uint8_frame(img_proc: np.ndarray) -> np.ndarray:
        arr = np.asarray(img_proc)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    # match audio length to video duration
    ideal_len = int(round(len(frames) * sr / fps))
    audio = _match_audio_length(audio, ideal_len, mode=audio_length_mode)
    n_samples, _ = audio.shape

    processed_frames: List[np.ndarray] = []
    audio_chunks: List[np.ndarray] = []

    n_frames = len(frames)
    if block_strategy != "fixed":
        mir_bar = tqdm(total=1, desc="mir analysis", unit="step")
    else:
        mir_bar = None

    block_segments = _resolve_block_segments(
        audio=audio,
        sr=sr,
        fps=fps,
        n_frames=n_frames,
        block_size=block_size,
        block_size_div=None,
        block_strategy=block_strategy,
        block_min_frames=block_min_frames,
        block_max_frames=block_max_frames,
        block_beats_per=block_beats_per,
        block_hop_length=block_hop_length,
        block_n_fft=block_n_fft,
        block_peak_threshold=block_peak_threshold,
        block_peak_distance_s=block_peak_distance_s,
        block_structure_kernel=block_structure_kernel,
        block_structure_max_frames=block_structure_max_frames,
        block_bpm_min=block_bpm_min,
        block_bpm_max=block_bpm_max,
    )
    if mir_bar is not None:
        mir_bar.update(1)
        mir_bar.close()

    block_iter = tqdm(
        block_segments,
        total=len(block_segments),
        desc="bi-conv video blocks",
        unit="block",
    )

    for block_start, block_end in block_iter:
        block_frames = [np.asarray(f) for f in frames[block_start : block_end + 1]]

        start, _ = _slice_for_frame(block_start, fps, sr, n_samples)
        _, stop = _slice_for_frame(block_end, fps, sr, n_samples)
        if stop <= start:
            expected = max(1, int(round((block_end - block_start + 1) * sr / fps)))
            chunk = np.zeros((expected, audio.shape[1]), dtype=np.float32)
        else:
            chunk = audio[start:stop, :]

        # mean image for image->sound branch; stream to avoid large stacks and
        # ensure float32 0..1 so color handling (e.g., YCbCr mid/side) stays valid.
        img_sum = None
        for frame_arr in block_frames:
            arr_f = np.asarray(frame_arr, dtype=np.float32)
            if arr_f.max() > 1.0:
                arr_f = arr_f / 255.0
            if img_sum is None:
                img_sum = np.zeros_like(arr_f, dtype=np.float32)
            img_sum += arr_f
        img_mean = img_sum / len(block_frames) if img_sum is not None else np.zeros(
            (1,), dtype=np.float32
        )

        # process according to serial mode, but at block granularity
        block_proc_frames_u8: List[np.ndarray] = []

        if serial_mode == "parallel":
            # sound->image uses original frames + shared chunk
            for frame_arr in block_frames:
                img_proc = spectral_sculpt(
                    image=frame_arr,
                    audio=chunk,
                    sr=sr,
                    mode=s2i_mode,
                    colorspace=s2i_colorspace,
                    safe_color=s2i_safe_color,
                    chroma_strength=s2i_chroma_strength,
                    chroma_clip=s2i_chroma_clip,
                    normalize=True,
                )
                block_proc_frames_u8.append(_to_uint8_frame(img_proc))

            aud_proc = _image2sound_apply(
                img=img_mean,
                audio=chunk,
                sr=sr,
                mode=i2s_mode,
                pad_mode=i2s_pad_mode,
                colorspace=i2s_colorspace,
                impulse_norm=i2s_impulse_norm,
                out_norm=i2s_out_norm,
                impulse_len=i2s_impulse_len,
                n_bins=i2s_n_bins,
                radius_mode=i2s_radius_mode,
                phase_mode=i2s_phase_mode,
                smoothing=i2s_smoothing,
            )
        elif serial_mode == "serial-image-first":
            img_mean_proc_sum = None
            for frame_arr in block_frames:
                img_proc = spectral_sculpt(
                    image=frame_arr,
                    audio=chunk,
                    sr=sr,
                    mode=s2i_mode,
                    colorspace=s2i_colorspace,
                    safe_color=s2i_safe_color,
                    chroma_strength=s2i_chroma_strength,
                    chroma_clip=s2i_chroma_clip,
                    normalize=True,
                )
                img_proc_f = np.asarray(img_proc, dtype=np.float32)
                if img_proc_f.max() > 1.0:
                    img_proc_f = img_proc_f / 255.0
                if img_mean_proc_sum is None:
                    img_mean_proc_sum = np.zeros_like(img_proc_f, dtype=np.float32)
                img_mean_proc_sum += img_proc_f
                block_proc_frames_u8.append(_to_uint8_frame(img_proc))

            img_mean_proc = (
                img_mean_proc_sum / len(block_proc_frames_u8)
                if img_mean_proc_sum is not None and len(block_proc_frames_u8) > 0
                else img_mean
            )

            aud_proc = _image2sound_apply(
                img=img_mean_proc,
                audio=chunk,
                sr=sr,
                mode=i2s_mode,
                pad_mode=i2s_pad_mode,
                colorspace=i2s_colorspace,
                impulse_norm=i2s_impulse_norm,
                out_norm=i2s_out_norm,
                impulse_len=i2s_impulse_len,
                n_bins=i2s_n_bins,
                radius_mode=i2s_radius_mode,
                phase_mode=i2s_phase_mode,
                smoothing=i2s_smoothing,
            )
        elif serial_mode == "serial-sound-first":
            aud_proc = _image2sound_apply(
                img=img_mean,
                audio=chunk,
                sr=sr,
                mode=i2s_mode,
                pad_mode=i2s_pad_mode,
                colorspace=i2s_colorspace,
                impulse_norm=i2s_impulse_norm,
                out_norm=i2s_out_norm,
                impulse_len=i2s_impulse_len,
                n_bins=i2s_n_bins,
                radius_mode=i2s_radius_mode,
                phase_mode=i2s_phase_mode,
                smoothing=i2s_smoothing,
            )
            for frame_arr in block_frames:
                img_proc = spectral_sculpt(
                    image=frame_arr,
                    audio=aud_proc,
                    sr=sr,
                    mode=s2i_mode,
                    colorspace=s2i_colorspace,
                    safe_color=s2i_safe_color,
                    chroma_strength=s2i_chroma_strength,
                    chroma_clip=s2i_chroma_clip,
                    normalize=True,
                )
                block_proc_frames_u8.append(_to_uint8_frame(img_proc))
        else:
            raise ValueError(f"Unknown serial_mode: {serial_mode}")

        # push processed frames
        processed_frames.extend(block_proc_frames_u8)

        audio_chunks.append(aud_proc)

    audio_out = np.concatenate(audio_chunks, axis=0) if audio_chunks else np.zeros(
        (0, audio.shape[1]), dtype=np.float32
    )
    return processed_frames, audio_out.astype(np.float32)


def biconv_video_from_files(
    video_path: str | Path,
    audio_path: Optional[str | Path] = None,
    *,
    fps: float | None = None,
    fps_policy: FPSPolicy = "auto",
    s2i_mode: Mode = "mono",
    s2i_colorspace: ColorMode = "luma",
    i2s_mode: Literal["flat", "hist", "radial"] = "radial",
    i2s_colorspace: Img2SoundColorMode = "luma",
    i2s_pad_mode: Img2SoundPadMode = "same-center",
    i2s_impulse_len: int | Literal["auto", "frame"] = "auto",
    i2s_radius_mode: RadiusMode = "linear",
    i2s_phase_mode: PhaseMode = "zero",
    i2s_smoothing: SmoothingMode = "hann",
    i2s_impulse_norm: ImpulseNorm = "energy",
    i2s_out_norm: OutNorm = "match_rms",
    i2s_n_bins: int = 256,
    serial_mode: DualSerialMode = "parallel",
    audio_length_mode: AudioLengthMode = "pad-zero",
    block_size: int = 1,
    block_size_div: int | None = None,
    block_strategy: BlockStrategy = "fixed",
    block_min_frames: int = 1,
    block_max_frames: int | None = None,
    block_beats_per: int = 1,
    block_hop_length: int = 512,
    block_n_fft: int = 2048,
    block_peak_threshold: float = 0.3,
    block_peak_distance_s: float = 0.2,
    block_structure_kernel: int = 16,
    block_structure_max_frames: int = 600,
    block_bpm_min: float = 60.0,
    block_bpm_max: float = 200.0,
    s2i_safe_color: bool = True,
    s2i_chroma_strength: float = 0.5,
    s2i_chroma_clip: float = 0.25,
    out_video: str | Path | None = None,
    out_audio: str | Path | None = None,
    mux_output: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, float, int]:
    """
    File-based bi-directional video processor.

    audio_path can point to the original video audio, any external audio, or
    even another video file (audio will be extracted automatically). If None,
    attempts to extract audio from the primary video.
    """
    video_path = _as_path(video_path)
    audio_used: Path | None = _as_path(audio_path) if audio_path else None

    if audio_used is None:
        audio, sr = _extract_audio_from_video(video_path)
    else:
        if _looks_like_video(audio_used):
            audio, sr = _extract_audio_from_video(audio_used)
        else:
            audio, sr = read_audio(audio_used, dtype="float32", always_2d=True)
    frames, meta_fps = read_video_frames(video_path)
    meta = _read_video_meta(video_path)
    if meta_fps and not meta.get("fps"):
        meta["fps"] = float(meta_fps)

    fps_used = _resolve_fps_for_video(
        video_path,
        meta=meta,
        fps_override=fps,
        fps_policy=fps_policy,
        n_frames=len(frames),
    )

    # derive block size from divisor if requested
    if block_strategy == "fixed" and block_size_div is not None:
        if block_size_div <= 0:
            raise ValueError("block_size_div must be positive")
        block_size = max(1, int(math.ceil(len(frames) / float(block_size_div))))

    frames_out, audio_out = biconv_video_arrays(
        frames=frames,
        audio=audio,
        sr=sr,
        fps=fps_used,
        s2i_mode=s2i_mode,
        s2i_colorspace=s2i_colorspace,
        i2s_mode=i2s_mode,
        i2s_colorspace=i2s_colorspace,
        i2s_pad_mode=i2s_pad_mode,
        i2s_impulse_len=i2s_impulse_len,
        i2s_radius_mode=i2s_radius_mode,
        i2s_phase_mode=i2s_phase_mode,
        i2s_smoothing=i2s_smoothing,
        i2s_impulse_norm=i2s_impulse_norm,
        i2s_out_norm=i2s_out_norm,
        i2s_n_bins=i2s_n_bins,
        serial_mode=serial_mode,
        audio_length_mode=audio_length_mode,
        block_size=block_size,
        block_strategy=block_strategy,
        block_min_frames=block_min_frames,
        block_max_frames=block_max_frames,
        block_beats_per=block_beats_per,
        block_hop_length=block_hop_length,
        block_n_fft=block_n_fft,
        block_peak_threshold=block_peak_threshold,
        block_peak_distance_s=block_peak_distance_s,
        block_structure_kernel=block_structure_kernel,
        block_structure_max_frames=block_structure_max_frames,
        block_bpm_min=block_bpm_min,
        block_bpm_max=block_bpm_max,
        s2i_safe_color=s2i_safe_color,
        s2i_chroma_strength=s2i_chroma_strength,
        s2i_chroma_clip=s2i_chroma_clip,
    )

    temp_video: Path | None = None
    temp_audio: Path | None = None

    if out_video is not None:
        out_video = _as_path(out_video)
        if mux_output:
            temp_video = out_video.with_name(
                f"{out_video.stem}_tmp_noaudio{out_video.suffix}"
            )
            write_video_frames(temp_video, frames_out, fps=fps_used)
        else:
            write_video_frames(out_video, frames_out, fps=fps_used)
    if out_audio is not None or mux_output:
        if out_audio is not None:
            temp_audio = _as_path(out_audio)
        else:
            import os

            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="biconv_audio_")
            os.close(fd)
            temp_audio = Path(tmp_path)
        temp_audio.parent.mkdir(parents=True, exist_ok=True)
        write_audio(temp_audio, audio_out, sr)

    if mux_output and out_video is not None and temp_video is not None and temp_audio is not None:
        if not _ffmpeg_available():
            raise RuntimeError("mux_output requested but ffmpeg not available on PATH")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video),
            "-i",
            str(temp_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-movflags",
            "+faststart",
            str(out_video),
        ]
        subprocess.run(cmd, check=True)
        try:
            temp_video.unlink(missing_ok=True)
            if out_audio is None:
                temp_audio.unlink(missing_ok=True)
        except OSError:
            pass
    elif out_audio is None and temp_audio is not None:
        # if user didn't ask for audio and mux not used, remove temp
        try:
            temp_audio.unlink(missing_ok=True)
        except OSError:
            pass

    return frames_out, audio_out, fps_used, sr


def biconv_video_to_files_stream(
    video_path: str | Path,
    audio_path: Optional[str | Path] = None,
    *,
    fps: float | None = None,
    fps_policy: FPSPolicy = "auto",
    s2i_mode: Mode = "mono",
    s2i_colorspace: ColorMode = "luma",
    i2s_mode: Literal["flat", "hist", "radial"] = "radial",
    i2s_colorspace: Img2SoundColorMode = "luma",
    i2s_pad_mode: Img2SoundPadMode = "same-center",
    i2s_impulse_len: int | Literal["auto", "frame"] = "auto",
    i2s_radius_mode: RadiusMode = "linear",
    i2s_phase_mode: PhaseMode = "zero",
    i2s_smoothing: SmoothingMode = "hann",
    i2s_impulse_norm: ImpulseNorm = "energy",
    i2s_out_norm: OutNorm = "match_rms",
    i2s_n_bins: int = 256,
    serial_mode: DualSerialMode = "parallel",
    audio_length_mode: AudioLengthMode = "pad-zero",
    block_size: int = 1,
    block_size_div: int | None = None,
    block_strategy: BlockStrategy = "fixed",
    block_min_frames: int = 1,
    block_max_frames: int | None = None,
    block_beats_per: int = 1,
    block_hop_length: int = 512,
    block_n_fft: int = 2048,
    block_peak_threshold: float = 0.3,
    block_peak_distance_s: float = 0.2,
    block_structure_kernel: int = 16,
    block_structure_max_frames: int = 600,
    block_bpm_min: float = 60.0,
    block_bpm_max: float = 200.0,
    s2i_safe_color: bool = True,
    s2i_chroma_strength: float = 0.5,
    s2i_chroma_clip: float = 0.25,
    out_video: str | Path | None = None,
    out_audio: str | Path | None = None,
    mux_output: bool = False,
) -> Tuple[int, Tuple[int, int], float, int]:
    """
    Streaming, file-based bi-directional video processor.

    Unlike `biconv_video_from_files`, this does NOT materialize the full frame
    list or the full output video in memory. It reads frames in blocks and
    writes video/audio incrementally.

    Returns
    -------
    n_frames_written, audio_shape, fps_used, sr
        audio_shape is (n_samples_written, n_channels). If audio isn't written
        (no `out_audio` and `mux_output=False`), audio_shape will be (0, 0).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    video_path = _as_path(video_path)
    audio_used: Path | None = _as_path(audio_path) if audio_path else None

    # --- audio ---
    if audio_used is None:
        audio, sr = _extract_audio_from_video(video_path)
    else:
        if _looks_like_video(audio_used):
            audio, sr = _extract_audio_from_video(audio_used)
        else:
            audio, sr = read_audio(audio_used, dtype="float32", always_2d=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]

    # --- video metadata (fps/duration) ---
    meta = _read_video_meta(video_path)
    fps_used = _resolve_fps_for_video(
        video_path,
        meta=meta,
        fps_override=fps,
        fps_policy=fps_policy,
    )

    if fps_used <= 0:
        raise ValueError("fps must be > 0")
    if sr <= 0:
        raise ValueError("sr must be > 0")

    if i2s_impulse_len == "frame":
        i2s_impulse_len = max(1, int(round(sr / fps_used)))

    # Best-effort estimate (used for progress/center-zero; never used to trim audio)
    n_frames_est: int | None = _estimate_total_frames_from_meta(meta, fps_used)

    # --- derive block size from divisor if requested ---
    if block_strategy == "fixed" and block_size_div is not None:
        if block_size_div <= 0:
            raise ValueError("block_size_div must be positive")
        if n_frames_est is None:
            # fallback: count frames (requires a second decode pass)
            try:
                import imageio.v3 as iio  # local import

                n_frames_est = sum(1 for _ in iio.imiter(video_path))
            except Exception:
                n_frames_est = None
        if not n_frames_est or n_frames_est <= 0:
            raise RuntimeError(
                "Could not estimate total frame count for --block-size-div; "
                "use --block-size instead."
            )
        block_size = max(1, int(math.ceil(n_frames_est / float(block_size_div))))

    if block_strategy != "fixed":
        mir_bar = tqdm(total=1, desc="mir analysis", unit="step")
    else:
        mir_bar = None

    block_segments = _resolve_block_segments(
        audio=audio,
        sr=sr,
        fps=fps_used,
        n_frames=n_frames_est,
        block_size=block_size,
        block_size_div=block_size_div,
        block_strategy=block_strategy,
        block_min_frames=block_min_frames,
        block_max_frames=block_max_frames,
        block_beats_per=block_beats_per,
        block_hop_length=block_hop_length,
        block_n_fft=block_n_fft,
        block_peak_threshold=block_peak_threshold,
        block_peak_distance_s=block_peak_distance_s,
        block_structure_kernel=block_structure_kernel,
        block_structure_max_frames=block_structure_max_frames,
        block_bpm_min=block_bpm_min,
        block_bpm_max=block_bpm_max,
    )
    if mir_bar is not None:
        mir_bar.update(1)
        mir_bar.close()

    block_sizes: list[int] | None = None
    if block_segments:
        block_sizes = [end - start + 1 for start, end in block_segments]

    # --- writers / temp paths ---
    out_video_p: Path | None = _as_path(out_video) if out_video is not None else None
    out_audio_p: Path | None = _as_path(out_audio) if out_audio is not None else None

    if mux_output and out_video_p is None:
        raise ValueError("mux_output=True requires out_video to be set.")

    if out_video_p is None and out_audio_p is None and not mux_output:
        raise ValueError("Nothing to write: pass out_video/out_audio or enable mux_output.")

    temp_video: Path | None = None
    temp_audio: Path | None = None

    if out_video_p is not None:
        out_video_p.parent.mkdir(parents=True, exist_ok=True)
        if mux_output:
            temp_video = out_video_p.with_name(f"{out_video_p.stem}_tmp_noaudio{out_video_p.suffix}")
        else:
            temp_video = out_video_p

    if out_audio_p is not None or mux_output:
        if out_audio_p is not None:
            temp_audio = out_audio_p
        else:
            import os

            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="biconv_audio_")
            os.close(fd)
            temp_audio = Path(tmp_path)
        temp_audio.parent.mkdir(parents=True, exist_ok=True)

    # RNG for pad-noise mode
    rng = np.random.default_rng()
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size > 0 else 1.0
    noise_scale = (0.01 * rms) if rms > 0 else 0.01

    # center-zero requires a target duration (best-effort from metadata)
    center_target_len: int | None = None
    if audio_length_mode == "center-zero":
        dur = meta.get("duration", None)
        try:
            dur_f = float(dur)
        except (TypeError, ValueError):
            dur_f = 0.0
        if dur_f > 0:
            center_target_len = int(max(1, round(dur_f * sr)))
        elif n_frames_est is not None and n_frames_est > 0:
            center_target_len = int(max(1, round(n_frames_est * sr / fps_used)))
        else:
            # Last resort: count frames to get a target duration.
            try:
                import imageio.v3 as iio  # local import

                n_frames_est = sum(1 for _ in iio.imiter(video_path))
            except Exception:
                n_frames_est = None

            if n_frames_est is not None and n_frames_est > 0:
                center_target_len = int(max(1, round(n_frames_est * sr / fps_used)))
            else:
                center_target_len = int(audio.shape[0])

    # Progress bar: estimate blocks if we have an estimate
    n_blocks_est: int | None = None
    if block_sizes:
        n_blocks_est = len(block_sizes)
    elif n_frames_est is not None and n_frames_est > 0:
        n_blocks_est = int((n_frames_est + block_size - 1) // block_size)

    def _to_uint8_frame(img_proc: np.ndarray) -> np.ndarray:
        arr = np.asarray(img_proc)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    n_frames_written = 0
    audio_samples_written = 0
    audio_channels_written = 0

    # Open video writer if requested
    video_writer = None
    if temp_video is not None:
        import imageio

        video_writer = imageio.get_writer(str(temp_video), fps=fps_used, macro_block_size=None)

    # Open audio writer lazily once we know output channels
    audio_writer: sf.SoundFile | None = None

    try:
        import imageio.v3 as iio

        frame_iter = iio.imiter(video_path)
        block_iter = tqdm(
            total=n_blocks_est,
            desc="bi-conv video blocks",
            unit="block",
        )

        block_start = 0
        block_sizes_iter = iter(block_sizes) if block_sizes else None
        while True:
            if block_sizes_iter is not None:
                try:
                    block_target = int(next(block_sizes_iter))
                except StopIteration:
                    block_sizes_iter = None
                    block_target = int(block_size)
            else:
                block_target = int(block_size)
            if block_target <= 0:
                block_target = 1

            # read frames for this block
            block_frames: list[np.ndarray] = []
            for _ in range(block_target):
                try:
                    f = next(frame_iter)
                except StopIteration:
                    break
                block_frames.append(np.asarray(f))

            if not block_frames:
                break

            block_end = block_start + len(block_frames) - 1

            # map frame interval -> audio interval (unclamped)
            start_s = int(round(block_start * sr / fps_used))
            stop_s = int(round((block_end + 1) * sr / fps_used))
            if stop_s <= start_s:
                stop_s = start_s + max(1, int(round(len(block_frames) * sr / fps_used)))

            chunk = _audio_chunk_for_interval(
                audio,
                start_s,
                stop_s,
                mode=audio_length_mode,
                rng=rng,
                noise_scale=noise_scale,
                center_target_len=center_target_len,
            )

            # mean image for image->sound branch (float32 0..1)
            img_sum = None
            for frame_arr in block_frames:
                arr_f = np.asarray(frame_arr, dtype=np.float32)
                if arr_f.max() > 1.0:
                    arr_f = arr_f / 255.0
                if img_sum is None:
                    img_sum = np.zeros_like(arr_f, dtype=np.float32)
                img_sum += arr_f
            img_mean = img_sum / len(block_frames) if img_sum is not None else np.zeros((1,), dtype=np.float32)

            if serial_mode == "parallel":
                for frame_arr in block_frames:
                    img_proc = spectral_sculpt(
                        image=frame_arr,
                        audio=chunk,
                        sr=sr,
                        mode=s2i_mode,
                        colorspace=s2i_colorspace,
                        safe_color=s2i_safe_color,
                        chroma_strength=s2i_chroma_strength,
                        chroma_clip=s2i_chroma_clip,
                        normalize=True,
                    )
                    frame_u8 = _to_uint8_frame(img_proc)
                    if video_writer is not None:
                        video_writer.append_data(frame_u8)
                    n_frames_written += 1

                aud_proc = _image2sound_apply(
                    img=img_mean,
                    audio=chunk,
                    sr=sr,
                    mode=i2s_mode,
                    pad_mode=i2s_pad_mode,
                    colorspace=i2s_colorspace,
                    impulse_norm=i2s_impulse_norm,
                    out_norm=i2s_out_norm,
                    impulse_len=i2s_impulse_len,
                    n_bins=i2s_n_bins,
                    radius_mode=i2s_radius_mode,
                    phase_mode=i2s_phase_mode,
                    smoothing=i2s_smoothing,
                )
            elif serial_mode == "serial-image-first":
                img_mean_proc_sum = None
                for frame_arr in block_frames:
                    img_proc = spectral_sculpt(
                        image=frame_arr,
                        audio=chunk,
                        sr=sr,
                        mode=s2i_mode,
                        colorspace=s2i_colorspace,
                        safe_color=s2i_safe_color,
                        chroma_strength=s2i_chroma_strength,
                        chroma_clip=s2i_chroma_clip,
                        normalize=True,
                    )
                    img_proc_f = np.asarray(img_proc, dtype=np.float32)
                    if img_proc_f.max() > 1.0:
                        img_proc_f = img_proc_f / 255.0
                    if img_mean_proc_sum is None:
                        img_mean_proc_sum = np.zeros_like(img_proc_f, dtype=np.float32)
                    img_mean_proc_sum += img_proc_f

                    frame_u8 = _to_uint8_frame(img_proc)
                    if video_writer is not None:
                        video_writer.append_data(frame_u8)
                    n_frames_written += 1

                img_mean_proc = (
                    img_mean_proc_sum / len(block_frames)
                    if img_mean_proc_sum is not None and len(block_frames) > 0
                    else img_mean
                )

                aud_proc = _image2sound_apply(
                    img=img_mean_proc,
                    audio=chunk,
                    sr=sr,
                    mode=i2s_mode,
                    pad_mode=i2s_pad_mode,
                    colorspace=i2s_colorspace,
                    impulse_norm=i2s_impulse_norm,
                    out_norm=i2s_out_norm,
                    impulse_len=i2s_impulse_len,
                    n_bins=i2s_n_bins,
                    radius_mode=i2s_radius_mode,
                    phase_mode=i2s_phase_mode,
                    smoothing=i2s_smoothing,
                )
            elif serial_mode == "serial-sound-first":
                aud_proc = _image2sound_apply(
                    img=img_mean,
                    audio=chunk,
                    sr=sr,
                    mode=i2s_mode,
                    pad_mode=i2s_pad_mode,
                    colorspace=i2s_colorspace,
                    impulse_norm=i2s_impulse_norm,
                    out_norm=i2s_out_norm,
                    impulse_len=i2s_impulse_len,
                    n_bins=i2s_n_bins,
                    radius_mode=i2s_radius_mode,
                    phase_mode=i2s_phase_mode,
                    smoothing=i2s_smoothing,
                )
                for frame_arr in block_frames:
                    img_proc = spectral_sculpt(
                        image=frame_arr,
                        audio=aud_proc,
                        sr=sr,
                        mode=s2i_mode,
                        colorspace=s2i_colorspace,
                        safe_color=s2i_safe_color,
                        chroma_strength=s2i_chroma_strength,
                        chroma_clip=s2i_chroma_clip,
                        normalize=True,
                    )
                    frame_u8 = _to_uint8_frame(img_proc)
                    if video_writer is not None:
                        video_writer.append_data(frame_u8)
                    n_frames_written += 1
            else:
                raise ValueError(f"Unknown serial_mode: {serial_mode}")

            # write audio chunk if requested
            if temp_audio is not None:
                if aud_proc.ndim == 1:
                    aud_proc = aud_proc[:, None]
                if audio_writer is None:
                    audio_channels_written = int(aud_proc.shape[1])
                    audio_writer = sf.SoundFile(
                        str(temp_audio),
                        mode="w",
                        samplerate=int(sr),
                        channels=audio_channels_written,
                        subtype="PCM_16",
                    )
                to_write = np.asarray(aud_proc, dtype=np.float32)
                to_write = np.clip(to_write, -1.0, 1.0)
                audio_writer.write(to_write)
                audio_samples_written += int(aud_proc.shape[0])

            block_start = block_end + 1
            block_iter.update(1)

        block_iter.close()
    finally:
        if video_writer is not None:
            video_writer.close()
        if audio_writer is not None:
            audio_writer.close()

    # mux if requested
    if mux_output and out_video_p is not None and temp_video is not None and temp_audio is not None:
        if not _ffmpeg_available():
            raise RuntimeError("mux_output requested but ffmpeg not available on PATH")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video),
            "-i",
            str(temp_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-movflags",
            "+faststart",
            str(out_video_p),
        ]
        subprocess.run(cmd, check=True)
        try:
            temp_video.unlink(missing_ok=True)
            if out_audio_p is None:
                temp_audio.unlink(missing_ok=True)
        except OSError:
            pass

    audio_shape = (
        (audio_samples_written, audio_channels_written)
        if temp_audio is not None
        else (0, 0)
    )
    return n_frames_written, audio_shape, fps_used, sr
