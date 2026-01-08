# src/exconv/xmodal/video.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Literal, overload, Optional
import subprocess
import tempfile
from io import BytesIO
import math

import numpy as np
import soundfile as sf
from tqdm import tqdm

from exconv.dsp.crossfade import CrossfadeMode, crossfade_weights
from exconv.dsp.envelopes import EnvelopeCurve, apply_adsr
from exconv.dsp.segments import (
    AudioLengthMode as DspAudioLengthMode,
    slice_for_frame,
    frame_range_samples,
    match_audio_length,
    audio_chunk_for_interval,
)
from exconv.video_meta import ffprobe_fps_info, parse_float
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
BlockEnvelopeCurve = EnvelopeCurve
BlockCrossover = CrossfadeMode
AudioLengthMode = DspAudioLengthMode

__all__ = [
    "AudioVideoMode",
    "BlockStrategy",
    "BlockEnvelopeCurve",
    "BlockCrossover",
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


_FPS_GUARD_RATIO = 1.2

FPSPolicy = Literal["auto", "metadata", "avg_frame_rate", "r_frame_rate"]


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

    meta_fps = parse_float(meta.get("fps"))
    meta_duration = parse_float(meta.get("duration"))

    ff_avg: Optional[float] = None
    ff_r: Optional[float] = None
    ff_duration: Optional[float] = None
    if fps_policy != "metadata":
        ff_avg, ff_r, ff_duration = ffprobe_fps_info(video_path)

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


def _block_overlap_sizes(
    segments: Sequence[tuple[int, int]],
    *,
    crossover: BlockCrossover,
    crossover_frames: int,
) -> list[int]:
    if crossover == "none" or crossover_frames <= 0 or len(segments) < 2:
        return [0 for _ in range(max(0, len(segments) - 1))]
    overlaps: list[int] = []
    lengths = [end - start + 1 for start, end in segments]
    for idx in range(len(segments) - 1):
        overlaps.append(int(min(crossover_frames, lengths[idx], lengths[idx + 1])))
    return overlaps


def _blend_frames(
    prev_frames: list[np.ndarray],
    cur_frames: list[np.ndarray],
    *,
    mode: BlockCrossover,
) -> list[np.ndarray]:
    if not prev_frames or not cur_frames:
        return []
    n = min(len(prev_frames), len(cur_frames))
    w_prev, w_cur = crossfade_weights(n, mode)
    prev_arr = np.stack(prev_frames[:n], axis=0).astype(np.float32)
    cur_arr = np.stack(cur_frames[:n], axis=0).astype(np.float32)
    w_shape = (n,) + (1,) * (prev_arr.ndim - 1)
    blended = prev_arr * w_prev.reshape(w_shape) + cur_arr * w_cur.reshape(w_shape)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return [frame for frame in blended]


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
        start, stop = slice_for_frame(idx, fps, sr, n_samples)

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
    block_crossover: BlockCrossover = "none",
    block_crossover_frames: int = 0,
    block_adsr_attack_s: float = 0.0,
    block_adsr_decay_s: float = 0.0,
    block_adsr_sustain: float = 1.0,
    block_adsr_release_s: float = 0.0,
    block_adsr_curve: BlockEnvelopeCurve = "linear",
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
    audio = match_audio_length(audio, ideal_len, mode=audio_length_mode)
    n_samples, _ = audio.shape

    if block_crossover_frames < 0:
        raise ValueError("block_crossover_frames must be >= 0")

    frames_out: List[np.ndarray] = []
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

    block_overlaps = _block_overlap_sizes(
        block_segments,
        crossover=block_crossover,
        crossover_frames=int(block_crossover_frames),
    )

    block_iter = tqdm(
        block_segments,
        total=len(block_segments),
        desc="bi-conv video blocks",
        unit="block",
    )

    prev_tail_frames: list[np.ndarray] = []
    prev_tail_audio: np.ndarray | None = None

    for block_index, (block_start, block_end) in enumerate(block_iter):
        prev_overlap = block_overlaps[block_index - 1] if block_index > 0 else 0
        next_overlap = (
            block_overlaps[block_index]
            if block_index < len(block_overlaps)
            else 0
        )

        ext_start = max(0, block_start - prev_overlap)
        ext_end = min(n_frames - 1, block_end + next_overlap)
        block_frames = [np.asarray(f) for f in frames[ext_start : ext_end + 1]]

        ext_start_s, ext_stop_s = frame_range_samples(
            ext_start,
            ext_end,
            sr=sr,
            fps=fps,
            n_samples=n_samples,
        )
        if ext_stop_s <= ext_start_s:
            expected = max(1, int(round((ext_end - ext_start + 1) * sr / fps)))
            chunk = np.zeros((expected, audio.shape[1]), dtype=np.float32)
        else:
            chunk = audio[ext_start_s:ext_stop_s, :]

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

        aud_proc_out = apply_adsr(
            aud_proc,
            sr,
            attack_s=block_adsr_attack_s,
            decay_s=block_adsr_decay_s,
            sustain=block_adsr_sustain,
            release_s=block_adsr_release_s,
            curve=block_adsr_curve,
        )
        if aud_proc_out.ndim == 1:
            aud_proc_out = aud_proc_out[:, None]

        local_start = block_start - ext_start
        local_end = block_end - ext_start
        base_frames = block_proc_frames_u8[local_start : local_end + 1]
        base_len = len(base_frames)

        overlap_head = 0
        if prev_overlap > 0 and prev_tail_frames:
            overlap_head = min(prev_overlap, len(prev_tail_frames), base_len)
        overlap_tail = 0
        if next_overlap > 0 and base_len > overlap_head:
            overlap_tail = min(next_overlap, base_len - overlap_head)

        if overlap_head > 0:
            cur_head = base_frames[:overlap_head]
            frames_out.extend(_blend_frames(prev_tail_frames, cur_head, mode=block_crossover))
        elif prev_overlap == 0 and prev_tail_frames:
            frames_out.extend(prev_tail_frames)
            prev_tail_frames = []

        mid_start = overlap_head
        mid_end = base_len - overlap_tail
        if mid_end > mid_start:
            frames_out.extend(base_frames[mid_start:mid_end])

        prev_tail_frames = base_frames[mid_end:] if overlap_tail > 0 else []

        def _audio_slice(start_frame: int, end_frame: int) -> np.ndarray:
            if end_frame < start_frame:
                return np.zeros((0, aud_proc_out.shape[1]), dtype=np.float32)
            seg_start_s, seg_stop_s = frame_range_samples(
                start_frame,
                end_frame,
                sr=sr,
                fps=fps,
                n_samples=n_samples,
            )
            local_seg_start = max(0, seg_start_s - ext_start_s)
            local_seg_stop = max(local_seg_start, seg_stop_s - ext_start_s)
            local_seg_stop = min(local_seg_stop, aud_proc_out.shape[0])
            return aud_proc_out[local_seg_start:local_seg_stop, :]

        if overlap_head > 0:
            cur_head_audio = _audio_slice(block_start, block_start + overlap_head - 1)
            if prev_tail_audio is not None:
                blend_len = min(prev_tail_audio.shape[0], cur_head_audio.shape[0])
                w_prev, w_cur = crossfade_weights(blend_len, block_crossover)
                w_prev = w_prev[:, None]
                w_cur = w_cur[:, None]
                blended = prev_tail_audio[:blend_len, :] * w_prev + cur_head_audio[:blend_len, :] * w_cur
                audio_chunks.append(blended.astype(np.float32))
            elif cur_head_audio.size > 0:
                audio_chunks.append(cur_head_audio)
        elif overlap_head == 0 and prev_tail_audio is not None:
            audio_chunks.append(prev_tail_audio)

        mid_start_frame = block_start + overlap_head
        mid_end_frame = block_end - overlap_tail
        if mid_end_frame >= mid_start_frame:
            mid_audio = _audio_slice(mid_start_frame, mid_end_frame)
            if mid_audio.size > 0:
                audio_chunks.append(mid_audio)

        if overlap_tail > 0:
            tail_start = block_end - overlap_tail + 1
            prev_tail_audio = _audio_slice(tail_start, block_end)
        else:
            prev_tail_audio = None

    if prev_tail_frames:
        frames_out.extend(prev_tail_frames)
    if prev_tail_audio is not None:
        audio_chunks.append(prev_tail_audio)

    audio_out = np.concatenate(audio_chunks, axis=0) if audio_chunks else np.zeros(
        (0, audio.shape[1]), dtype=np.float32
    )
    return frames_out, audio_out.astype(np.float32)


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
    block_crossover: BlockCrossover = "none",
    block_crossover_frames: int = 0,
    block_adsr_attack_s: float = 0.0,
    block_adsr_decay_s: float = 0.0,
    block_adsr_sustain: float = 1.0,
    block_adsr_release_s: float = 0.0,
    block_adsr_curve: BlockEnvelopeCurve = "linear",
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
        block_crossover=block_crossover,
        block_crossover_frames=block_crossover_frames,
        block_adsr_attack_s=block_adsr_attack_s,
        block_adsr_decay_s=block_adsr_decay_s,
        block_adsr_sustain=block_adsr_sustain,
        block_adsr_release_s=block_adsr_release_s,
        block_adsr_curve=block_adsr_curve,
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
    block_crossover: BlockCrossover = "none",
    block_crossover_frames: int = 0,
    block_adsr_attack_s: float = 0.0,
    block_adsr_decay_s: float = 0.0,
    block_adsr_sustain: float = 1.0,
    block_adsr_release_s: float = 0.0,
    block_adsr_curve: BlockEnvelopeCurve = "linear",
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
    if block_crossover_frames < 0:
        raise ValueError("block_crossover_frames must be >= 0")

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

    block_overlaps = _block_overlap_sizes(
        block_segments,
        crossover=block_crossover,
        crossover_frames=int(block_crossover_frames),
    )

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

        def _write_frames(frames_list: list[np.ndarray]) -> None:
            nonlocal n_frames_written
            if not frames_list:
                return
            for frame in frames_list:
                if video_writer is not None:
                    video_writer.append_data(frame)
                n_frames_written += 1

        def _write_audio(seg: np.ndarray) -> None:
            nonlocal audio_writer, audio_channels_written, audio_samples_written
            if temp_audio is None or seg.size == 0:
                return
            if seg.ndim == 1:
                seg = seg[:, None]
            if audio_writer is None:
                audio_channels_written = int(seg.shape[1])
                audio_writer = sf.SoundFile(
                    str(temp_audio),
                    mode="w",
                    samplerate=int(sr),
                    channels=audio_channels_written,
                    subtype="PCM_16",
                )
            to_write = np.asarray(seg, dtype=np.float32)
            to_write = np.clip(to_write, -1.0, 1.0)
            audio_writer.write(to_write)
            audio_samples_written += int(to_write.shape[0])

        block_start = 0
        block_sizes_iter = iter(block_sizes) if block_sizes else None
        carry_frames: list[np.ndarray] = []
        prev_tail_frames: list[np.ndarray] = []
        prev_tail_audio: np.ndarray | None = None
        block_index = 0

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

            next_overlap = 0
            if block_overlaps:
                if block_index < len(block_overlaps):
                    next_overlap = int(block_overlaps[block_index])
            elif block_crossover != "none" and block_crossover_frames > 0:
                next_overlap = min(int(block_crossover_frames), block_target)

            if len(carry_frames) > block_target:
                carry_frames = carry_frames[-block_target:]
            prev_overlap = len(carry_frames)
            base_needed = max(0, block_target - prev_overlap)
            read_target = base_needed + next_overlap

            fresh_frames: list[np.ndarray] = []
            for _ in range(read_target):
                try:
                    f = next(frame_iter)
                except StopIteration:
                    break
                fresh_frames.append(np.asarray(f))

            if not fresh_frames and not carry_frames:
                break

            if len(fresh_frames) < base_needed:
                base_needed = len(fresh_frames)
                block_target = prev_overlap + base_needed
                next_overlap = 0

            current_frames = fresh_frames[:base_needed]
            next_carry = fresh_frames[base_needed : base_needed + next_overlap]

            if block_target <= 0:
                break
            if not current_frames and not carry_frames:
                break

            block_frames = [*carry_frames, *current_frames, *next_carry]
            block_end = block_start + block_target - 1

            ext_start = max(0, block_start - prev_overlap)
            ext_end = block_end + len(next_carry)
            ext_start_s, ext_stop_s = frame_range_samples(
                ext_start,
                ext_end,
                sr=sr,
                fps=fps_used,
                n_samples=None,
            )
            if ext_stop_s <= ext_start_s:
                expected = max(
                    1, int(round((ext_end - ext_start + 1) * sr / fps_used))
                )
                chunk = np.zeros((expected, audio.shape[1]), dtype=np.float32)
            else:
                chunk = audio_chunk_for_interval(
                    audio,
                    ext_start_s,
                    ext_stop_s,
                    mode=audio_length_mode,
                    rng=rng,
                    noise_scale=noise_scale,
                    center_target_len=center_target_len,
                )

            img_sum = None
            for frame_arr in block_frames:
                arr_f = np.asarray(frame_arr, dtype=np.float32)
                if arr_f.max() > 1.0:
                    arr_f = arr_f / 255.0
                if img_sum is None:
                    img_sum = np.zeros_like(arr_f, dtype=np.float32)
                img_sum += arr_f
            img_mean = img_sum / len(block_frames) if img_sum is not None else np.zeros((1,), dtype=np.float32)

            block_proc_frames_u8: list[np.ndarray] = []
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

            aud_proc_out = apply_adsr(
                aud_proc,
                sr,
                attack_s=block_adsr_attack_s,
                decay_s=block_adsr_decay_s,
                sustain=block_adsr_sustain,
                release_s=block_adsr_release_s,
                curve=block_adsr_curve,
            )
            if aud_proc_out.ndim == 1:
                aud_proc_out = aud_proc_out[:, None]

            local_start = block_start - ext_start
            local_end = local_start + block_target - 1
            base_frames = block_proc_frames_u8[local_start : local_end + 1]
            base_len = len(base_frames)

            overlap_head = 0
            if prev_overlap > 0 and prev_tail_frames:
                overlap_head = min(prev_overlap, len(prev_tail_frames), base_len)
            overlap_tail = 0
            if len(next_carry) > 0 and base_len > overlap_head:
                overlap_tail = min(len(next_carry), base_len - overlap_head)

            if overlap_head > 0:
                cur_head = base_frames[:overlap_head]
                _write_frames(_blend_frames(prev_tail_frames, cur_head, mode=block_crossover))
            elif prev_overlap == 0 and prev_tail_frames:
                _write_frames(prev_tail_frames)
                prev_tail_frames = []

            mid_start = overlap_head
            mid_end = base_len - overlap_tail
            if mid_end > mid_start:
                _write_frames(base_frames[mid_start:mid_end])

            prev_tail_frames = base_frames[mid_end:] if overlap_tail > 0 else []

            def _audio_slice(start_frame: int, end_frame: int) -> np.ndarray:
                if end_frame < start_frame:
                    return np.zeros((0, aud_proc_out.shape[1]), dtype=np.float32)
                seg_start_s, seg_stop_s = frame_range_samples(
                    start_frame,
                    end_frame,
                    sr=sr,
                    fps=fps_used,
                    n_samples=None,
                )
                local_seg_start = max(0, seg_start_s - ext_start_s)
                local_seg_stop = max(local_seg_start, seg_stop_s - ext_start_s)
                local_seg_stop = min(local_seg_stop, aud_proc_out.shape[0])
                return aud_proc_out[local_seg_start:local_seg_stop, :]

            if overlap_head > 0:
                cur_head_audio = _audio_slice(block_start, block_start + overlap_head - 1)
                if prev_tail_audio is not None:
                    blend_len = min(prev_tail_audio.shape[0], cur_head_audio.shape[0])
                    w_prev, w_cur = crossfade_weights(blend_len, block_crossover)
                    w_prev = w_prev[:, None]
                    w_cur = w_cur[:, None]
                    blended = prev_tail_audio[:blend_len, :] * w_prev + cur_head_audio[:blend_len, :] * w_cur
                    _write_audio(blended.astype(np.float32))
                elif cur_head_audio.size > 0:
                    _write_audio(cur_head_audio)
            elif overlap_head == 0 and prev_tail_audio is not None:
                _write_audio(prev_tail_audio)

            mid_start_frame = block_start + overlap_head
            mid_end_frame = block_end - overlap_tail
            if mid_end_frame >= mid_start_frame:
                mid_audio = _audio_slice(mid_start_frame, mid_end_frame)
                _write_audio(mid_audio)

            if overlap_tail > 0:
                tail_start = block_end - overlap_tail + 1
                prev_tail_audio = _audio_slice(tail_start, block_end)
            else:
                prev_tail_audio = None

            carry_frames = next_carry
            block_start = block_end + 1
            block_index += 1
            block_iter.update(1)

        if prev_tail_frames:
            _write_frames(prev_tail_frames)
        if prev_tail_audio is not None:
            _write_audio(prev_tail_audio)

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
