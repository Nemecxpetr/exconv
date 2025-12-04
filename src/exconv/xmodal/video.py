# src/exconv/xmodal/video.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Literal, overload, Optional
import subprocess
import tempfile
from io import BytesIO

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

__all__ = [
    "AudioVideoMode",
    "sound2image_video_arrays",
    "sound2image_video_from_files",
    "biconv_video_arrays",
    "biconv_video_from_files",
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
        Override FPS if metadata is missing/incorrect. If None, the FPS from
        `read_video_frames` metadata is used.
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

    fps_used: float
    if fps is not None:
        fps_used = float(fps)
    else:
        if not meta_fps or meta_fps <= 0:
            raise RuntimeError(
                "Could not determine FPS from video metadata; "
                "pass fps explicitly."
            )
        fps_used = float(meta_fps)

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
    i2s_impulse_len: int | Literal["auto"] = "auto",
    i2s_radius_mode: RadiusMode = "linear",
    i2s_phase_mode: PhaseMode = "zero",
    i2s_smoothing: SmoothingMode = "hann",
    i2s_impulse_norm: ImpulseNorm = "energy",
    i2s_out_norm: OutNorm = "match_rms",
    i2s_n_bins: int = 256,
    # serial/parallel behavior
    serial_mode: DualSerialMode = "parallel",
    audio_length_mode: AudioLengthMode = "pad-zero",
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

    # match audio length to video duration
    ideal_len = int(round(len(frames) * sr / fps))
    audio = _match_audio_length(audio, ideal_len, mode=audio_length_mode)
    n_samples, _ = audio.shape

    processed_frames: List[np.ndarray] = []
    audio_chunks: List[np.ndarray] = []

    frame_iter = tqdm(frames, total=len(frames), desc="bi-conv video", unit="frame")
    for idx, frame in enumerate(frame_iter):

        frame_arr = np.asarray(frame)
        start, stop = _slice_for_frame(idx, fps, sr, n_samples)
        if stop <= start:
            chunk = np.zeros((max(1, int(round(sr / fps))), audio.shape[1]), dtype=np.float32)
        else:
            chunk = audio[start:stop, :]

        if serial_mode == "parallel":
            img_proc = spectral_sculpt(
                image=frame_arr,
                audio=chunk,
                sr=sr,
                mode=s2i_mode,
                colorspace=s2i_colorspace,
                normalize=True,
            )
            aud_proc = _image2sound_apply(
                img=frame_arr,
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
            img_proc = spectral_sculpt(
                image=frame_arr,
                audio=chunk,
                sr=sr,
                mode=s2i_mode,
                colorspace=s2i_colorspace,
                normalize=True,
            )
            aud_proc = _image2sound_apply(
                img=img_proc,
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
                img=frame_arr,
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
            img_proc = spectral_sculpt(
                image=frame_arr,
                audio=aud_proc,
                sr=sr,
                mode=s2i_mode,
                colorspace=s2i_colorspace,
                normalize=True,
            )
        else:
            raise ValueError(f"Unknown serial_mode: {serial_mode}")

        if np.issubdtype(img_proc.dtype, np.floating):
            frame_u8 = np.clip(img_proc * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            frame_u8 = np.clip(img_proc, 0, 255).astype(np.uint8)

        processed_frames.append(frame_u8)
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
    s2i_mode: Mode = "mono",
    s2i_colorspace: ColorMode = "luma",
    i2s_mode: Literal["flat", "hist", "radial"] = "radial",
    i2s_colorspace: Img2SoundColorMode = "luma",
    i2s_pad_mode: Img2SoundPadMode = "same-center",
    i2s_impulse_len: int | Literal["auto"] = "auto",
    i2s_radius_mode: RadiusMode = "linear",
    i2s_phase_mode: PhaseMode = "zero",
    i2s_smoothing: SmoothingMode = "hann",
    i2s_impulse_norm: ImpulseNorm = "energy",
    i2s_out_norm: OutNorm = "match_rms",
    i2s_n_bins: int = 256,
    serial_mode: DualSerialMode = "parallel",
    audio_length_mode: AudioLengthMode = "pad-zero",
    out_video: str | Path | None = None,
    out_audio: str | Path | None = None,
    mux_output: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, float, int]:
    """
    File-based bi-directional video processor.

    audio_path can point to the original video audio or any external source.
    If None, attempts to extract audio from the video.
    """
    video_path = _as_path(video_path)
    audio_used: Path | None = _as_path(audio_path) if audio_path else None

    if audio_used is None:
        audio, sr = _extract_audio_from_video(video_path)
    else:
        audio, sr = read_audio(audio_used, dtype="float32", always_2d=True)
    frames, meta_fps = read_video_frames(video_path)

    if fps is None:
        if not meta_fps or meta_fps <= 0:
            raise RuntimeError("Could not determine FPS; pass fps explicitly.")
        fps_used = float(meta_fps)
    else:
        fps_used = float(fps)

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
            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="biconv_audio_")
            Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp_path).write_bytes(b"")  # ensure file exists
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
            "-c",
            "copy",
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
