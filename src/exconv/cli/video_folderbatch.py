from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from exconv.video_meta import (
    build_exconv_metadata,
    ffmpeg_metadata_args,
    ffprobe_available,
    ffprobe_fps_info,
)
from exconv.cli.settings import (
    add_settings_args,
    strip_settings_args,
    load_settings,
    select_settings,
    apply_settings_to_parser,
    serialize_args,
    save_settings,
)

VIDEO_EXTS = {
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

_FPS_GUARD_RATIO = 1.2


def _path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _iter_videos(root: Path, recursive: bool) -> Iterable[Path]:
    if not root.exists():
        return []
    candidates = root.rglob("*") if recursive else root.iterdir()
    return (
        p
        for p in sorted(candidates)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def _set_thread_env(n: int) -> None:
    val = str(int(n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = val


@dataclass(frozen=True)
class FpsDecision:
    fps_override: Optional[float]
    fps_policy: Optional[str]


def _resolve_fps_override(
    video_path: Path,
    *,
    default_fps: Optional[float],
    guard_mode: str,
    decision_cache: Optional[dict[tuple[float, float], FpsDecision]] = None,
) -> FpsDecision:
    if default_fps is not None or guard_mode == "off":
        return FpsDecision(fps_override=default_fps, fps_policy=None)

    avg, r, _duration = ffprobe_fps_info(video_path)
    if avg is None or r is None or avg <= 0 or r <= 0:
        return FpsDecision(fps_override=None, fps_policy=None)

    ratio = max(r / avg, avg / r)
    if ratio < _FPS_GUARD_RATIO:
        return FpsDecision(fps_override=None, fps_policy=None)

    key = (round(avg, 3), round(r, 3))
    if decision_cache is not None and key in decision_cache:
        return decision_cache[key]

    recommended = r
    print(
        "[fps-guard] This problem has been detected for "
        f"{video_path.name}: avg_frame_rate={avg:.3f} r_frame_rate={r:.3f}."
    )
    print(
        "[fps-guard] Using avg_frame_rate can slow playback; "
        f"recommended fps={recommended:.3f}."
    )

    if guard_mode == "auto" or not sys.stdin.isatty():
        print(f"[fps-guard] Using fps={recommended:.3f} for {video_path.name}.")
        decision = FpsDecision(fps_override=recommended, fps_policy=None)
        if decision_cache is not None:
            decision_cache[key] = decision
        return decision

    while True:
        resp = input("Use recommended fps for all matching files? [Y/n] ").strip().lower()
        if resp in ("", "y", "yes"):
            decision = FpsDecision(fps_override=recommended, fps_policy=None)
            if decision_cache is not None:
                decision_cache[key] = decision
            return decision
        if resp in ("n", "no"):
            break
        print("Please enter Y or n.")

    while True:
        val = input("Enter FPS to use (blank to keep metadata): ").strip()
        if val == "":
            decision = FpsDecision(fps_override=None, fps_policy="metadata")
            if decision_cache is not None:
                decision_cache[key] = decision
            return decision
        try:
            fps_val = float(val)
        except ValueError:
            print("Invalid FPS. Enter a number like 30 or 29.97.")
            continue
        if fps_val <= 0:
            print("FPS must be > 0.")
            continue
        decision = FpsDecision(fps_override=fps_val, fps_policy=None)
        if decision_cache is not None:
            decision_cache[key] = decision
        return decision


@dataclass(frozen=True)
class JobSpec:
    video_path: Path
    audio_path: Optional[Path]
    out_main: Path
    out_video_only: Optional[Path]
    out_audio_only: Optional[Path]
    tmp_video: Path
    tmp_audio: Path
    fps: Optional[float]
    fps_policy: str
    serial_mode: str
    block_size: int
    block_size_div: Optional[int]
    block_strategy: str
    block_min_frames: int
    block_max_frames: Optional[int]
    block_beats_per: int
    block_crossover: str
    block_crossover_frames: int
    block_adsr_attack_s: float
    block_adsr_decay_s: float
    block_adsr_sustain: float
    block_adsr_release_s: float
    block_adsr_curve: str
    s2i_mode: str
    s2i_colorspace: str
    s2i_safe_color: bool
    s2i_chroma_strength: float
    s2i_chroma_clip: float
    i2s_mode: str
    i2s_colorspace: str
    i2s_pad_mode: str
    i2s_radius_mode: str
    i2s_phase_mode: str
    i2s_smoothing: str
    i2s_impulse_len: str
    i2s_impulse_norm: str
    i2s_out_norm: str
    i2s_n_bins: int
    audio_source_for_video_only: Path
    write_main: bool
    write_video_only: bool
    write_audio_only: bool


def _mux(
    video_src: Path,
    audio_src: Path,
    out_path: Path,
    *,
    metadata: dict[str, str] | None = None,
    copy_audio: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_src),
        "-i",
        str(audio_src),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
    ]
    if copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "256k"]
    cmd += ffmpeg_metadata_args(metadata)
    cmd += ["-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _build_biconv_metadata(spec: JobSpec, *, variant: str) -> dict[str, str]:
    settings = {
        "tool": "video-folderbatch",
        "fps": spec.fps,
        "fps_policy": spec.fps_policy,
        "serial_mode": spec.serial_mode,
        "audio_length_mode": "pad-zero",
        "block_size": spec.block_size,
        "block_size_div": spec.block_size_div,
        "block_strategy": spec.block_strategy,
        "block_min_frames": spec.block_min_frames,
        "block_max_frames": spec.block_max_frames,
        "block_beats_per": spec.block_beats_per,
        "block_crossover": spec.block_crossover,
        "block_crossover_frames": spec.block_crossover_frames,
        "block_adsr_attack_s": spec.block_adsr_attack_s,
        "block_adsr_decay_s": spec.block_adsr_decay_s,
        "block_adsr_sustain": spec.block_adsr_sustain,
        "block_adsr_release_s": spec.block_adsr_release_s,
        "block_adsr_curve": spec.block_adsr_curve,
        "s2i_mode": spec.s2i_mode,
        "s2i_colorspace": spec.s2i_colorspace,
        "s2i_safe_color": spec.s2i_safe_color,
        "s2i_chroma_strength": spec.s2i_chroma_strength,
        "s2i_chroma_clip": spec.s2i_chroma_clip,
        "i2s_mode": spec.i2s_mode,
        "i2s_colorspace": spec.i2s_colorspace,
        "i2s_phase_mode": spec.i2s_phase_mode,
        "i2s_impulse_len": spec.i2s_impulse_len,
        "i2s_pad_mode": spec.i2s_pad_mode,
        "i2s_radius_mode": spec.i2s_radius_mode,
        "i2s_smoothing": spec.i2s_smoothing,
        "i2s_impulse_norm": spec.i2s_impulse_norm,
        "i2s_out_norm": spec.i2s_out_norm,
        "i2s_n_bins": spec.i2s_n_bins,
        "audio_source": "video" if spec.audio_path is None else "external",
    }
    return build_exconv_metadata("video-biconv", variant, settings)


def _process_one(spec: JobSpec) -> tuple[bool, str, float]:
    start = time.perf_counter()
    try:
        from exconv.cli.video_biconv import run_video_biconv

        run_video_biconv(
            video_path=spec.video_path,
            audio_path=spec.audio_path,
            out_video=spec.tmp_video,
            out_audio=spec.tmp_audio,
            fps=spec.fps,
            fps_policy=spec.fps_policy,
            mux=False,
            serial_mode=spec.serial_mode,  # type: ignore[arg-type]
            audio_length_mode="pad-zero",
            block_size=spec.block_size,
            block_size_div=spec.block_size_div,
            block_strategy=spec.block_strategy,
            block_min_frames=spec.block_min_frames,
            block_max_frames=spec.block_max_frames,
            block_beats_per=spec.block_beats_per,
            block_crossover=spec.block_crossover,
            block_crossover_frames=spec.block_crossover_frames,
            block_adsr_attack_s=spec.block_adsr_attack_s,
            block_adsr_decay_s=spec.block_adsr_decay_s,
            block_adsr_sustain=spec.block_adsr_sustain,
            block_adsr_release_s=spec.block_adsr_release_s,
            block_adsr_curve=spec.block_adsr_curve,
            s2i_mode=spec.s2i_mode,
            s2i_colorspace=spec.s2i_colorspace,
            i2s_mode=spec.i2s_mode,
            i2s_colorspace=spec.i2s_colorspace,
            i2s_pad_mode=spec.i2s_pad_mode,
            i2s_impulse_len=spec.i2s_impulse_len,
            i2s_radius_mode=spec.i2s_radius_mode,
            i2s_phase_mode=spec.i2s_phase_mode,
            i2s_smoothing=spec.i2s_smoothing,
            i2s_impulse_norm=spec.i2s_impulse_norm,
            i2s_out_norm=spec.i2s_out_norm,
            i2s_n_bins=spec.i2s_n_bins,
            s2i_safe_color=spec.s2i_safe_color,
            s2i_chroma_strength=spec.s2i_chroma_strength,
            s2i_chroma_clip=spec.s2i_chroma_clip,
        )

        if spec.write_main:
            metadata_main = _build_biconv_metadata(spec, variant="biconv")
            _mux(
                spec.tmp_video,
                spec.tmp_audio,
                spec.out_main,
                metadata=metadata_main,
                copy_audio=False,
            )
        if spec.out_video_only is not None and spec.write_video_only:
            metadata_video_only = _build_biconv_metadata(spec, variant="video-only")
            _mux(
                spec.tmp_video,
                spec.audio_source_for_video_only,
                spec.out_video_only,
                metadata=metadata_video_only,
                copy_audio=True,
            )
        if spec.out_audio_only is not None and spec.write_audio_only:
            metadata_audio_only = _build_biconv_metadata(spec, variant="audio-only")
            _mux(
                spec.video_path,
                spec.tmp_audio,
                spec.out_audio_only,
                metadata=metadata_audio_only,
                copy_audio=False,
            )
        for tmp in (spec.tmp_video, spec.tmp_audio):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception as exc:
        return False, f"{spec.video_path.name}: {exc}", time.perf_counter() - start
    return True, spec.video_path.name, time.perf_counter() - start


def _add_video_folderbatch_args(parser: argparse.ArgumentParser) -> None:
    add_settings_args(parser)
    parser.add_argument("project", help="Project name under samples/input/video/<project>.")
    parser.add_argument(
        "--root",
        default="samples",
        help='Root folder (default: "samples").',
    )
    parser.add_argument(
        "--out-project",
        default=None,
        help=(
            "Optional project path for outputs under <root>/output/video/<out-project>. "
            "If omitted, defaults to the same value as --project."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders under the project input folder.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes (default: 1).",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help=(
            "If set, forces NumPy/SciPy threadpools (OMP/MKL/OpenBLAS) to this "
            "many threads (useful when --jobs > 1)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (default: skip).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without processing.",
    )
    parser.add_argument(
        "--suffix",
        default="_biconv",
        help='Suffix for output filenames (default: "_biconv").',
    )

    parser.add_argument(
        "--audio",
        default=None,
        help=(
            "Optional audio file to use for ALL videos. If omitted, audio is "
            "extracted from each input video (ffmpeg required)."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS if metadata missing/incorrect.",
    )
    parser.add_argument(
        "--fps-guard",
        choices=["off", "ask", "auto"],
        default="auto",
        help=(
            "Detect mismatched avg/r frame rates and prompt to override FPS. "
            "Use 'auto' to apply the recommendation without prompting; "
            "'off' keeps metadata FPS."
        ),
    )

    parser.add_argument(
        "--serial-mode",
        choices=["parallel", "serial-image-first", "serial-sound-first"],
        default="parallel",
        help="Bi-conv chaining mode.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Process frames in fixed-size blocks (overrides --block-size-div when set).",
    )
    parser.add_argument(
        "--block-size-div",
        type=int,
        default=12,
        help="Split video into N blocks (default: 12).",
    )
    parser.add_argument(
        "--block-strategy",
        choices=["fixed", "beats", "novelty", "structure"],
        default="fixed",
        help="Block segmentation strategy (fixed or audio-driven).",
    )
    parser.add_argument(
        "--block-min-frames",
        type=int,
        default=1,
        help="Minimum block length (frames) for audio-driven strategies.",
    )
    parser.add_argument(
        "--block-max-frames",
        type=int,
        default=None,
        help="Maximum block length (frames) for audio-driven strategies.",
    )
    parser.add_argument(
        "--beats-per-block",
        type=int,
        default=1,
        help="Group this many beats into a block for --block-strategy beats.",
    )
    parser.add_argument(
        "--crossover",
        choices=["none", "equal", "power", "lin"],
        default="none",
        help="Crossfade mode across block boundaries.",
    )
    parser.add_argument(
        "--crossover-frames",
        type=int,
        default=1,
        help="Frames per block side used for crossover blending.",
    )
    parser.add_argument(
        "--block-adsr-attack-s",
        type=float,
        default=0.0,
        help="ADSR attack (seconds) applied to each block's output audio.",
    )
    parser.add_argument(
        "--block-adsr-decay-s",
        type=float,
        default=0.0,
        help="ADSR decay (seconds) applied to each block's output audio.",
    )
    parser.add_argument(
        "--block-adsr-sustain",
        type=float,
        default=1.0,
        help="ADSR sustain level [0..1] applied to each block's output audio.",
    )
    parser.add_argument(
        "--block-adsr-release-s",
        type=float,
        default=0.0,
        help="ADSR release (seconds) applied to each block's output audio.",
    )
    parser.add_argument(
        "--block-adsr-curve",
        choices=["linear", "equal-energy", "equal-power"],
        default="linear",
        help="Curve shaping for ADSR attack/decay/release segments.",
    )
    parser.add_argument(
        "--s2i-mode",
        choices=["mono", "stereo", "mid-side"],
        default="mid-side",
        help="Sound->image mode.",
    )
    parser.add_argument(
        "--s2i-colorspace",
        choices=["luma", "color"],
        default="color",
        help="Sound->image colorspace.",
    )
    parser.add_argument(
        "--s2i-unsafe-color",
        dest="s2i_safe_color",
        action="store_false",
        help="Disable chroma-safe filtering in color mode.",
    )
    parser.set_defaults(s2i_safe_color=True)
    parser.add_argument(
        "--s2i-chroma-strength",
        type=float,
        default=0.05,
        help="Chroma safety filter strength.",
    )
    parser.add_argument(
        "--s2i-chroma-clip",
        type=float,
        default=0.1,
        help="Chroma safety clip radius around 0.5.",
    )
    parser.add_argument(
        "--i2s-mode",
        choices=["flat", "hist", "radial"],
        default="radial",
        help="Image->sound impulse mode.",
    )
    parser.add_argument(
        "--i2s-colorspace",
        choices=["luma", "rgb-mean", "rgb-stereo", "ycbcr-mid-side"],
        default="ycbcr-mid-side",
        help="Image->sound colorspace.",
    )
    parser.add_argument(
        "--i2s-pad-mode",
        choices=["full", "same-center", "same-first"],
        default="same-center",
        help="Image->sound convolution pad mode.",
    )
    parser.add_argument(
        "--i2s-impulse-len",
        default="frame",
        help="Impulse length (integer, 'auto', or 'frame'=one frame's worth of samples).",
    )
    parser.add_argument(
        "--i2s-radius-mode",
        choices=["linear", "log"],
        default="linear",
        help="Radial binning (radial mode).",
    )
    parser.add_argument(
        "--i2s-phase-mode",
        choices=["zero", "random", "image", "min-phase", "spiral"],
        default="spiral",
        help="Phase strategy (radial mode).",
    )
    parser.add_argument(
        "--i2s-smoothing",
        choices=["none", "hann"],
        default="hann",
        help="Smoothing on radial profile.",
    )
    parser.add_argument(
        "--i2s-impulse-norm",
        choices=["energy", "peak", "none"],
        default="energy",
        help="Impulse normalization.",
    )
    parser.add_argument(
        "--i2s-out-norm",
        choices=["match_rms", "match_peak", "none"],
        default="match_rms",
        help="Output normalization for convolved audio.",
    )
    parser.add_argument(
        "--i2s-n-bins",
        type=int,
        default=256,
        help="Histogram bins (hist mode).",
    )
    parser.add_argument(
        "--variants",
        choices=["both", "all"],
        default="both",
        help=(
            "both: only the main processed video; all: also render video-only "
            "(original audio) and audio-only (original video) variants."
        ),
    )


def _cmd_video_folderbatch(args: argparse.Namespace) -> int:
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    if args.block_size is not None and args.block_size <= 0:
        raise SystemExit("--block-size must be positive")
    if args.block_size_div is not None and args.block_size_div <= 0:
        raise SystemExit("--block-size-div must be positive")
    if args.block_min_frames <= 0:
        raise SystemExit("--block-min-frames must be positive")
    if args.block_max_frames is not None and args.block_max_frames <= 0:
        raise SystemExit("--block-max-frames must be positive")
    if args.beats_per_block <= 0:
        raise SystemExit("--beats-per-block must be positive")
    if args.crossover_frames < 0:
        raise SystemExit("--crossover-frames must be >= 0")
    if args.block_adsr_attack_s < 0:
        raise SystemExit("--block-adsr-attack-s must be >= 0")
    if args.block_adsr_decay_s < 0:
        raise SystemExit("--block-adsr-decay-s must be >= 0")
    if args.block_adsr_release_s < 0:
        raise SystemExit("--block-adsr-release-s must be >= 0")
    if not (0.0 <= args.block_adsr_sustain <= 1.0):
        raise SystemExit("--block-adsr-sustain must be in [0, 1]")
    if args.i2s_n_bins <= 0:
        raise SystemExit("--i2s-n-bins must be positive")

    if args.blas_threads is not None:
        if args.blas_threads <= 0:
            raise SystemExit("--blas-threads must be positive")
        _set_thread_env(args.blas_threads)

    block_strategy = args.block_strategy
    block_size = int(args.block_size) if args.block_size is not None else 1
    block_size_div = None if args.block_size is not None else int(args.block_size_div)
    if block_strategy != "fixed":
        block_size_div = None

    root = _path(args.root)
    project = args.project
    out_project = args.out_project or project
    in_dir = root / "input" / "video" / project
    out_dir = root / "output" / "video" / out_project
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = _path(args.audio) if args.audio else None

    videos = list(_iter_videos(in_dir, recursive=args.recursive))
    if not videos:
        print(f"[video] No videos found in {in_dir}")
        return 1

    guard_mode = args.fps_guard
    if guard_mode == "ask" and not sys.stdin.isatty():
        guard_mode = "auto"
    if guard_mode != "off" and not ffprobe_available():
        print("[fps-guard] ffprobe not found; skipping FPS mismatch checks.")
        guard_mode = "off"
    default_fps_policy = "metadata" if guard_mode == "off" else "auto"

    specs: list[JobSpec] = []
    fps_guard_decisions: dict[tuple[float, float], FpsDecision] = {}
    skipped = 0
    for vp in videos:
        base = f"{vp.stem}{args.suffix}"
        out_main = out_dir / f"{base}{vp.suffix}"
        out_video_only = out_dir / f"{base}_video{vp.suffix}" if args.variants == "all" else None
        out_audio_only = out_dir / f"{base}_audio{vp.suffix}" if args.variants == "all" else None

        targets = [out_main]
        if out_video_only is not None:
            targets.append(out_video_only)
        if out_audio_only is not None:
            targets.append(out_audio_only)

        if all(t.exists() for t in targets) and not args.overwrite:
            skipped += 1
            continue

        tmp_video = out_dir / f"{base}__proc_video{vp.suffix}"
        tmp_audio = out_dir / f"{base}__proc_audio.wav"

        audio_src_for_video_only = _path(args.audio) if args.audio else vp
        write_main = args.overwrite or not out_main.exists()
        write_video_only = args.overwrite or (out_video_only is not None and not out_video_only.exists())
        write_audio_only = args.overwrite or (out_audio_only is not None and not out_audio_only.exists())

        fps_override = args.fps
        fps_policy = default_fps_policy
        if args.fps is None and guard_mode != "off":
            decision = _resolve_fps_override(
                vp,
                default_fps=args.fps,
                guard_mode=guard_mode,
                decision_cache=fps_guard_decisions,
            )
            if decision.fps_override is not None:
                fps_override = decision.fps_override
            if decision.fps_policy is not None:
                fps_policy = decision.fps_policy
        specs.append(
            JobSpec(
                video_path=vp,
                audio_path=audio_path,
                out_main=out_main,
                out_video_only=out_video_only,
                out_audio_only=out_audio_only,
                tmp_video=tmp_video,
                tmp_audio=tmp_audio,
                fps=fps_override,
                fps_policy=fps_policy,
                serial_mode=args.serial_mode,
                block_size=block_size,
                block_size_div=block_size_div,
                block_strategy=block_strategy,
                block_min_frames=int(args.block_min_frames),
                block_max_frames=(
                    int(args.block_max_frames)
                    if args.block_max_frames is not None
                    else None
                ),
                block_beats_per=int(args.beats_per_block),
                block_crossover=str(args.crossover),
                block_crossover_frames=int(args.crossover_frames),
                block_adsr_attack_s=float(args.block_adsr_attack_s),
                block_adsr_decay_s=float(args.block_adsr_decay_s),
                block_adsr_sustain=float(args.block_adsr_sustain),
                block_adsr_release_s=float(args.block_adsr_release_s),
                block_adsr_curve=str(args.block_adsr_curve),
                s2i_mode=args.s2i_mode,
                s2i_colorspace=args.s2i_colorspace,
                s2i_safe_color=bool(args.s2i_safe_color),
                s2i_chroma_strength=float(args.s2i_chroma_strength),
                s2i_chroma_clip=float(args.s2i_chroma_clip),
                i2s_mode=args.i2s_mode,
                i2s_colorspace=args.i2s_colorspace,
                i2s_pad_mode=args.i2s_pad_mode,
                i2s_radius_mode=args.i2s_radius_mode,
                i2s_phase_mode=args.i2s_phase_mode,
                i2s_smoothing=args.i2s_smoothing,
                i2s_impulse_len=str(args.i2s_impulse_len),
                i2s_impulse_norm=args.i2s_impulse_norm,
                i2s_out_norm=args.i2s_out_norm,
                i2s_n_bins=int(args.i2s_n_bins),
                audio_source_for_video_only=audio_src_for_video_only,
                write_main=write_main,
                write_video_only=write_video_only,
                write_audio_only=write_audio_only,
            )
        )

    print(f"[config] in_dir={in_dir}")
    print(f"[config] out_dir={out_dir}")
    print(f"[config] found={len(videos)} queued={len(specs)} skipped={skipped}")
    print(
        f"[config] jobs={args.jobs} variants={args.variants} "
        f"block_strategy={block_strategy} block_size={block_size} "
        f"block_size_div={block_size_div} beats_per_block={args.beats_per_block} "
        f"block_min_frames={args.block_min_frames} block_max_frames={args.block_max_frames}"
    )

    if args.dry_run:
        for spec in specs:
            outs = [spec.out_main]
            if spec.out_video_only:
                outs.append(spec.out_video_only)
            if spec.out_audio_only:
                outs.append(spec.out_audio_only)
            outs_str = ", ".join(p.name for p in outs)
            print(f"[dry-run] {spec.video_path.name} -> {outs_str}")
        return 0

    ok = 0
    failed = 0
    t0 = time.perf_counter()

    if args.jobs == 1:
        for spec in specs:
            success, msg, seconds = _process_one(spec)
            if success:
                ok += 1
                print(f"[ok] {msg} ({seconds:.1f}s)")
            else:
                failed += 1
                print(f"[fail] {msg} ({seconds:.1f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_process_one, spec) for spec in specs]
            for fut in as_completed(futures):
                success, msg, seconds = fut.result()
                if success:
                    ok += 1
                    print(f"[ok] {msg} ({seconds:.1f}s)")
                else:
                    failed += 1
                    print(f"[fail] {msg} ({seconds:.1f}s)")

    total = time.perf_counter() - t0
    print(f"[done] ok={ok} failed={failed} skipped={skipped} total={total/60.0:.1f} min")
    return 0 if failed == 0 else 2


def register_video_folderbatch_subcommand(
    subparsers: argparse._SubParsersAction,
) -> None:
    p = subparsers.add_parser(
        "video-folderbatch",
        help=(
            "Batch-run exconv video-biconv over samples/input/video/<project> "
            "and write to samples/output/video/<project>."
        ),
    )
    _add_video_folderbatch_args(p)
    p.set_defaults(func=_cmd_video_folderbatch)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-folderbatch",
        description=(
            "Batch-run exconv video-biconv over all videos in "
            "samples/input/video/<project> and write to samples/output/video/<project>."
        ),
    )
    _add_video_folderbatch_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    cleaned_argv, settings_path, save_path = strip_settings_args(raw_argv)
    if settings_path:
        settings_data = load_settings(Path(settings_path))
        settings = select_settings(settings_data, "video-folderbatch")
        apply_settings_to_parser(parser, settings)
    args = parser.parse_args(cleaned_argv)
    if save_path:
        exclude = {"settings_path", "save_settings_path", "func"}
        settings_out = serialize_args(args, parser, exclude=exclude)
        save_settings(Path(save_path), settings_out, command="video-folderbatch")
    return _cmd_video_folderbatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
