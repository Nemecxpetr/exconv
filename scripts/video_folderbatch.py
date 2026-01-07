#!/usr/bin/env python
"""
scripts/video_folderbatch.py

Batch bi-directional video convolution for a "project" folder.

Directory convention
--------------------
Given a project name, e.g. "my_project", this script looks for:

    samples/input/video/<project>/*

and writes:

    samples/output/video/<project>/*

With --variants all, it renders three outputs per input:
- <name><suffix><ext>              (video + audio processed)
- <name><suffix>_video<ext>        (video processed, original audio)
- <name><suffix>_audio<ext>        (original video, audio processed)

Use --out-project to send outputs to a different subfolder under
samples/output/video while keeping inputs under <project>.

By default it runs the equivalent pipeline for each input video:

    exconv video-biconv
      --s2i-mode mid-side
      --s2i-colorspace color
      --s2i-chroma-strength 0.05
      --s2i-chroma-clip 0.1
      --i2s-mode radial
      --i2s-colorspace ycbcr-mid-side
      --i2s-phase-mode spiral
      --i2s-impulse-len frame
      --serial-mode parallel
      --block-size 1
      --block-size-div 12

Notes on performance
--------------------
- Use --jobs > 1 to process multiple videos concurrently (separate processes).
- If you use --jobs > 1, consider setting --blas-threads 1 to avoid CPU
  oversubscription from NumPy/SciPy threadpools.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import subprocess


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


def _path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _iter_videos(root: Path, recursive: bool) -> Iterable[Path]:
    if not root.exists():
        return []
    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.iterdir()
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
class JobSpec:
    video_path: Path
    audio_path: Optional[Path]
    out_main: Path
    out_video_only: Optional[Path]
    out_audio_only: Optional[Path]
    tmp_video: Path
    tmp_audio: Path
    fps: Optional[float]
    # requested pipeline defaults
    serial_mode: str
    block_size: int
    block_size_div: Optional[int]
    s2i_mode: str
    s2i_colorspace: str
    s2i_safe_color: bool
    s2i_chroma_strength: float
    s2i_chroma_clip: float
    i2s_mode: str
    i2s_colorspace: str
    i2s_phase_mode: str
    i2s_impulse_len: str
    audio_source_for_video_only: Path
    write_main: bool
    write_video_only: bool
    write_audio_only: bool


def _mux(video_src: Path, audio_src: Path, out_path: Path, *, copy_audio: bool = False) -> None:
    """
    Lightweight mux helper; copy video stream, optionally copy audio.
    """
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
    cmd += ["-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _process_one(spec: JobSpec) -> tuple[bool, str, float]:
    """
    Returns: (ok, message, seconds)
    """
    start = time.perf_counter()
    try:
        from exconv.cli.video_biconv import run_video_biconv

        run_video_biconv(
            video_path=spec.video_path,
            audio_path=spec.audio_path,
            out_video=spec.tmp_video,
            out_audio=spec.tmp_audio,
            fps=spec.fps,
            mux=False,
            serial_mode=spec.serial_mode,  # type: ignore[arg-type]
            audio_length_mode="pad-zero",  # fixed default for this batch script
            block_size=spec.block_size,
            block_size_div=spec.block_size_div,
            s2i_mode=spec.s2i_mode,
            s2i_colorspace=spec.s2i_colorspace,
            i2s_mode=spec.i2s_mode,
            i2s_colorspace=spec.i2s_colorspace,
            i2s_pad_mode="same-center",
            i2s_impulse_len=spec.i2s_impulse_len,
            i2s_radius_mode="linear",
            i2s_phase_mode=spec.i2s_phase_mode,
            i2s_smoothing="hann",
            i2s_impulse_norm="energy",
            i2s_out_norm="match_rms",
            i2s_n_bins=256,
            s2i_safe_color=spec.s2i_safe_color,
            s2i_chroma_strength=spec.s2i_chroma_strength,
            s2i_chroma_clip=spec.s2i_chroma_clip,
        )

        # Render variants (render-only, reuse processed video/audio)
        if spec.write_main:
            _mux(spec.tmp_video, spec.tmp_audio, spec.out_main, copy_audio=False)
        if spec.out_video_only is not None and spec.write_video_only:
            _mux(
                spec.tmp_video,
                spec.audio_source_for_video_only,
                spec.out_video_only,
                copy_audio=True,
            )
        if spec.out_audio_only is not None and spec.write_audio_only:
            _mux(
                spec.video_path,
                spec.tmp_audio,
                spec.out_audio_only,
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_folderbatch",
        description=(
            "Batch-run exconv video-biconv over all videos in "
            "samples/input/video/<project> and write to samples/output/video/<project>."
        ),
    )

    p.add_argument("project", help="Project name under samples/input/video/<project>.")
    p.add_argument(
        "--root",
        default="samples",
        help='Root folder (default: "samples").',
    )
    p.add_argument(
        "--out-project",
        default=None,
        help=(
            "Optional project path for outputs under <root>/output/video/<out-project>. "
            "If omitted, defaults to the same value as --project."
        ),
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders under the project input folder.",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes (default: 1).",
    )
    p.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help=(
            "If set, forces NumPy/SciPy threadpools (OMP/MKL/OpenBLAS) to this "
            "many threads (useful when --jobs > 1)."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (default: skip).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without processing.",
    )
    p.add_argument(
        "--suffix",
        default="_biconv",
        help='Suffix for output filenames (default: "_biconv").',
    )

    # basic video-biconv toggles
    p.add_argument(
        "--audio",
        default=None,
        help=(
            "Optional audio file to use for ALL videos. If omitted, audio is "
            "extracted from each input video (ffmpeg required)."
        ),
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS if metadata missing/incorrect.",
    )

    # pipeline options (default to the requested settings)
    p.add_argument(
        "--serial-mode",
        choices=["parallel", "serial-image-first", "serial-sound-first"],
        default="parallel",
        help="Bi-conv chaining mode.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Process frames in fixed-size blocks (overrides --block-size-div when set).",
    )
    p.add_argument(
        "--block-size-div",
        type=int,
        default=12,
        help="Split video into N blocks (default: 12).",
    )
    p.add_argument(
        "--s2i-mode",
        choices=["mono", "stereo", "mid-side"],
        default="mid-side",
        help="Sound->image mode.",
    )
    p.add_argument(
        "--s2i-colorspace",
        choices=["luma", "color"],
        default="color",
        help="Sound->image colorspace.",
    )
    p.add_argument(
        "--s2i-unsafe-color",
        dest="s2i_safe_color",
        action="store_false",
        help="Disable chroma-safe filtering in color mode.",
    )
    p.set_defaults(s2i_safe_color=True)
    p.add_argument(
        "--s2i-chroma-strength",
        type=float,
        default=0.05,
        help="Chroma safety filter strength.",
    )
    p.add_argument(
        "--s2i-chroma-clip",
        type=float,
        default=0.1,
        help="Chroma safety clip radius around 0.5.",
    )
    p.add_argument(
        "--i2s-mode",
        choices=["flat", "hist", "radial"],
        default="radial",
        help="Image->sound impulse mode.",
    )
    p.add_argument(
        "--i2s-colorspace",
        choices=["luma", "rgb-mean", "rgb-stereo", "ycbcr-mid-side"],
        default="ycbcr-mid-side",
        help="Image->sound colorspace.",
    )
    p.add_argument(
        "--i2s-phase-mode",
        choices=["zero", "random", "image", "min-phase", "spiral"],
        default="spiral",
        help="Phase strategy (radial mode).",
    )
    p.add_argument(
        "--i2s-impulse-len",
        default="frame",
        help="Impulse length (integer, 'auto', or 'frame'=one frame's worth of samples).",
    )
    p.add_argument(
        "--variants",
        choices=["both", "all"],
        default="both",
        help=(
            "both: only the main processed video; all: also render video-only "
            "(original audio) and audio-only (original video) variants."
        ),
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    if args.block_size is not None and args.block_size <= 0:
        raise SystemExit("--block-size must be positive")
    if args.block_size_div is not None and args.block_size_div <= 0:
        raise SystemExit("--block-size-div must be positive")

    if args.blas_threads is not None:
        if args.blas_threads <= 0:
            raise SystemExit("--blas-threads must be positive")
        _set_thread_env(args.blas_threads)

    block_size = int(args.block_size) if args.block_size is not None else 1
    block_size_div = None if args.block_size is not None else int(args.block_size_div)

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

    specs: list[JobSpec] = []
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

        specs.append(
            JobSpec(
                video_path=vp,
                audio_path=audio_path,
                out_main=out_main,
                out_video_only=out_video_only,
                out_audio_only=out_audio_only,
                tmp_video=tmp_video,
                tmp_audio=tmp_audio,
                fps=args.fps,
                serial_mode=args.serial_mode,
                block_size=block_size,
                block_size_div=block_size_div,
                s2i_mode=args.s2i_mode,
                s2i_colorspace=args.s2i_colorspace,
                s2i_safe_color=bool(args.s2i_safe_color),
                s2i_chroma_strength=float(args.s2i_chroma_strength),
                s2i_chroma_clip=float(args.s2i_chroma_clip),
                i2s_mode=args.i2s_mode,
                i2s_colorspace=args.i2s_colorspace,
                i2s_phase_mode=args.i2s_phase_mode,
                i2s_impulse_len=str(args.i2s_impulse_len),
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
        f"block_size={block_size} block_size_div={block_size_div}"
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


if __name__ == "__main__":
    raise SystemExit(main())
