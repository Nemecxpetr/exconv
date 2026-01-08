from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import imageio.v3 as iio

from exconv.io import read_audio, write_audio, read_image, write_image, as_uint8, write_video_frames
from exconv.conv1d import (
    Audio,
    auto_convolve as audio_auto_convolve,
    pair_convolve as audio_pair_convolve,
)
from exconv.xmodal.sound2image import spectral_sculpt


AUDIO_EXTS = (".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _find_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return False
    return True


def _mux_audio(video_src: Path, audio_src: Path, out_path: Path) -> None:
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
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def process_audio_batch(
    audio_dir: Path,
    out_self_dir: Path,
    out_pair_dir: Path,
    *,
    mode: str = "same-center",
    order: int = 2,
    circular: bool = False,
    normalize: str = "rms",
    subtype: str = "PCM_16",
) -> None:
    """Self + pair convolution for all audio files in audio_dir."""
    audio_files = _find_files(audio_dir, AUDIO_EXTS)
    if not audio_files:
        print(f"[audio] No audio files found in {audio_dir}")
        return

    print(f"[audio] Found {len(audio_files)} files in {audio_dir}")
    _ensure_dir(out_self_dir)
    _ensure_dir(out_pair_dir)

    audio_objs: List[Audio] = []
    for p in audio_files:
        samples, sr = read_audio(p, dtype="float32", always_2d=False)
        audio_objs.append(Audio(samples=samples, sr=sr))
        print(f"  loaded {p.name} (sr={sr}, shape={samples.shape})")

    print("[audio] Running self-convolution...")
    for p, a in zip(audio_files, audio_objs):
        out = audio_auto_convolve(
            a,
            mode=mode,
            circular=circular,
            normalize=normalize,
            order=order,
        )
        out_name = f"{p.stem}__SELF_o{order}.wav"
        out_path = out_self_dir / out_name
        write_audio(out_path, out.samples, out.sr, subtype=subtype)
        print(f"    self -> {out_path}")

    print("[audio] Running pair-convolution (unordered pairs)...")
    for (i, a_i), (j, a_j) in combinations(enumerate(audio_objs), 2):
        p_i = audio_files[i]
        p_j = audio_files[j]

        if a_i.sr != a_j.sr:
            print(
                f"    [skip] {p_i.name} - {p_j.name} (sr mismatch {a_i.sr} vs {a_j.sr})"
            )
            continue

        try:
            out = audio_pair_convolve(
                a_i,
                a_j,
                mode=mode,
                circular=circular,
                normalize=normalize,
            )
        except Exception as exc:
            print(f"    [error] {p_i.name} - {p_j.name}: {exc}")
            continue

        out_name = f"{p_i.stem}__PAIR__{p_j.stem}.wav"
        out_path = out_pair_dir / out_name
        write_audio(out_path, out.samples, out.sr, subtype=subtype)
        print(f"    pair {p_i.name} - {p_j.name} -> {out_path}")


def process_sound2image_batch(
    audio_dir: Path,
    image_dir: Path,
    out_dir: Path,
    *,
    mode: str = "mono",
    colorspace: str = "luma",
    normalize: bool = True,
    animate: bool = False,
    animate_format: str = "gif",
    animate_fps: float = 10.0,
    animate_loop: int = 0,
    animate_audio: bool = False,
) -> None:
    """
    For every image in image_dir and every audio in audio_dir, run spectral_sculpt.
    """
    audio_files = _find_files(audio_dir, AUDIO_EXTS)
    if not audio_files:
        print(f"[sound2image] No audio files found in {audio_dir} - skipping.")
        return

    image_files = _find_files(image_dir, IMAGE_EXTS)
    if not image_files:
        print(f"[sound2image] No image files found in {image_dir} - skipping.")
        return

    print(
        f"[sound2image] {len(image_files)} images x {len(audio_files)} sounds "
        f"from img={image_dir}, audio={audio_dir}"
    )
    _ensure_dir(out_dir)

    audio_data: List[Tuple[Path, np.ndarray, int]] = []
    for p in audio_files:
        a, sr = read_audio(p, dtype="float32", always_2d=True)
        audio_data.append((p, a, sr))
        print(f"  loaded audio {p.name} (sr={sr}, shape={a.shape})")

    anim_paths: Dict[Path, List[Path]] = {}
    anim_dir = out_dir / "animations"
    if animate:
        _ensure_dir(anim_dir)
        anim_paths = {p: [] for p, _, _ in audio_data}

    for img_path in image_files:
        img = read_image(img_path, mode="RGB", dtype="uint8")
        print(f"  loaded image {img_path.name} (shape={img.shape})")

        for a_path, a, sr in audio_data:
            try:
                out_f = spectral_sculpt(
                    image=img,
                    audio=a,
                    sr=sr,
                    mode=mode,
                    colorspace=colorspace,
                    normalize=normalize,
                )
            except Exception as exc:
                print(f"    [error] img={img_path.name} - audio={a_path.name}: {exc}")
                continue

            out_u8 = as_uint8(out_f)
            out_name = f"{img_path.stem}__S2I__{a_path.stem}.png"
            out_path = out_dir / out_name
            write_image(out_path, out_u8)
            print(f"    sound2image {img_path.name} - {a_path.name} -> {out_path}")
            if animate:
                anim_paths[a_path].append(out_path)

    if not animate:
        return

    write_gif = animate_format in {"gif", "both"}
    write_mp4 = animate_format in {"mp4", "both"}
    if write_mp4 and animate_audio and not _ffmpeg_available():
        raise RuntimeError("ffmpeg is required for --s2i-animate-audio.")

    duration = 1.0 / float(animate_fps)
    for a_path, frame_paths in anim_paths.items():
        if not frame_paths:
            continue
        frames = [iio.imread(str(p)) for p in frame_paths]
        stem = a_path.stem
        if write_gif:
            out_gif = anim_dir / f"{stem}__S2I.gif"
            iio.imwrite(
                str(out_gif),
                frames,
                format="GIF",
                loop=int(animate_loop),
                duration=duration,
            )
            print(f"    gif {a_path.name} -> {out_gif}")
        if write_mp4:
            out_mp4 = anim_dir / f"{stem}__S2I.mp4"
            if animate_audio:
                tmp_mp4 = anim_dir / f"{stem}__S2I__video.mp4"
                write_video_frames(tmp_mp4, frames, fps=float(animate_fps))
                _mux_audio(tmp_mp4, a_path, out_mp4)
                try:
                    tmp_mp4.unlink()
                except FileNotFoundError:
                    pass
            else:
                write_video_frames(out_mp4, frames, fps=float(animate_fps))
            print(f"    video {a_path.name} -> {out_mp4}")


def _add_folderbatch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "project",
        help=(
            "Project name. Looks under <root>/input/audio/<project> and "
            "<root>/input/img/<project>."
        ),
    )
    parser.add_argument(
        "--root",
        default="samples",
        help='Root folder (default: "samples").',
    )

    g_audio = parser.add_argument_group("audio-convolution options")
    g_audio.add_argument(
        "--audio-mode",
        choices=["full", "same-first", "same-center"],
        default="same-center",
        help="Linear convolution size policy for audio.",
    )
    g_audio.add_argument(
        "--audio-order",
        type=int,
        default=2,
        help="Self-convolution order (2 = x*x, 3 = x*x*x, ...).",
    )
    g_audio.add_argument(
        "--audio-circular",
        action="store_true",
        help="Use circular convolution for audio (length preserved).",
    )
    g_audio.add_argument(
        "--audio-normalize",
        choices=["rms", "peak", "none"],
        default="rms",
        help="Output normalization mode for audio.",
    )
    g_audio.add_argument(
        "--audio-subtype",
        default="PCM_16",
        help='libsndfile subtype for writing audio (e.g. "PCM_16", "PCM_24", "FLOAT").',
    )

    g_s2i = parser.add_argument_group("sound2image options")
    g_s2i.add_argument(
        "--s2i-mode",
        choices=["mono", "stereo", "mid-side"],
        default="mono",
        help="Sound2image spectral mode.",
    )
    g_s2i.add_argument(
        "--s2i-colorspace",
        choices=["luma", "color"],
        default="luma",
        help="Process only luminance or full YCbCr-based color.",
    )
    g_s2i.add_argument(
        "--s2i-no-normalize",
        dest="s2i_normalize",
        action="store_false",
        help="Disable sound2image output normalization to [0,1].",
    )
    g_s2i.add_argument(
        "--s2i-animate",
        action="store_true",
        help="Write per-audio animations from the sound2image outputs.",
    )
    g_s2i.add_argument(
        "--s2i-animate-format",
        choices=["gif", "mp4", "both"],
        default="mp4",
        help="Animation format when --s2i-animate is set.",
    )
    g_s2i.add_argument(
        "--s2i-animate-fps",
        type=float,
        default=10.0,
        help="Frames per second for sound2image animations.",
    )
    g_s2i.add_argument(
        "--s2i-animate-loop",
        type=int,
        default=0,
        help="GIF loop count (0 = infinite).",
    )
    g_s2i.add_argument(
        "--s2i-animate-audio",
        dest="s2i_animate_audio",
        action="store_true",
        help="Mux source audio into mp4 animations (requires ffmpeg).",
    )
    g_s2i.add_argument(
        "--s2i-animate-no-audio",
        dest="s2i_animate_audio",
        action="store_false",
        help="Disable audio mux for mp4 animations.",
    )
    parser.set_defaults(s2i_normalize=True, s2i_animate_audio=None)


def _cmd_folderbatch(args: argparse.Namespace) -> int:
    root = _path(args.root)
    project = args.project

    audio_dir = root / "input" / "audio" / project
    img_dir = root / "input" / "img" / project

    out_audio_self = root / "output" / "audio" / project / "self"
    out_audio_pair = root / "output" / "audio" / project / "pair"
    out_s2i = root / "output" / "sound2image" / project

    print(f"[config] root={root}")
    print(f"[config] project={project}")
    print(f"[config] audio_dir={audio_dir}")
    print(f"[config] img_dir={img_dir}")
    print(f"[config] out_audio_self={out_audio_self}")
    print(f"[config] out_audio_pair={out_audio_pair}")
    print(f"[config] out_sound2image={out_s2i}")

    if audio_dir.exists():
        process_audio_batch(
            audio_dir,
            out_audio_self,
            out_audio_pair,
            mode=args.audio_mode,
            order=args.audio_order,
            circular=args.audio_circular,
            normalize=args.audio_normalize,
            subtype=args.audio_subtype,
        )
    else:
        print(f"[audio] Audio directory {audio_dir} does not exist - skipping.")

    if img_dir.exists() and audio_dir.exists():
        if args.s2i_animate:
            if args.s2i_animate_fps <= 0:
                raise SystemExit("--s2i-animate-fps must be > 0")
            if args.s2i_animate_loop < 0:
                raise SystemExit("--s2i-animate-loop must be >= 0")
            animate_audio = args.s2i_animate_audio
            if animate_audio is None:
                animate_audio = args.s2i_animate_format in {"mp4", "both"}
            if animate_audio and args.s2i_animate_format == "gif":
                print("[warn] --s2i-animate-audio has no effect when format is gif.")
            if animate_audio and args.s2i_animate_format in {"mp4", "both"} and not _ffmpeg_available():
                raise SystemExit("ffmpeg is required for --s2i-animate-audio.")
        else:
            animate_audio = False

        process_sound2image_batch(
            audio_dir,
            img_dir,
            out_s2i,
            mode=args.s2i_mode,
            colorspace=args.s2i_colorspace,
            normalize=args.s2i_normalize,
            animate=args.s2i_animate,
            animate_format=args.s2i_animate_format,
            animate_fps=args.s2i_animate_fps,
            animate_loop=args.s2i_animate_loop,
            animate_audio=animate_audio,
        )
    else:
        if not img_dir.exists():
            print(f"[sound2image] Image directory {img_dir} does not exist - skipping.")
        if not audio_dir.exists():
            print(
                f"[sound2image] No audio directory {audio_dir}, nothing to combine with images."
            )

    print("[done] folderbatch finished.")
    return 0


def register_folderbatch_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "folderbatch",
        help="Batch audio self/pair convolution and optional sound2image.",
    )
    _add_folderbatch_args(p)
    p.set_defaults(func=_cmd_folderbatch)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="folderbatch",
        description=(
            "Batch self/pair audio convolution and optional sound2image "
            "processing for a project folder."
        ),
    )
    _add_folderbatch_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _cmd_folderbatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
