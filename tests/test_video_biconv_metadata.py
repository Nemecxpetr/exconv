from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

from exconv.xmodal import biconv_video_to_files_stream


@dataclass(frozen=True)
class VideoInfo:
    fps_meta: float | None
    duration_meta: float | None
    size_meta: tuple[int, int] | None
    counted_frames: int
    first_frame_shape: tuple[int, ...]


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if v == float("inf") or v == float("-inf"):
        return None
    return v


def _read_video_info(path: Path) -> VideoInfo:
    import imageio.v3 as iio

    meta = iio.immeta(path)
    meta = dict(meta) if isinstance(meta, dict) else {}
    fps = _safe_float(meta.get("fps"))
    duration = _safe_float(meta.get("duration"))

    size = meta.get("size")
    size_meta = tuple(size) if isinstance(size, (tuple, list)) else None

    frame_iter = iio.imiter(path)
    first = next(frame_iter)
    first_arr = np.asarray(first)
    n = 1 + sum(1 for _ in frame_iter)

    return VideoInfo(
        fps_meta=fps,
        duration_meta=duration,
        size_meta=size_meta,  # type: ignore[arg-type]
        counted_frames=n,
        first_frame_shape=tuple(first_arr.shape),
    )


def _as_rgb_u8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _tile_to_size(img_rgb_u8: np.ndarray, *, H: int, W: int) -> np.ndarray:
    h0, w0 = img_rgb_u8.shape[:2]
    if h0 <= 0 or w0 <= 0:
        raise ValueError("Empty image")
    rep_y = int(np.ceil(H / h0))
    rep_x = int(np.ceil(W / w0))
    tiled = np.tile(img_rgb_u8, (rep_y, rep_x, 1))
    return np.ascontiguousarray(tiled[:H, :W, :])


def _write_scrolling_checker_video(
    out_path: Path,
    *,
    checker_path: Path,
    fps: float,
    n_frames: int,
    size_hw: tuple[int, int],
) -> None:
    import imageio.v3 as iio

    H, W = size_hw
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("Use even H/W for stable yuv420p encoding.")

    img = iio.imread(checker_path)
    base = _tile_to_size(_as_rgb_u8(img), H=H, W=W)

    frames: list[np.ndarray] = []
    for idx in range(n_frames):
        dx = (idx * 3) % W
        dy = (idx * 2) % H
        frame = np.roll(base, shift=(dy, dx), axis=(0, 1))
        frames.append(frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames, fps=float(fps), macro_block_size=None)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_video_biconv_preserves_basic_metadata(test_assets_dir: Path, tmp_path: Path) -> None:
    """
    Generate a tiny deterministic MP4 (checkerboard scroll) with known FPS, then
    run biconv with different settings and ensure the output video's fps/frames/size
    stay consistent.

    Also writes a CSV report into samples/input/test_assets for easy inspection.
    """
    checker = test_assets_dir / "img_checker.png"
    audio = test_assets_dir / "audio_plucks.wav"
    if not checker.exists() or not audio.exists():
        pytest.skip("Missing required test assets (img_checker.png / audio_plucks.wav).")

    in_video = test_assets_dir / "video_checker_scroll_25fps.mp4"
    _write_scrolling_checker_video(
        in_video,
        checker_path=checker,
        fps=25.0,
        n_frames=16,
        size_hw=(96, 128),
    )

    baseline = _read_video_info(in_video)
    if baseline.fps_meta is None or baseline.fps_meta <= 0:
        pytest.fail("Input test video has no valid fps metadata; cannot run test.")

    variants = [
        {"block_size": 1, "i2s_impulse_len": "frame"},
        {"block_size": 1, "i2s_impulse_len": "auto"},
        {"block_size": 4, "i2s_impulse_len": "frame"},
        {"block_size": 4, "i2s_impulse_len": "auto"},
        {"block_size": 4, "i2s_impulse_len": 256},
    ]

    csv_path = test_assets_dir / "video_biconv_metadata_last.csv"
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        rows.append(
            {
                "label": "input",
                "block_size": "",
                "i2s_impulse_len": "",
                "fps_meta": baseline.fps_meta,
                "duration_meta": baseline.duration_meta,
                "counted_frames": baseline.counted_frames,
                "size_meta": baseline.size_meta,
                "first_frame_shape": baseline.first_frame_shape,
                "path": str(in_video),
            }
        )

        for v in variants:
            label = f"bs{v['block_size']}_imp{v['i2s_impulse_len']}"
            out_video = tmp_path / f"{in_video.stem}__{label}{in_video.suffix}"

            biconv_video_to_files_stream(
                video_path=in_video,
                audio_path=audio,
                fps=None,
                serial_mode="parallel",
                audio_length_mode="pad-zero",
                block_size=int(v["block_size"]),
                s2i_mode="mono",
                s2i_colorspace="luma",
                i2s_mode="radial",
                i2s_colorspace="luma",
                i2s_phase_mode="zero",
                i2s_impulse_len=v["i2s_impulse_len"],  # type: ignore[arg-type]
                out_video=out_video,
                out_audio=None,
                mux_output=False,
            )

            info = _read_video_info(out_video)
            rows.append(
                {
                    "label": label,
                    "block_size": v["block_size"],
                    "i2s_impulse_len": v["i2s_impulse_len"],
                    "fps_meta": info.fps_meta,
                    "duration_meta": info.duration_meta,
                    "counted_frames": info.counted_frames,
                    "size_meta": info.size_meta,
                    "first_frame_shape": info.first_frame_shape,
                    "path": str(out_video),
                }
            )

            if info.fps_meta is None:
                errors.append(f"{label}: output fps metadata missing")
            else:
                if abs(info.fps_meta - baseline.fps_meta) > 1e-2:
                    errors.append(f"{label}: fps changed {baseline.fps_meta} -> {info.fps_meta}")

            if info.counted_frames != baseline.counted_frames:
                errors.append(
                    f"{label}: frame count changed {baseline.counted_frames} -> {info.counted_frames}"
                )

            if info.first_frame_shape != baseline.first_frame_shape:
                errors.append(
                    f"{label}: frame shape changed {baseline.first_frame_shape} -> {info.first_frame_shape}"
                )

    finally:
        _write_csv(csv_path, rows)

    if errors:
        pytest.fail("Metadata changed:\n- " + "\n- ".join(errors) + f"\nCSV: {csv_path}")

