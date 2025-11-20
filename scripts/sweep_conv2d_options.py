import os
import csv
import itertools
from typing import List

import numpy as np
from PIL import Image

from exconv.conv2d.image import auto_convolve, pair_convolve
from exconv.conv2d.kernels import gaussian_2d


# -------------------------------------------------------------------
# Paths: repo-root relative
# -------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "samples", "input", "img")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "samples", "output", "img_sweep")


# -------------------------------------------------------------------
# Parameter grid
# -------------------------------------------------------------------

MODES = ["full", "same-first", "same-center"]
CIRCULARS = [False, True]
COLORSPACES = ["luma", "channels"]
NORMALIZES = ["clip", "rescale", "none"]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def list_images(folder: str, exts: List[str] = None) -> List[str]:
    if exts is None:
        exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    paths = []
    if not os.path.isdir(folder):
        return []
    for name in os.listdir(folder):
        lower = name.lower()
        if any(lower.endswith(e) for e in exts):
            paths.append(os.path.join(folder, name))
    return sorted(paths)


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float outputs to uint8 for saving, keep uint8 as-is."""
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)
    return arr


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def center_crop_to(ref: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """
    Center-crop 'arr' spatially so its HxW matches 'ref'.
    Channels must already match or be singleton-compatible.
    """
    ref = np.asarray(ref)
    arr = np.asarray(arr)

    H0, W0 = ref.shape[:2]
    H1, W1 = arr.shape[:2]

    if H1 < H0 or W1 < W0:
        raise ValueError(
            f"Cannot center-crop: arr smaller than ref "
            f"(arr={arr.shape}, ref={ref.shape})"
        )

    start_h = (H1 - H0) // 2
    start_w = (W1 - W0) // 2

    if arr.ndim == 2:
        return arr[start_h:start_h + H0, start_w:start_w + W0]
    elif arr.ndim == 3:
        return arr[start_h:start_h + H0, start_w:start_w + W0, :]
    else:
        raise ValueError(f"Unsupported ndim for center_crop_to: {arr.ndim}")


# -------------------------------------------------------------------
# Main sweep
# -------------------------------------------------------------------

def main():
    print(f"INPUT_DIR : {INPUT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = list_images(INPUT_DIR)
    if not images:
        print(f"No images found in {INPUT_DIR}")
        return

    # CSV for metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    csv_fields = [
        "image",
        "conv_type",
        "mode",
        "circular",
        "colorspace",
        "normalize",
        "out_shape",
        "mse_vs_input",
        "mae_vs_input",
    ]
    csv_file = open(metrics_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    # Fixed Gaussian kernel for pair convolution
    gaussian_kernel = gaussian_2d(sigma=2.0)

    for img_path in images:
        print(f"\n=== Image: {img_path} ===")
        base = os.path.splitext(os.path.basename(img_path))[0]

        # Load input as RGB uint8
        img = np.asarray(Image.open(img_path).convert("RGB"))

        # Precompute luma reference
        img_luma = np.dot(
            img[..., :3].astype(np.float64),
            np.array([0.2126, 0.7152, 0.0722])
        )

        for mode, circular, colorspace, normalize in itertools.product(
            MODES, CIRCULARS, COLORSPACES, NORMALIZES
        ):
            print(
                f"  -> mode={mode}, circular={circular}, "
                f"colorspace={colorspace}, norm={normalize}"
            )

            # ---------------- AUTO-CONVOLUTION ----------------
            out_auto = auto_convolve(
                img,
                mode=mode,
                circular=circular,
                colorspace=colorspace,
                normalize=normalize,
            )

            # Metrics
            if out_auto.ndim == 2:
                ref = img_luma
                out_cmp = out_auto
                if out_cmp.shape != ref.shape:
                    out_cmp = center_crop_to(ref, out_cmp)
            else:
                ref = img
                out_cmp = out_auto
                if out_cmp.shape != ref.shape:
                    out_cmp = center_crop_to(ref, out_cmp)

            m_mse = mse(out_cmp, ref)
            m_mae = mae(out_cmp, ref)

            # Save image
            auto_dir = os.path.join(OUTPUT_DIR, base, "auto")
            os.makedirs(auto_dir, exist_ok=True)

            auto_name = (
                f"{base}"
                f"_auto"
                f"_mode-{mode}"
                f"_circ-{int(circular)}"
                f"_cs-{colorspace}"
                f"_norm-{normalize}.jpg"
            )
            auto_path = os.path.join(auto_dir, auto_name)

            auto_to_save = ensure_uint8(out_auto)
            Image.fromarray(auto_to_save).save(auto_path, quality=95)

            # Save diff image
            diff_ref = ref
            diff_out = center_crop_to(diff_ref, out_auto)
            diff = np.abs(diff_out.astype(np.float64) - diff_ref.astype(np.float64))
            diff = diff / (diff.max() + 1e-9) * 255.0
            diff = diff.astype(np.uint8)

            diff_name = auto_name.replace(".jpg", "_DIFF.jpg")
            diff_path = os.path.join(auto_dir, diff_name)
            Image.fromarray(diff).save(diff_path, quality=95)

            # Metrics row for auto
            writer.writerow(
                dict(
                    image=base,
                    conv_type="auto",
                    mode=mode,
                    circular=int(circular),
                    colorspace=colorspace,
                    normalize=normalize,
                    out_shape=str(out_auto.shape),
                    mse_vs_input=m_mse,
                    mae_vs_input=m_mae,
                )
            )

            # ---------------- PAIR-CONVOLUTION ----------------
            out_pair = pair_convolve(
                img,
                kernel=gaussian_kernel,
                mode=mode,
                circular=circular,
                colorspace=colorspace,
                normalize=normalize,
            )

            if out_pair.ndim == 2:
                ref = img_luma
                out_cmp = out_pair
                if out_cmp.shape != ref.shape:
                    out_cmp = center_crop_to(ref, out_cmp)
            else:
                ref = img
                out_cmp = out_pair
                if out_cmp.shape != ref.shape:
                    out_cmp = center_crop_to(ref, out_cmp)

            m_mse = mse(out_cmp, ref)
            m_mae = mae(out_cmp, ref)

            # Save image
            pair_dir = os.path.join(OUTPUT_DIR, base, "pair")
            os.makedirs(pair_dir, exist_ok=True)

            pair_name = (
                f"{base}"
                f"_pair"
                f"_mode-{mode}"
                f"_circ-{int(circular)}"
                f"_cs-{colorspace}"
                f"_norm-{normalize}.jpg"
            )
            pair_path = os.path.join(pair_dir, pair_name)

            pair_to_save = ensure_uint8(out_pair)
            Image.fromarray(pair_to_save).save(pair_path, quality=95)

            # Save diff image
            diff_ref = ref
            diff_out = center_crop_to(diff_ref, out_pair)
            diff = np.abs(diff_out.astype(np.float64) - diff_ref.astype(np.float64))
            diff = diff / (diff.max() + 1e-9) * 255.0
            diff = diff.astype(np.uint8)

            diff_name = pair_name.replace(".jpg", "_DIFF.jpg")
            diff_path = os.path.join(pair_dir, diff_name)
            Image.fromarray(diff).save(diff_path, quality=95)

            # Metrics row for pair
            writer.writerow(
                dict(
                    image=base,
                    conv_type="pair",
                    mode=mode,
                    circular=int(circular),
                    colorspace=colorspace,
                    normalize=normalize,
                    out_shape=str(out_pair.shape),
                    mse_vs_input=m_mse,
                    mae_vs_input=m_mae,
                )
            )

    csv_file.close()
    print(f"\nFinished sweep.")
    print(f"Metrics written to: {metrics_path}")
    print(f"Outputs written under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
