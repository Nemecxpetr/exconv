"""
Generate example outputs for the exconv library in a modular way.

Features
--------
- Scans an input folder for *any* audio + image files (no hard-coded names).
- If none are found, optionally calls `generate_test_assets.main()` to create
  tiny synthetic assets (audio + images).
- Produces:
    * audio auto-convolution examples (multiple modes)
    * audio pair-convolution examples (multiple modes)
    * image auto + Gaussian pair convolution examples (multiple modes)
    * sound→image spectral sculpting for several modes

Outputs
-------
samples/output/examples/audio/
samples/output/examples/images/
samples/output/examples/sound2image/

Run
---
python scripts/generate_examples.py
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

# ---------------------------------------------------------------------
# Locate project root (directory containing "scripts")
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

# ---------------------------------------------------------------------
# Import asset generator from scripts/generate_test_assests.py
# ---------------------------------------------------------------------
GEN_ASSETS_PATH = SCRIPTS_DIR / "generate_test_assests.py"

if GEN_ASSETS_PATH.exists():
    # dynamic import: safe even if filename has a typo
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_test_assests", str(GEN_ASSETS_PATH)
    )
    gen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_mod)      # type: ignore
    generate_assets_main = gen_mod.main
else:
    generate_assets_main = None

# ---------------------------------------------------------------------
# exconv imports
# ---------------------------------------------------------------------
from exconv.io import (
    read_audio,
    write_audio,
    read_image,
    write_image,
    as_uint8,
)
from exconv.conv1d import Audio, auto_convolve as audio_auto, pair_convolve as audio_pair
from exconv.conv2d import (
    image_auto_convolve,
    image_pair_convolve,
    gaussian_2d,
)
from exconv.xmodal.sound2image import spectral_sculpt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
INPUT_DIR = PROJECT_ROOT / "samples" / "input" / "test_assets"
OUTPUT_DIR = PROJECT_ROOT / "samples" / "output" / "examples"

AUDIO_OUT_DIR = OUTPUT_DIR / "audio"
IMAGE_OUT_DIR = OUTPUT_DIR / "images"
S2I_OUT_DIR = OUTPUT_DIR / "sound2image"

AUDIO_EXT = {".wav", ".flac", ".aiff", ".aif", ".ogg"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_files(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    audio_files = []
    image_files = []

    if not input_dir.exists():
        return audio_files, image_files

    for p in sorted(input_dir.iterdir()):
        if p.is_file():
            if p.suffix.lower() in AUDIO_EXT:
                audio_files.append(p)
            elif p.suffix.lower() in IMG_EXT:
                image_files.append(p)

    return audio_files, image_files


def maybe_generate_assets(input_dir: Path):
    audio_files, image_files = find_files(input_dir)

    if audio_files and image_files:
        return audio_files, image_files

    # No assets found
    print(f"No test assets found in: {input_dir}")
    print("I can generate synthetic test audio + images.")
    print("Estimated size: ~8–10 MB total for inputs + outputs.")
    ans = input("Generate test assets now? [y/N]: ").strip().lower()

    if ans not in ("y", "yes"):
        return audio_files, image_files

    if generate_assets_main is None:
        print("ERROR: Could not import generate_test_assests.py")
        return audio_files, image_files

    # ensure directory exists
    input_dir.mkdir(parents=True, exist_ok=True)
    print("Generating assets...")
    generate_assets_main(str(input_dir))

    return find_files(input_dir)


# ---------------------------------------------------------------------
# Example Generators
# ---------------------------------------------------------------------

def run_audio_examples(audio_files: List[Path]) -> None:
    if not audio_files:
        print("No audio files — skipping audio examples.")
        return

    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    modes = ["full", "same-first", "same-center"]

    # -----------------------
    # Auto-convolution
    # -----------------------
    print("\n=== Audio auto-convolution (multiple modes) ===")

    for ap in audio_files:
        print(f"  [auto] {ap.name}")
        samples, sr = read_audio(ap, dtype="float32", always_2d=False)
        audio = Audio(samples=samples, sr=sr)

        # Linear modes
        for mode in modes:
            out = audio_auto(
                audio,
                mode=mode,
                circular=False,
                normalize="rms",
                order=2,
            )
            out_path = AUDIO_OUT_DIR / f"{ap.stem}_auto_{mode}.wav"
            write_audio(out_path, out.samples, out.sr, subtype="PCM_16")
            print(f"     -> {out_path.name}")

        # Circular
        out_circ = audio_auto(
            audio,
            mode="same-center",
            circular=True,
            normalize="rms",
            order=2,
        )
        out_circ_path = AUDIO_OUT_DIR / f"{ap.stem}_auto_circular.wav"
        write_audio(out_circ_path, out_circ.samples, out_circ.sr, subtype="PCM_16")
        print(f"     -> {out_circ_path.name} (circular)")

    # -----------------------
    # Pair-convolution
    # -----------------------
    if len(audio_files) < 2:
        print("\nNot enough audio files for pair-convolution examples (need ≥ 2).")
        return

    print("\n=== Audio pair-convolution (multiple modes) ===")
    # Pair each file with the next one (wrap around)
    for i, ap in enumerate(audio_files):
        bp = audio_files[(i + 1) % len(audio_files)]
        print(f"  [pair] {ap.name} * {bp.name}")

        x_samples, sr_x = read_audio(ap, dtype="float32", always_2d=False)
        h_samples, sr_h = read_audio(bp, dtype="float32", always_2d=False)

        x = Audio(samples=x_samples, sr=sr_x)
        h = Audio(samples=h_samples, sr=sr_h)

        # Linear modes
        for mode in modes:
            y = audio_pair(
                x,
                h,
                mode=mode,
                circular=False,
                normalize="rms",
            )
            out_path = AUDIO_OUT_DIR / f"{ap.stem}_PAIR_{bp.stem}_{mode}.wav"
            write_audio(out_path, y.samples, y.sr, subtype="PCM_16")
            print(f"     -> {out_path.name}")

        # Circular
        y_circ = audio_pair(
            x,
            h,
            mode="same-center",
            circular=True,
            normalize="rms",
        )
        out_circ_path = AUDIO_OUT_DIR / f"{ap.stem}_PAIR_{bp.stem}_circular.wav"
        write_audio(out_circ_path, y_circ.samples, y_circ.sr, subtype="PCM_16")
        print(f"     -> {out_circ_path.name} (circular)")


def run_image_examples(image_files: List[Path]) -> None:
    if not image_files:
        print("No images — skipping image examples.")
        return

    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== Image auto & Gaussian pair-convolution (multiple modes) ===")

    kernel = gaussian_2d(sigma=3.0, truncate=3.0, normalize=True)
    modes = ["full", "same-first", "same-center"]

    for ip in image_files:
        print(f"  -> {ip.name}")
        img = read_image(ip, mode="RGB", dtype="uint8")

        # Linear modes
        for mode in modes:
            auto = image_auto_convolve(
                img,
                mode=mode,
                circular=False,
                colorspace="channels",
                normalize="rescale",
            )
            auto_path = IMAGE_OUT_DIR / f"{ip.stem}_auto_{mode}.png"
            write_image(auto_path, as_uint8(auto))

            pair = image_pair_convolve(
                img,
                kernel=kernel,
                mode=mode,
                circular=False,
                colorspace="channels",
                normalize="rescale",
            )
            pair_path = IMAGE_OUT_DIR / f"{ip.stem}_gauss_{mode}.png"
            write_image(pair_path, as_uint8(pair))

        # Circular examples
        auto_circ = image_auto_convolve(
            img,
            mode="same-center",
            circular=True,
            colorspace="channels",
            normalize="rescale",
        )
        auto_circ_path = IMAGE_OUT_DIR / f"{ip.stem}_auto_circular.png"
        write_image(auto_circ_path, as_uint8(auto_circ))

        pair_circ = image_pair_convolve(
            img,
            kernel=kernel,
            mode="same-center",
            circular=True,
            colorspace="channels",
            normalize="rescale",
        )
        pair_circ_path = IMAGE_OUT_DIR / f"{ip.stem}_gauss_circular.png"
        write_image(pair_circ_path, as_uint8(pair_circ))


def run_sound2image_examples(image_files: List[Path], audio_files: List[Path]) -> None:
    if not image_files or not audio_files:
        print("Missing audio or images — skipping sound2image examples.")
        return

    S2I_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== Sound2Image spectral sculpting ===")

    modes = ["mono", "stereo", "mid-side"]

    for ap in audio_files:
        audio, sr = read_audio(ap, dtype="float32", always_2d=True)
        print(f"\nAudio: {ap.name}")

        for ip in image_files:
            img = read_image(ip, mode="RGB", dtype="uint8")
            print(f"  Image: {ip.name}")

            for mode in modes:
                try:
                    out = spectral_sculpt(
                        image=img,
                        audio=audio,
                        sr=sr,
                        mode=mode,
                        colorspace="color",
                        normalize=True,
                    )
                except Exception as e:
                    print(f"    [ERROR] {mode}: {e}")
                    continue

                out_path = (
                    S2I_OUT_DIR /
                    f"{ip.stem}__{ap.stem}__s2i_{mode}.png"
                )
                write_image(out_path, as_uint8(out))
                print(f"    -> {out_path.name}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    audio_files, image_files = maybe_generate_assets(INPUT_DIR)

    if not audio_files and not image_files:
        print("No assets available — nothing to do.")
        return 1

    run_audio_examples(audio_files)
    run_image_examples(image_files)
    run_sound2image_examples(image_files, audio_files)

    print("\nDone. Example outputs saved under:")
    print(f"  {OUTPUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
