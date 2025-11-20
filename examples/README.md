# Example Asset Generator for **exconv**

This folder contains tools for generating audio, image, and sound-to-image examples demonstrating the capabilities of the **exconv** library.

---

## Overview

The main script is:

```
scripts/generate_examples.py
```

It automatically produces a full suite of example outputs for:

* **1D audio convolution**

  * auto-convolution
  * pair-convolution
  * linear modes: `full`, `same-first`, `same-center`
  * circular convolution

* **2D image convolution**

  * auto-convolution
  * Gaussian pair-convolution
  * linear + circular modes

* **Sound → Image spectral sculpting**

  * mono mode
  * stereo YCbCr mode
  * mid-side mode
  * applied to every combination of audio × image

Generated examples are stored in:

```
samples/output/examples/
```

---

## Requirements

Install the library:

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Input Assets

Place your assets in:

```
samples/input/test_assets/
```

### Supported formats

| Type  | Extensions                           |
| ----- | ------------------------------------ |
| Audio | .wav, .flac, .aiff, .aif, .ogg       |
| Image | .png, .jpg, .jpeg, .tif, .tiff, .bmp |

If no assets are present, the script asks whether it should generate synthetic test assets.

---

## Running the Generator

From the project root:

```bash
python scripts/generate_examples.py
```

This creates audio, image, and sound2image examples using all available input files.

---

## Output Structure

```
samples/
└── output/
    └── examples/
        ├── audio/
        ├── images/
        └── sound2image/
```

### Audio examples

* `{name}_auto_{mode}.wav`
* `{A}_PAIR_{B}_{mode}.wav`

Modes include: `full`, `same-first`, `same-center`, and `circular`.

### Image examples

* `{name}_auto_{mode}.png`
* `{name}_gauss_{mode}.png`

### Sound2Image examples

* `{image}__{audio}__s2i_{mode}.png`

Modes: `mono`, `stereo`, `mid-side`.

---

## How Convolution Works (High-Level)

### Audio

FFT-based linear and circular convolution using `exconv.conv1d.Audio`.

### Image

Per-channel 2D convolution, including Gaussian kernels.

### Sound-to-Image

Audio spectrum → radial remapping → applied to luminance and/or chroma channels depending on mode.

---

## License

See the project `LICENSE` file.

---

## Note on Script Organization

You can optionally move example scripts into a top-level `examples/` folder:

```
examples/
└── generate_examples.py
```

This is common in Python projects and keeps `scripts/` reserved for internal utilities.

Both choices work — moving to `examples/` is cleaner if you consider them public-facing demonstration scripts.
