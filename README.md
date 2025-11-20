# exconv – Experimental Convolution Toolkit

Experimental convolution toolkit for **sound**, **images**, and simple
**cross-modal** mappings.

- 1D FFT-based convolution for audio (`exconv.conv1d`)
- 2D FFT-based convolution and kernels for images (`exconv.conv2d`)
- Audio↔image experiments (sound-to-image spectral sculpting) (`exconv.xmodal`)
- Unified IO for audio and images (`exconv.io`)
- Minimal CLI for quick demos (`exconv`, `exconv-image`)

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Nemecxpetr/experimental-convolution.git
cd experimental-convolution
pip install -e .
```

After installation, you should have:

```bash
exconv --version
exconv-image --help
```

---

## Quickstart (Python API)

### 1. Audio auto-convolution

```python
import numpy as np
from exconv.io import read_audio, write_audio
from exconv.conv1d import Audio, auto_convolve

# Load file
samples, sr = read_audio("samples/input/audio/drum.wav", dtype="float32")

# Wrap into Audio container
a = Audio(samples=samples, sr=sr)

# 2nd-order self-convolution, same length as input
boom = auto_convolve(
    a,
    mode="same-center",
    circular=False,
    normalize="rms",
    order=2,
)

# Save result
write_audio("samples/output/audio/drum_boom.wav", boom.samples, boom.sr)
```

---

### 2. Image auto/pair convolution

```python
import numpy as np
from exconv.io import read_image, write_image, as_uint8
from exconv.conv2d import image_auto_convolve, image_pair_convolve, gaussian_2d

# Load RGB image as uint8
img = read_image("samples/input/img/photo.png", mode="RGB", dtype="uint8")

# Simple auto-convolution blur in luma space
auto = image_auto_convolve(
    img,
    mode="same-center",
    circular=False,
    colorspace="luma",
    normalize="rescale",
)

# Gaussian pair-convolution in per-channel space
kernel = gaussian_2d(sigma=3.0, truncate=3.0, normalize=True)
pair = image_pair_convolve(
    img,
    kernel=kernel,
    mode="same-center",
    circular=False,
    colorspace="channels",
    normalize="rescale",
)

write_image("samples/output/img/photo_auto.png", as_uint8(auto))
write_image("samples/output/img/photo_gauss.png", as_uint8(pair))
```

---

### 3. Sound → image spectral sculpting

```python
from exconv.io import read_audio, read_image, write_image, as_uint8
from exconv.xmodal.sound2image import spectral_sculpt

# Load image & audio
img = read_image("samples/input/img/glitch_bean.png", mode="RGB", dtype="uint8")
audio, sr = read_audio("samples/input/audio/original.wav", dtype="float32", always_2d=True)

# Mono luma mode: audio spectrum weights radial rings in the luma FFT
out_luma = spectral_sculpt(
    image=img,
    audio=audio,
    sr=sr,
    mode="mono",
    colorspace="luma",
    normalize=True,
)

# Stereo color mode: mid/side information mapped into YCbCr channels
out_color = spectral_sculpt(
    image=img,
    audio=audio,
    sr=sr,
    mode="mid-side",
    colorspace="color",
    normalize=True,
)

write_image("samples/output/img/glitch_bean_mono.png", as_uint8(out_luma))
write_image("samples/output/img/glitch_bean_mid_side.png", as_uint8(out_color))
```

See `docs/design.md` for a deeper explanation of how spectra are mapped into
radial filters and Y/Cb/Cr channels.

---

## CLI usage

The package exposes a small demo CLI in `exconv.cli.exconv_cli`, registered
as the `exconv` command.

### 1. Audio auto-convolution

```bash
exconv audio-auto   --in  input.wav   --out out_auto.wav   --mode same-center   --order 2   --normalize rms
```

Options:

| Option         | Values                            | Description                         |
|----------------|-----------------------------------|-------------------------------------|
| `--mode`       | `full`, `same-first`, `same-center` | Linear size policy.               |
| `--order`      | integer ≥ 1                       | Self-convolution order.            |
| `--circular`   | flag                              | Use circular instead of linear.    |
| `--normalize`  | `rms`, `peak`, `none`             | Output normalization.              |
| `--subtype`    | e.g. `PCM_16`, `PCM_24`, `FLOAT`  | libsndfile subtype.                |

Internally this wraps `exconv.conv1d.auto_convolve`.

---

### 2. Image auto/pair convolution

Auto-convolution:

```bash
exconv img-auto   --in  img.png   --out img_auto.png   --mode same-center   --colorspace channels   --normalize rescale
```

Gaussian kernel convolution:

```bash
exconv img-auto   --in  img.png   --out img_gauss.png   --mode same-center   --colorspace luma   --normalize rescale   --kernel "gaussian:sigma=2.0"
```

Kernel syntax examples:

- `gaussian:sigma=2.0`
- `gaussian:σ=3.0,radius=7`
- `gaussian:sigma=1.5,truncate=4.0`

`exconv img-auto` uses `image_auto_convolve` for pure auto-conv and
`image_pair_convolve` plus a parsed Gaussian kernel when `--kernel` is given.

---

### 3. Sound → image spectral sculpting

```bash
exconv sound2image   --img    img.png   --audio  audio.wav   --out    sculpted.png   --colorspace luma
```

Additional options (subject to change while the experiment evolves):

| Option          | Values                                              | Description                      |
|-----------------|-----------------------------------------------------|----------------------------------|
| `--colorspace`  | `luma`, `channels`                                  | Luma-only vs color processing.  |
| `--stereo-mode` | `collapse`, `mid-side-color`, `lr-color`, `mid-side-angular` | Different stereo mappings. |
| `--min-gain`    | float                                               | Min chroma gain in color modes. |
| `--beta`        | float                                               | Chroma scaling factor.          |
| `--alpha`       | float                                               | Angular modulation strength.    |
| `--no-normalize`| flag                                                | Disable output normalization.   |

The command internally calls the `spectral_sculpt` function in
`exconv.xmodal.sound2image`.

---

### 4. Image demo helper (`exconv-image`)

The `exconv-image` entry point runs a simpler demo for auto/pair convolution
on a single image and saves JPEG outputs:

```bash
exconv-image samples/input/img/glitch_bean.png samples/output/img   --mode same-center   --kernel_path samples/input/img/small_kernel.png
```

Results are stored in `OUTPUT_DIR/<mode>/...`.

---

## Documentation

- **Design notes**: [`docs/design.md`](docs/design.md) – rationale, modes,
  color handling, cross-modal mapping.
- **API reference**: [`docs/api.md`](docs/api.md) – signatures and parameter
  semantics.

---

## License

See the [LICENSE](LICENSE) file in this repository.
