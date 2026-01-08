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
git clone https://github.com/Nemecxpetr/exconv.git
cd exconv
pip install -e .
```

Notes:
- `ffmpeg` is required for video audio extraction/muxing (used by `imageio[ffmpeg]` and the `video-biconv` command).
- `tqdm` provides progress bars for per-frame processing.

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

### Getting started (bundled assets)

Use the included samples for a quick spin:

- Audio auto: `exconv audio-auto --in samples/input/audio/drum.wav --out samples/output/audio/drum_auto.wav --order 2`
- Image auto: `exconv img-auto --in samples/input/img/glitch_bean.png --out samples/output/img/glitch_auto.png`
- Sound→image: `exconv sound2image --img samples/input/img/glitch_bean.png --audio samples/input/audio/original.wav --out samples/output/img/glitch_sculpt.png --colorspace luma`
- Image→sound: `python scripts/image2sound_demo.py --audio samples/input/test_assets/audio_long_sines.wav --image samples/input/test_assets/img_checker.png --mode radial --impulse-len auto --phase-mode spiral --out-dir samples/output/audio/img2sound_demo`
- Bi-conv video: `exconv video-biconv --video samples/input/video/test_01.mp4 --out-video samples/output/video/test_01_biconv.mp4 --out-audio samples/output/audio/test_01_biconv.wav --serial-mode parallel --audio-length-mode pad-zero --i2s-phase-mode spiral --i2s-impulse-len auto`
- Batch audio + sound2image: `exconv folderbatch my_project --root samples --audio-mode same-center --audio-order 2 --audio-normalize rms`
- Batch video: `exconv video-folderbatch my_project --jobs 2 --suffix _biconv --serial-mode parallel`
- Animate frames: `exconv animate samples/output/sound2image/my_project/animations out.mp4 --fps 12`

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

### 5. Image -> sound demo (scripts/image2sound_demo.py)

Derive an impulse from an image and convolve audio with it (flat, histogram, or radial FFT mapping):

```bash
python scripts/image2sound_demo.py \
  --audio samples/input/test_assets/audio_long_sines.wav \
  --image samples/input/test_assets/img_checker.png \
  --mode radial \
  --colorspace ycbcr-mid-side \
  --phase-mode spiral \
  --impulse-len auto
```

Key options:
- `--mode`: `flat`, `hist`, `radial`
- `--colorspace`: `luma`, `rgb-mean`, `rgb-stereo`, `ycbcr-mid-side`
- `--phase-mode` (radial): `zero`, `random`, `image`, `min-phase`, `spiral`
- `--impulse-len`: integer or `auto` (match input audio length)
- `--pad-mode`: `same-center` (default), `same-first`, `full`

Outputs: convolved audio WAV and an impulse visualization PNG in `--out-dir`.

### 6. Bi-directional video convolution (`exconv video-biconv`)

Per-frame sound->image and image->sound with parallel/serial chaining. Provide a video for frames and an audio file (can be the original track or external):

```bash
exconv video-biconv \
  --video input.mp4 \
  --out-video out_biconv.mp4 \
  --out-audio out_biconv.wav \
  --serial-mode parallel \
  --audio-length-mode pad-zero \
  --s2i-mode mono --s2i-colorspace luma \
  --i2s-mode radial --i2s-colorspace ycbcr-mid-side \
  --i2s-phase-mode spiral --i2s-impulse-len auto
```

Core controls:

| Option | Default | Notes |
|--------|---------|-------|
| `--audio` | None | If omitted, audio is extracted from input video (ffmpeg required). Can also be a video path. |
| `--fps` | None | Override FPS if metadata is missing/incorrect. |
| `--fps-policy` | `auto` | How FPS is chosen when `--fps` is unset (`auto`, `metadata`, `avg_frame_rate`, `r_frame_rate`). |
| `--mux/--no-mux` | `--mux` | Mux processed audio into output video (requires ffmpeg). |
| `--serial-mode` | `parallel` | `parallel`, `serial-image-first`, `serial-sound-first`. |
| `--audio-length-mode` | `pad-zero` | See strategies table below. |

Block segmentation:

| Option | Default | Notes |
|--------|---------|-------|
| `--block-strategy` | `fixed` | `fixed` (frame-count), `beats`/`novelty`/`structure` (audio-driven). |
| `--block-size` | `1` | Fixed frames per block (fixed strategy only). |
| `--block-size-div` | None | Split into N blocks (fixed strategy only). |
| `--block-min-frames` | `1` | Minimum block length (audio-driven strategies). |
| `--block-max-frames` | None | Maximum block length (audio-driven strategies). |
| `--beats-per-block` | `1` | Group this many beats into a single block (beats strategy). |

Sound->image (`s2i-*`):

| Option | Values | Notes |
|--------|--------|-------|
| `--s2i-mode` | `mono`, `stereo`, `mid-side` | Matches `spectral_sculpt` modes. |
| `--s2i-colorspace` | `luma`, `color` | Sound->image colorspace. |
| `--s2i-safe-color/--s2i-unsafe-color` | flag | Chroma-safe filtering toggle. |
| `--s2i-chroma-strength` | float | Chroma safety blend strength. |
| `--s2i-chroma-clip` | float | Max chroma deviation around 0.5 when safe-color is on. |

Image->sound (`i2s-*`):

| Option | Values | Notes |
|--------|--------|-------|
| `--i2s-mode` | `flat`, `hist`, `radial` | Impulse generation mode. |
| `--i2s-colorspace` | `luma`, `rgb-mean`, `rgb-stereo`, `ycbcr-mid-side` | Colorspace for impulse derivation. |
| `--i2s-pad-mode` | `same-center`, `same-first`, `full` | Convolution pad mode. |
| `--i2s-impulse-len` | `int`, `auto`, `frame` | `auto` matches audio chunk; `frame` uses one frame duration. |
| `--i2s-radius-mode` | `linear`, `log` | Radial binning mode (radial). |
| `--i2s-phase-mode` | `zero`, `random`, `image`, `min-phase`, `spiral` | Phase strategy (radial). |
| `--i2s-smoothing` | `none`, `hann` | Radial smoothing. |
| `--i2s-impulse-norm` | `energy`, `peak`, `none` | Impulse normalization. |
| `--i2s-out-norm` | `match_rms`, `match_peak`, `none` | Output normalization. |
| `--i2s-n-bins` | int | Histogram bins (hist mode). |

Bi-conv modes (serial/parallel):

| Mode                 | Image input        | Audio input        |
|----------------------|--------------------|--------------------|
| `parallel`           | original frame     | original chunk     |
| `serial-image-first` | sound-shaped frame | original chunk     |
| `serial-sound-first` | original frame     | image-shaped audio |

Audio length strategies:

| Option         | Behavior                                    |
|----------------|---------------------------------------------|
| `trim`         | Trim/pad end with zeros to video duration   |
| `pad-zero`     | Zero-pad tail (default)                     |
| `center-zero`  | Center audio and pad both sides with zeros  |
| `pad-loop`     | Loop audio to fill                          |
| `pad-noise`    | Pad tail with low-level noise               |

Outputs: processed video and audio files written to the given paths.

---

## Documentation

- **Design notes**: [`docs/design.md`](docs/design.md) — rationale, modes,
  color handling, cross-modal mapping.
- **API reference**: [`docs/api.md`](docs/api.md) — signatures and parameter
  semantics.
- **CLI + scripts**: [`docs/scripts.md`](docs/scripts.md) - batch subcommands and legacy helpers.
- **Changelog**: [`CHANGELOG.md`](CHANGELOG.md) — latest additions.

---

## License

See the [LICENSE](LICENSE) file in this repository.
