<!-- docs/api.md -->



# exconv API Reference



This document describes the public Python API of the `exconv` package.



At the top-level, `exconv` re-exports the main subpackages:



```python

import exconv



exconv.core      # low-level FFT/grid/norm utilities

exconv.io        # audio & image IO helpers

exconv.conv1d    # 1D convolution (audio)

exconv.conv2d    # 2D convolution (images & scalar fields)

exconv.dsp       # signal-processing utilities (envelopes, crossfades, normalization)

exconv.xmodal    # cross-modal tools, via submodules

exconv.mir       # beat/novelty/structure helpers for block segmentation

exconv.video_meta # ffprobe helpers for FPS metadata

exconv.__version__

```



---



## 1. Core utilities (`exconv.core`)



### 1.1 FFT helpers (`exconv.core.fft`)



#### `next_fast_len_ge`



```python

from exconv.core import next_fast_len_ge



m = next_fast_len_ge(n: int) -> int

```



| Param | Type | Description                           |
|-------|------|---------------------------------------|
| `n`   | int  | Minimum desired FFT length.           |



Returns a fast FFT length `m >= n`, using SciPy's `next_fast_len` if

available, then NumPy's, otherwise the next power of two.



---



#### `fftnd` / `ifftnd`



```python

from exconv.core import fftnd, ifftnd



X = fftnd(

    x: np.ndarray,

    axes: int | Sequence[int] | None = None,

    real_input: bool = False,

    n: Optional[Sequence[int]] = None,

) -> np.ndarray



y = ifftnd(

    X: np.ndarray,

    axes: int | Sequence[int] | None = None,

    real_output: bool = False,

    n: Optional[Sequence[int]] = None,

) -> np.ndarray

```



| Param          | Type                      | Description                                     |
|----------------|---------------------------|-------------------------------------------------|
| `axes`         | int / seq / None          | Axes to transform. `None`  all axes.          |
| `real_input`   | bool                      | If `True`, uses `rfftn` (realcomplex).        |
| `real_output`  | bool                      | If `True`, uses `irfftn` (complexreal).       |
| `n`            | seq of int or `None`      | FFT lengths per axis (in `axes` order).        |



These are thin wrappers over NumPy's FFT modules, adding consistent axis

normalization and optional real FFT paths.



---



#### `linear_freq_multiply`



```python

from exconv.core import linear_freq_multiply



y = linear_freq_multiply(

    x: np.ndarray,

    h: np.ndarray,

    axes: int | Sequence[int] | None = None,

    mode: str = "full",          # "full","same-first","same-center","circular"

    use_real_fft: bool = True,

) -> np.ndarray

```



| Param      | Type                    | Description                                      |
|------------|-------------------------|--------------------------------------------------|
| `x`        | ndarray                 | Signal.                                          |
| `h`        | ndarray                 | Kernel, broadcastable outside `axes`.           |
| `axes`     | int / seq / None        | Convolution axes (defaults to all).             |
| `mode`     | str                     | Size policy (see design doc).                   |
| `use_real_fft` | bool                | Use real FFT when both inputs are real.         |



- For `"circular"` no padding is applied and the size along each conv axis

  matches `x.shape[axis]`.

- For linear modes (`"full"`, `"same-first"`, `"same-center"`), the function

  pads, multiplies in the frequency domain and crops as specified.



---



#### `pad_to_linear_shape`



```python

from exconv.core import pad_to_linear_shape



x_padded, pads = pad_to_linear_shape(

    x: np.ndarray,

    kernel_shape: Sequence[int],

    axes: int | Sequence[int] | None = None,

)

```



Returns a padded copy of `x` so that a full linear convolution with a kernel

of `kernel_shape` along `axes` fits. `pads` gives per-axis `(before, after)`

pad widths for `np.pad`.



---



#### `make_hermitian_symmetric_unshifted`



```python

from exconv.core import make_hermitian_symmetric_unshifted



Xh = make_hermitian_symmetric_unshifted(

    F: np.ndarray,

    axes: int | Sequence[int] | None = None,

) -> np.ndarray

```



Enforces Hermitian symmetry on an *unshifted* complex spectrum `F`, such that

`np.fft.ifftn(Xh, axes=axes)` is (numerically) real. Used mainly in tests and

spectral experimentation.



---



### 1.2 Grids and windows (`exconv.core.grids`)



```python

from exconv.core import freq_grid_2d, radial_grid_2d, hann, tukey

```



- `freq_grid_2d(H, W) -> (k1, k2)`  

  Returns 2D integer frequency coordinates under `fftshift` convention

  (zero in the center, negative frequencies left/top).



- `radial_grid_2d(H, W, norm="unit") -> `  

  Radius grid in `[0,1]`, normalized by the farthest reachable bin corner.



- `hann(N) -> w`  

  Periodic Hann window of length `N`.



- `tukey(N, alpha=0.5) -> w`  

  Tukey (tapered cosine) window, implemented without SciPy.



---



### 1.3 Normalization helpers (`exconv.core.norms`)



```python

from exconv.core import rms_normalize, peak_normalize, apply_normalize

```



- `rms_normalize(x, target_rms=0.1)`  

  Scales `x` so that its global RMS equals `target_rms`.



- `peak_normalize(x, peak=0.99)`  

  Scales `x` so that `max(abs(x)) == peak`.



- `apply_normalize(x, mode)`  

  Dispatches based on `mode`:

  - `"rms"`  `rms_normalize`

  - `"peak"`  `peak_normalize`

  - `None` / `"none"`  no-op



Used by audio convolution routines to normalize only the final output.



---



### 1.4 DSP utilities (`exconv.dsp`)


```python
from exconv.dsp import (
    crossfade_weights,
    apply_adsr,
    normalize_range_01,
    normalize_impulse,
    normalize_to_reference,
    slice_for_frame,
    frame_range_samples,
    match_audio_length,
    audio_chunk_for_interval,
    hann,
    tukey,
)
```

Selected helpers:

- `crossfade_weights(n, mode)` returns `(w_prev, w_cur)` for a crossfade:
  - `mode="lin"`: linear amplitude fade, `w_prev = 1 - t`, `w_cur = t`.
  - `mode="equal"`: equal-energy fade, `w = sqrt(t)`.
  - `mode="power"`: equal-power fade, `w_prev = cos(t*pi/2)`, `w_cur = sin(t*pi/2)`.

- `apply_adsr(x, sr, attack_s, decay_s, sustain, release_s, curve)`
  applies an ADSR envelope to a signal (used for per-block audio shaping).

- Normalization:
  - `normalize_impulse(x, mode)` with `mode` in `{"none","peak","energy"}`.
  - `normalize_to_reference(x, ref, mode)` with `mode` in `{"none","match_peak","match_rms"}`.
  - `normalize_range_01`, `clip_01`, `normalize_chroma_midpoint` for images.

- Segment helpers:
  - `slice_for_frame(frame_idx, fps, sr, n_samples)`
  - `frame_range_samples(start_frame, end_frame, sr, fps, n_samples)`
  - `match_audio_length(audio, target_len, mode)`
  - `audio_chunk_for_interval(audio, start, stop, mode, ...)`

- Window helpers:
  - `hann(n)`, `tukey(n, alpha=0.5)`


---


### 1.5 Video metadata helpers (`exconv.video_meta`)


```python
from exconv.video_meta import (
    ffprobe_available,
    ffprobe_fps_info,
    parse_float,
    parse_fraction,
)
```

- `ffprobe_available()` reports whether `ffprobe` is available on PATH.
- `ffprobe_fps_info(path)` returns `(avg_frame_rate, r_frame_rate, duration)`.
- `parse_float` and `parse_fraction` handle ffprobe-style numeric fields.

Used by video IO to guard FPS metadata and resolve missing/invalid values.


---


## 2. IO utilities (`exconv.io`)



High-level IO facade re-exports audio and image helpers:



```python

from exconv.io import (

    read_audio, read_segment, write_audio, to_mono, to_stereo,

    read_image, write_image, as_float32, as_uint8, rgb_to_luma, luma_to_rgb, upscale_image,

)

```



### 2.1 Audio IO (`exconv.io.audio`)



- `read_audio(path, dtype="float32", always_2d=False) -> (data, sr)`  

  Reads an audio file via `soundfile`. Returns `(N,)` mono or `(N, C)` if

  `always_2d` is `True` or multi-channel.



- `read_segment(path, *, start=None, stop=None, unit="seconds", dtype="float32", always_2d=False)`  

  Reads only a segment of the file without loading it all.  

  `unit="seconds"` or `"samples"`. Out-of-range indices are clamped.



- `write_audio(path, data, sr, subtype="PCM_16", clip=None, dtype=None)`  

  Writes `(N,)` or `(N, C)` audio via `soundfile`. Handles dtype casting and

  optional clipping (default: clip floats for integer PCM subtypes).



- `to_mono(x)`  

  Converts `(N,)` or `(N, C)` to `(N,)` by averaging channels.



- `to_stereo(x)`  

  - `(N,)` / `(N,1)`  duplicated `(N,2)`  

  - `(N,2)`  unchanged  

  - `(N,C>2)`  first two channels only.



---



### 2.2 Image IO (`exconv.io.image`)



- `read_image(path, mode="RGB", dtype="uint8") -> np.ndarray`  

  Reads an image via Pillow. `mode="keep"` preserves native mode.



- `write_image(path, data, mode=None)`  

  Saves 2D or 3D arrays as images, auto-selecting Pillow mode based on shape

  and converting to `uint8` via `as_uint8`.



- `as_float32(x)`  

  Reasonable float32 normalization based on dtype (see design doc).



- `as_uint8(x)`  

  Converts any numeric array into `uint8` with sensible scaling/clipping.



- `rgb_to_luma(x)`  

  RGB(A)  luma using Rec.709 coefficients.



- `luma_to_rgb(x, dtype=None)`  

  2D luma  `(H, W, 3)` RGB by channel replication.


- `upscale_image(x, scale=1.0, method="lanczos", model=None)`  

  Resize/upscale via Pillow resampling or OpenCV DNN super-res; returns uint8.
  OpenCV backends (opencv-*) require a .pb model and opencv-contrib-python (installed with exconv).



---



## 3. 1D audio convolution (`exconv.conv1d`)



```python

from exconv.conv1d import (
    Audio,
    AudioConvolutionResult,
    auto_convolve,
    pair_convolve,
    multi_convolve,
    convolution_family,
)

```



### 3.1 `Audio` container



```python

from dataclasses import dataclass

import np as numpy



@dataclass

class Audio:

    samples: np.ndarray   # (N,) mono or (N, C)

    sr: int               # Hz



    @property

    def n_samples(self) -> int: ...

    @property

    def n_channels(self) -> int: ...

    @property

    def is_mono(self) -> bool: ...

    def copy(self) -> "Audio": ...

```



The class is intentionally minimal: just data + convenience properties.



---



### 3.2 `auto_convolve`



```python

from exconv.conv1d import auto_convolve



out = auto_convolve(

    audio: Audio,

    mode: str = "same-center",   # "full","same-first","same-center"

    circular: bool = False,

    normalize: Optional[str] = "rms",   # "rms","peak",None,"none"

    order: int = 2,

) -> Audio

```



| Param       | Description                                      |
|------------|--------------------------------------------------|
| `audio`    | Input audio object.                              |
| `mode`     | Linear size policy when `circular=False`.        |
| `circular` | If `True`, perform circular convolution.         |
| `normalize`| Output normalization applied once at the end.    |
| `order`    | n-th order self-conv: 1  copy, 2  x*x, 3  x*x*x |



Channel semantics: if `samples` is multi-channel, channels are convolved

independently if their counts match, otherwise both inputs are downmixed to

mono for the convolution.



---



### 3.3 `pair_convolve`



```python

from exconv.conv1d import pair_convolve



out = pair_convolve(

    x: Audio,

    h: Audio,

    mode: str = "same-center",

    circular: bool = False,

    normalize: Optional[str] = "rms",

) -> Audio

```



| Param     | Description                                      |
|-----------|--------------------------------------------------|
| `x`       | Reference audio; defines output length in `"same-*"` and circular modes. |
| `h`       | Kernel audio; must share the same sample rate.   |
| `mode`    | Linear size policy if not `circular`.            |
| `circular`| Circular vs linear convolution.                  |
| `normalize`| Output normalization.                           |



Raises `ValueError` if sample rates differ.



---



### 3.4 `multi_convolve`



```python

from exconv.conv1d import multi_convolve



out = multi_convolve(

    audios: list[Audio],

    mode: str = "same-center",

    circular: bool = False,

    normalize: Optional[str] = "rms",

) -> Audio

```



N-fold 1D convolution of all input signals. In the frequency domain this is
equivalent to multiplying every input spectrum at a common FFT length and
taking one inverse FFT.



| Param | Description |
|-------|-------------|
| `audios` | Input audio objects. Must contain at least one item and all sample rates must match. |
| `mode` | Linear size policy if `circular=False`. |
| `circular` | If `True`, use circular convolution with the first audio length. |
| `normalize` | Output normalization applied once at the end. |



Length behavior:

- `mode="full"`: output length is `sum(lengths) - (len(audios) - 1)`.
- `mode="same-first"`: first `audios[0].n_samples` samples of the full result.
- `mode="same-center"`: centered crop of length `audios[0].n_samples`.
- `circular=True`: output length is `audios[0].n_samples`.



Channel semantics match `pair_convolve`: equal channel counts are processed
per-channel; mixed mono/stereo/multi-channel inputs are downmixed to mono.



---



### 3.5 `convolution_family`



```python

from exconv.conv1d import convolution_family



results = convolution_family(

    audios: list[Audio],

    names: Optional[list[str]] = None,

    *,

    mode: str = "same-center",

    circular: bool = False,

    normalize: Optional[str] = "rms",

    self_order: int = 2,

    include_self: bool = True,

    include_pairs: bool = True,

    include_multi: bool = True,

) -> list[AudioConvolutionResult]

```



Generates a stable family of related audio convolutions: self-convolutions,
unordered pair-convolutions, and one all-input N-fold convolution.



| Param | Description |
|-------|-------------|
| `audios` | Input audio objects. |
| `names` | Optional labels aligned with `audios`; defaults to string indices. |
| `mode` | Linear size policy for produced convolutions. |
| `circular` | Use circular convolution for produced convolutions. |
| `normalize` | Output normalization applied to each result. |
| `self_order` | Order for self-convolution outputs. |
| `include_self` | Include one self-convolution per input. |
| `include_pairs` | Include every unordered pair. |
| `include_multi` | Include one all-input N-fold result when there are at least two inputs. |



`AudioConvolutionResult` fields:

| Field | Description |
|-------|-------------|
| `kind` | `"self"`, `"pair"`, or `"multi"`. |
| `indices` | Input indices used by this result. |
| `names` | Input labels used by this result. |
| `audio` | The output `Audio`. |



---



## 4. 2D image convolution (`exconv.conv2d`)



Top-level exports:



```python

from exconv.conv2d import (

    image_auto_convolve, image_pair_convolve,

    gaussian_1d, gaussian_2d, gaussian_separable,

    laplacian_3x3, gabor_kernel, separable_conv2d,

    apply_gamma,

)

```



### 4.1 Image auto/pair convolution (`exconv.conv2d.image`)



#### `image_auto_convolve`



```python

out = image_auto_convolve(

    img: np.ndarray,

    mode: Literal["full","same-first","same-center"] = "same-center",

    circular: bool = False,

    colorspace: Literal["luma","channels"] = "luma",

    normalize: str = "clip",        # "clip","rescale"

    gamma=None,

) -> np.ndarray

```



Self-convolves `img` with itself; a wrapper over `image_pair_convolve(img, img, ...)`.



#### `image_pair_convolve`



```python

out = image_pair_convolve(

    img: np.ndarray,

    kernel: np.ndarray,

    mode: Literal["full","same-first","same-center"] = "same-center",

    circular: bool = False,

    colorspace: Literal["luma","channels"] = "luma",

    normalize: str = "clip",        # "clip","rescale"

    gamma=None,

) -> np.ndarray

```



- `img`: 2D or 3D array (single-channel or color).

- `kernel`: 2D or 3D (optional per-channel kernels).

- `normalize`:

  - `"clip"`: clip to inputs numeric range.

  - `"rescale"`: linearly rescale output min/max to input min/max.

- `gamma`: optional gamma correction applied on the final image in per-channel

  mode.



---



### 4.2 Kernels (`exconv.conv2d.kernels`)



- `gaussian_1d(sigma, radius=None, truncate=3.0, normalize=True)`  `(K,)`

- `gaussian_2d(sigma, radius=None, truncate=3.0, normalize=True)`  `(K, K)`

- `gaussian_separable(sigma, radius=None, truncate=3.0, normalize=True)`  `(k_row, k_col)`

- `laplacian_3x3(center_weight=-4.0, eight_connected=False)`  `(3,3)` kernel

- `gabor_kernel(ksize, sigma, theta, lambd, gamma=0.5, psi=0.0, normalize=False)`  `(ksize, ksize)` Gabor

- `separable_conv2d(img, k_row, k_col=None)`  

  Separable spatial conv using 1D reference convolutions with `"same-center"`

  cropping; output has same shape as `img`.



---



## 5. Cross-modal sound->image (`exconv.xmodal.sound2image`)


### 5.1 Sound -> image (`exconv.xmodal.sound2image`)



```python

from exconv.xmodal.sound2image import spectral_sculpt



out = spectral_sculpt(

    image: np.ndarray,

    audio: np.ndarray,

    sr: int,

    *,

    mode: Literal["mono","stereo","mid-side"] = "mono",

    colorspace: Literal["luma","color"] = "luma",

    normalize: bool = True,

    safe_color: bool = True,

    chroma_strength: float = 0.5,

    chroma_clip: float = 0.25,

) -> np.ndarray

```



| Param       | Description                                        |
|------------|----------------------------------------------------|
| `image`    | 2D or 3D image (float or integer).                 |
| `audio`    | 1D mono or 2D multi-channel array.                 |
| `sr`       | Sample rate (currently unused, reserved).          |
| `mode`     | How audio channels map to radial filters.          |
| `colorspace`| `"luma"` (luminance only) or `"color"` (YCbCr).   |
| `normalize`| If `True`, normalizes/clips the final output. |
| `safe_color` | In color mode, normalize Y and compress chroma before RGB conversion. |
| `chroma_strength` | Blend from original chroma (`0.0`) to fully filtered chroma (`1.0`). |
| `chroma_clip` | Maximum chroma deviation around `0.5` when `safe_color=True`. |



See `docs/design.md` for the complete mapping of spectra to Y/Cb/Cr channels.



---


### 5.2 Image -> sound (`exconv.xmodal.image2sound`)


```python
from exconv.xmodal.image2sound import (
    image2sound_flat,
    image2sound_hist,
    image2sound_radial,
)
```

These build a 1D impulse from an image and convolve it with audio:

- `image2sound_flat(...)` flattens grayscale values into an impulse.
- `image2sound_hist(...)` uses a grayscale histogram as the impulse.
- `image2sound_radial(...)` builds an impulse from a radial FFT magnitude
  profile, with selectable phase and smoothing options.

Key parameters (shared):

- `pad_mode`: `"full"`, `"same-first"`, `"same-center"` (size policy).
- `colorspace`: `luma`, `rgb-mean`, `rgb-stereo`, `ycbcr-mid-side`.
- `impulse_norm`: `none`, `peak`, `energy`.
- `out_norm`: `none`, `match_peak`, `match_rms`.

For the detailed math and phase/radius options, see `docs/design.md` section 6.3.


---


### 5.3 Video cross-modal (`exconv.xmodal.video`)


```python
from exconv.xmodal.video import (
    sound2image_video_arrays,
    sound2image_video_from_files,
    biconv_video_arrays,
    biconv_video_from_files,
    biconv_video_to_files_stream,
)
```

Key notes:

- `sound2image_video_*` applies `spectral_sculpt` per-frame and builds output
  audio either via per-buffer auto-convolution or by reusing the original.
- `biconv_video_*` runs bi-directional cross-modal processing on blocks,
  with optional block ADSR and crossovers.
- For block strategy, crossover, and serial chaining semantics, see
  `docs/design.md` section 6.4.


---


## 6. Music information retrieval (`exconv.mir`)


```python
from exconv.mir.blocks import (
    audio_to_frame_blocks,
    spectral_flux_novelty,
    structure_novelty,
    estimate_tempo,
)
```

These helpers derive beat/novelty/structure timing from audio and are used by
video block strategies (`beats`, `novelty`, `structure`). See `docs/design.md`
section 6.4 for how these map to video frames and block boundaries.


---



## 7. Command-line interface (`exconv.cli`)



The CLI entry point is `exconv.cli.exconv_cli:main`, registered as `exconv`

in `pyproject.toml`.

Settings files (shared):
  - `--settings <path>` loads defaults for the command from JSON or CSV.
  - `--save-settings <path>` writes current option values (non-positional) for the command.
  - `--show-settings` prints the resolved option values as JSON and exits.
  - `--save-settings <path> --update-settings` updates only the provided option values in-place and exits.
  - JSON can store per-command blocks keyed by command name; CSV is flat `key,value`.
  - CLI flags override loaded settings.



### 7.1 Top-level



```bash

exconv --version

exconv --help

```



### 7.2 `audio-auto`



```bash

exconv audio-auto     --in input.wav     --out out.wav     --mode same-center     --order 2     --normalize rms

```



Maps directly to `conv1d.auto_convolve`.



### 7.3 `img-auto`



```bash

exconv img-auto     --in img.png     --out out.png     --mode same-center     --colorspace channels     --normalize rescale     --kernel "gaussian:sigma=2.0"

```



Uses `image_auto_convolve` or `image_pair_convolve` with Gaussian kernels

parsed by `_parse_gaussian_kernel_spec`.

Optional upscaling: `--upscale <factor>` with `--upscale-method`
(`lanczos`, `bicubic`, or `opencv-*`). OpenCV methods require
`--upscale-model` (opencv-contrib-python ships with exconv).



### 7.4 `sound2image`



```bash

exconv sound2image     --img img.png     --audio audio.wav     --out sculpted.png     --colorspace luma

```



Provides a CLI facade over `spectral_sculpt` for quick experiments.
Optional upscaling: `--upscale <factor>` with `--upscale-method`
(`lanczos`, `bicubic`, or `opencv-*`). OpenCV methods require
`--upscale-model` (opencv-contrib-python ships with exconv).
See `docs/design.md` sections 6.1-6.2 for the sound->image mapping details.

### 7.5 `video-biconv`

```bash
exconv video-biconv     --video input.mp4     --out-video out_biconv.mp4     --serial-mode parallel     --audio-length-mode pad-zero     --i2s-mode radial     --i2s-impulse-len auto
```

Key notes:

- `--audio` is optional; if omitted, audio is extracted from the input video (ffmpeg required). If `--audio` points to a video file, its audio track is auto-extracted.
- `--serial-mode`: `parallel`, `serial-image-first`, `serial-sound-first`.
- `--audio-length-mode`: `trim`, `pad-zero`, `pad-loop`, `pad-noise`, `center-zero`.
- `--preview-seconds`: limit processing to the first N seconds (quick test run).
- `--block-strategy`: `fixed` uses frame counts; `beats`/`novelty`/`structure` analyze audio to set block boundaries.
- `--block-size`: process frames in blocks (e.g., 12/24/50/120/240). The block time range defines the audio chunk reused for sound->image; image->sound uses the block mean image (or mean of processed frames in `serial-image-first`) with that chunk.
- `--crossover`: optional block boundary crossfade for both frames and audio.
  - `none`: hard boundary, no blending.
  - `lin`: linear amplitude fade (`w_prev = 1 - t`, `w_cur = t`).
  - `equal`: equal-energy fade (`w = sqrt(t)`).
  - `power`: equal-power fade (`w_prev = cos(t*pi/2)`, `w_cur = sin(t*pi/2)`).
- `--crossover-frames`: number of frames on each side used for blending; the
  overlap is clamped to the lengths of adjacent blocks.
- For a deeper explanation of serial chaining and block parameters, see `docs/design.md` (section 6.4).
- Sound->image options mirror `spectral_sculpt` (`--s2i-mode` mono/stereo/mid-side, `--s2i-colorspace` luma/color).
- Image->sound options mirror the image2sound demo (`--i2s-mode` flat/hist/radial with colorspace/phase/padding/length controls).
Optional upscaling: `--upscale <factor>` with `--upscale-method`
(`lanczos`, `bicubic`, or `opencv-*`). OpenCV methods require
`--upscale-model` (opencv-contrib-python ships with exconv).
See `docs/design.md` section 6.3 for image->sound details and section 6.4 for block behavior.

### 7.6 `folderbatch`

```bash
exconv folderbatch my_project --root samples --audio-mode same-center --audio-order 2 --audio-normalize rms --audio-multi-circular
```

Batch audio self, pair and all-input N-fold convolution plus optional
sound->image outputs across a project folder. See `docs/scripts.md` for the
complete grouped flag list and the expected `samples/` folder layout.

Audio outputs:

| Output folder | Contents |
|---------------|----------|
| `output/audio/<project>/self/` | One self-convolution per input. |
| `output/audio/<project>/pair/` | One output per unordered input pair. |
| `output/audio/<project>/multi/` | One all-files N-fold convolution. |

Audio options:

| Option | Description |
|--------|-------------|
| `--audio-mode` | Linear convolution size policy: `full`, `same-first`, `same-center`. |
| `--audio-order` | Self-convolution order. |
| `--audio-circular` | Use circular convolution for all audio outputs. |
| `--audio-multi-circular` | Use circular convolution only for the all-files N-fold output. |
| `--audio-normalize` | `rms`, `peak`, or `none`. |
| `--audio-subtype` | libsndfile subtype for WAV/FLAC outputs. |

Use `--s2i-animate` to emit per-audio animations into the `animations/`
subfolder (defaults to mp4 with audio unless `--s2i-animate-no-audio`).
For sound->image parameter semantics, see `docs/design.md` sections 6.1-6.2.

### 7.7 `video-folderbatch`

```bash
exconv video-folderbatch my_project --jobs 2 --suffix _biconv --serial-mode parallel
```

Batch-run `video-biconv` over every video in a project folder, with optional
audio-driven block segmentation and per-block crossfades. Use
`--preview-seconds` for quick test runs. See `docs/scripts.md`
for the full set of batch flags.
For block strategy and crossover details, see `docs/design.md` section 6.4.

### 7.8 `animate`

```bash
exconv animate frames_dir out.mp4 --fps 12
```

Create a GIF/MP4 from a directory of images. Use `--pattern` to filter frames
and `--audio` to mux a soundtrack into MP4 output (ffmpeg required).

---

## 8. Image demo CLI (`exconv-image`)


The legacy image demo is exposed as a separate entry point:



```bash

exconv-image INPUT_PATH OUTPUT_DIR [--kernel_path KERNEL] [--mode ...]

```



Implemented in `exconv.cli.image_demo`, it runs auto-convolution and optional

pair-convolution on a single image and saves JPEG outputs.



---

