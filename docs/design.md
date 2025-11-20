<!-- docs/design.md -->

# exconv Design Notes

Experimental convolution toolkit for sound, images and cross-modal synthesis.

This document captures the design rationale and main conventions behind the
`exconv` library: how we treat linear vs circular convolution, padding, color
handling for images, and audio→image spectral mapping.

---

## 1. Design goals

- **One mental model across domains**  
  Audio (1D), images (2D) and cross-modal mappings should feel like variants
  of the same operation: “take two signals, multiply spectra, come back”.  
  The FFT layer (`exconv.core.fft`) is shared for all domains.

- **Explicit size policies**  
  Instead of hiding padding and cropping, the user picks a *mode*:
  `"full"`, `"same-first"`, `"same-center"` or `"circular"`.  
  The same names mean the same thing for 1D and 2D API.

- **Minimal but composable high-level APIs**  
  - 1D audio: `Audio` + `auto_convolve` / `pair_convolve`.  
  - 2D images: `image_auto_convolve` / `image_pair_convolve` with color modes.  
  - Cross-modal: `spectral_sculpt(image, audio, ...)`.  

- **IO done once, correctly**  
  Audio/image IO is centrally handled in `exconv.io` (dtype, clipping,
  stereo/mono, RGB↔luma, etc.), so convolution code can assume clean
  ndarrays.

---

## 2. Signals and shapes

All core functions operate on NumPy arrays:

- Audio: `(N,)` mono or `(N, C)` channels, time on axis 0.  
- Images: 2D `(H, W)` or 3D `(H, W, C)` for color.  
- FFT axes are explicit (e.g. `(0,)` for time, `(0,1)` for 2D spatial).

Because the low-level FFT utilities (`fftnd`, `ifftnd`, `linear_freq_multiply`)
are shape-generic, adding e.g. 3D volumetric convolution is conceptually
straightforward.

---

## 3. Linear vs circular convolution

### 3.1 Linear convolution via FFT

For linear convolution we use the classic pipeline: pad → FFT → multiply →
IFFT → crop.

```txt
x  ----->  FFT  -----
                   |            IFFT  ----->  crop  ---> y (linear)
h  ----->  FFT  ----  *-mul-*  -----
```

FFT sizes are chosen per-axis using `_fft_shapes_for_linear`, which in turn
uses `next_fast_len_ge` (SciPy’s `next_fast_len` if available, otherwise
NumPy or power-of-two fallback).

#### Size modes

Given `x` and `h` on some axes:

- `full` – the mathematically full linear convolution

  ```txt
  length(full) = len(x) + len(h) - 1
  ```

- `same-first` – first `len(x)` samples/pixels (like NumPy’s `"same"` but
  explicit that the reference is the *first* argument).

- `same-center` – center crop: the output is the same length as `x`, but
  aligned so that the kernel’s “center of mass” stays in the middle of the
  window. Used for blur-like effects. Cropping logic is shared between 1D/2D.

ASCII sketch for a 1D case:

```txt
full:       [...............]  (N + K - 1)
same-first: [.........        ] (N)
same-center:[   .........     ] (N)
```

These modes are passed directly to `linear_freq_multiply` and surfaced in the
public APIs (`audio_auto_convolve`, `image_auto_convolve`, etc.).

### 3.2 Circular convolution

Circular convolution is used when we do **no padding**: we FFT, multiply and
IFFT at the original signal length along each conv axis.

```txt
x (N) ---- FFT_N -----
                      |           IFFT_N ----> y (N) circular
h (N) ---- FFT_N ----  *-mul-*  -----
```

All high-level APIs expose a `circular: bool` flag:

- For 1D audio, circular keeps `Audio.n_samples` unchanged.  
- For images, circular keeps `(H, W)` unchanged, creating wrap-around artifacts
  instead of black padding borders.

---

## 4. Padding strategy

Two layers handle padding:

1. `pad_to_linear_shape(x, kernel_shape, axes=...)` – pads `x` at the **end**
   along each conv axis so that a full linear convolution fits. Returns both
   the padded array and the pad widths, so you can pad other tensors
   consistently.

2. `_fft_shapes_for_linear(sig_shape, ker_shape)` – picks FFT sizes that are
   fast and at least `sig + ker - 1` along each axis.

High-level modules (audio/image) usually **don’t expose this** directly; they
instead offer the `mode` and `circular` knobs. For more exotic pipelines
(e.g. multi-stage linear conv), you can call `pad_to_linear_shape` yourself.

---

## 5. Color handling (images)

Color handling is split into:

- **IO/representation** – `read_image`, `write_image`, conversions between
  `uint8` and float, RGB↔luma.  
- **Convolution decisions** – `colorspace` and optional `gamma` handling
  inside `exconv.conv2d.image`.

### 5.1 IO conventions

- `read_image(path, mode="RGB", dtype="uint8")` returns an ndarray:
  - `"L"` → 2D
  - `"RGB"` / `"RGBA"` → 3D  
- `as_float32` normalizes:
  - `uint8` → `[0, 1]`
  - integer dtypes → `[0, 1]` or `[-1, 1]` depending on sign
  - floats → cast only  
- `as_uint8` converts any numeric image to `uint8`, scaling/clipping as needed,
  for deterministic saving.

We intentionally *don’t* assume any particular float range in conv code: it
just uses whatever range the caller chooses (often 0..1 for images). IO
helpers are where we compress to 8-bit.

### 5.2 Luma vs per-channel convolution

The main high-level entry is `conv2d.image_pair_convolve`:

```python
pair_convolve(
    img,
    kernel,
    mode="same-center",
    circular=False,
    colorspace="luma",    # or "channels"
    normalize="clip",     # or "rescale"
    gamma=None,
)
```

- `colorspace="luma"`:
  - `img` → luminance via Rec.709 coefficients.  
  - `kernel` → reduced to a 2D luma kernel (if 3D, we also luminance-reduce).  
  - We convolve once in 2D and return a 2D result.
  - `image_auto_convolve` simply uses `img` as the `kernel`.

- `colorspace="channels"`:
  - If `img` is 2D → treat as single channel.
  - If `img` is 3D with `C` channels:
    - 2D `kernel` → same kernel for each channel.
    - 3D `kernel` with `kernel.shape[2] == C` → per-channel kernels.

Normalization after convolution is driven by the *input* dtype: we either
clip or rescale into that numeric range. For integer input we use dtype
limits, for floats we derive min/max from the input.

#### Gamma correction

We provide an `apply_gamma` helper in `conv2d.color` and expose it via
`pair_convolve(..., gamma=...)` in the per-channel path.

```txt
img  →  conv2d (per-channel)  →  normalize → [optional gamma] → out
```

Gamma is applied *after* convolution, on the final 3-channel image.

---

## 6. Cross-modal mapping: sound → image

Cross-modal processing lives in `exconv.xmodal.sound2image`. The goal is to
“sculpt” an image using spectral information from an audio signal, in a way
that can capture mono, stereo and mid-side relations.

The main public entry is:

```python
spectral_sculpt(
    image: np.ndarray,
    audio: np.ndarray,
    sr: int,
    *,
    mode: {"mono", "stereo", "mid-side"} = "mono",
    colorspace: {"luma", "color"} = "luma",
    normalize: bool = True,
) -> np.ndarray
```

### 6.1 Audio → spectra

Steps (conceptually):

1. **Channel prep** via `_prepare_audio_channels`:

   - Start from mono `(N,)` or multi-channel `(N, C)` audio.
   - Convert to stereo deterministically with `to_stereo` if needed, so that
     `L` and `R` always exist.
   - Derive:
     - `mono = (L + R) / 2`
     - `mid  = (L + R) / 2`
     - `sideL = L - mid`, `sideR = R - mid`.

2. **Spectra** via `_rfft_magnitude`:
   - 1D real FFT (`rfft`) of each signal → magnitude → normalized to max=1.  
   - Silent input → return an all-ones curve (identity filter).

3. **Radial filters** via `_radial_filter_from_curve`:
   - Construct a 2D radius grid `ρ` in `[0,1]` using `radial_grid_2d(H, W)`.  
   - Map `curve` indices along `ρ`:

     ```txt
     idx = int(ρ * (F - 1))
     H2 = curve[idx]  # shape (H, W)
     ```

   - This yields isotropic “ring patterns” that encode the audio spectrum.

ASCII sketch of `ρ`:

```txt
ρ = 0      0.25     0.5     0.75      1.0
  +-------------------------------------→
  |        •         •        •
  |   •                         •
  |        •         •        •
  v
```

### 6.2 Image path

Depending on `colorspace`:

#### `colorspace="luma"`

- If `image` is RGB, we compute luminance via `rgb_to_luma`; for 2D images we
  use them as-is.  
- We choose which spectrum drives the luma filter:

  | `mode`     | filter used for luma |
  |-----------|----------------------|
  | `"mono"`  | spectrum of `mono`   |
  | `"stereo"`| spectrum of `mono`   |
  | `"mid-side"` | spectrum of `mid` |

- Luma slice is filtered with `_fft_filter_apply`:

  ```txt
  FFT2(luma) → * H_luma → IFFT2 → y_luma
  ```

- If original image was color, we expand `y_luma` back to RGB by channel
  replication via `luma_to_rgb`.

Normalization:

- If `normalize=True`, we globally rescale to `[0,1]` via `_normalize_01_global`.

#### `colorspace="color"`

- We first ensure a 3-channel RGB representation (2D → replicated RGB).  
- Convert RGB ↔ YCbCr using `_rgb_to_ycbcr` / `_ycbcr_to_rgb`.

  ```txt
  RGB → Y, Cb, Cr
  ```

- Filters are assigned as:

  | `mode`     | Y filter        | Cb filter      | Cr filter      |
  |-----------|-----------------|----------------|----------------|
  | `"mono"`  | `H_mono`        | identity       | identity       |
  | `"stereo"`| `H_mono`        | `H_L`          | `H_R`          |
  | `"mid-side"` | `H_mid`     | `H_sideL`      | `H_sideR`      |

- We recombine to YCbCr, clip softly, convert back to RGB and (optionally)
  clip to `[0,1]`.

This design keeps all the interesting stereo logic in the 2D filter
construction, while the image path is just “apply this scalar filter to
chosen channels”.

---

## 7. CLI and library boundary

The CLI (`exconv.cli.exconv_cli`) is deliberately thin: it wires argparse
to the high-level APIs:

- `audio-auto` → `conv1d.audio.auto_convolve`
- `img-auto` → `conv2d.image_auto_convolve` / `pair_convolve` with Gaussian
  kernels parsed from strings.
- `sound2image` → `xmodal.sound2image.spectral_sculpt` (sound→image demo).

The design intention is:

- New research experiments should first be prototyped in Python using the
  library API.
- Once they stabilize, a small CLI façade can be added for quick runs and
  batch experiments.

---
