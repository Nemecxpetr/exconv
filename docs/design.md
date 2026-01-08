<!-- docs/design.md -->



# exconv Design Notes



Experimental convolution toolkit for sound, images and cross-modal synthesis.



This document captures the design rationale and main conventions behind the

`exconv` library: how we treat linear vs circular convolution, padding, color

handling for images, and audioimage spectral mapping.



---



## 1. Design goals



- **One mental model across domains**  

  Audio (1D), images (2D) and cross-modal mappings should feel like variants

  of the same operation: take two signals, multiply spectra, come back.  

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

  stereo/mono, RGBluma, etc.), so convolution code can assume clean

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



For linear convolution we use the classic pipeline: pad  FFT  multiply 

IFFT  crop.



```txt

x  ----->  FFT  -----

                   |            IFFT  ----->  crop  ---> y (linear)

h  ----->  FFT  ----  *-mul-*  -----

```



FFT sizes are chosen per-axis using `_fft_shapes_for_linear`, which in turn

uses `next_fast_len_ge` (SciPys `next_fast_len` if available, otherwise

NumPy or power-of-two fallback).



#### Size modes



Given `x` and `h` on some axes:



- `full`  the mathematically full linear convolution  
  $\text{length(full)} = \mathrm{len}(x) + \mathrm{len}(h) - 1$



- `same-first`  first `len(x)` samples/pixels (like NumPys `"same"` but

  explicit that the reference is the *first* argument).



- `same-center`  center crop: the output is the same length as `x`, but

  aligned so that the kernels center of mass stays in the middle of the

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



1. `pad_to_linear_shape(x, kernel_shape, axes=...)`  pads `x` at the **end**

   along each conv axis so that a full linear convolution fits. Returns both

   the padded array and the pad widths, so you can pad other tensors

   consistently.



2. `_fft_shapes_for_linear(sig_shape, ker_shape)`  picks FFT sizes that are

   fast and at least `sig + ker - 1` along each axis.



High-level modules (audio/image) usually **dont expose this** directly; they

instead offer the `mode` and `circular` knobs. For more exotic pipelines

(e.g. multi-stage linear conv), you can call `pad_to_linear_shape` yourself.



---



## 5. Color handling (images)



Color handling is split into:



- **IO/representation**  `read_image`, `write_image`, conversions between

  `uint8` and float, RGBluma.  

- **Convolution decisions**  `colorspace` and optional `gamma` handling

  inside `exconv.conv2d.image`.



### 5.1 IO conventions



- `read_image(path, mode="RGB", dtype="uint8")` returns an ndarray:

  - `"L"`  2D

  - `"RGB"` / `"RGBA"`  3D  

- `as_float32` normalizes:

  - `uint8`  `[0, 1]`

  - integer dtypes  `[0, 1]` or `[-1, 1]` depending on sign

  - floats  cast only  

- `as_uint8` converts any numeric image to `uint8`, scaling/clipping as needed,

  for deterministic saving.



We intentionally *dont* assume any particular float range in conv code: it

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

  - `img`  luminance via Rec.709 coefficients.  

  - `kernel`  reduced to a 2D luma kernel (if 3D, we also luminance-reduce).  

  - We convolve once in 2D and return a 2D result.

  - `image_auto_convolve` simply uses `img` as the `kernel`.



- `colorspace="channels"`:

  - If `img` is 2D  treat as single channel.

  - If `img` is 3D with `C` channels:

    - 2D `kernel`  same kernel for each channel.

    - 3D `kernel` with `kernel.shape[2] == C`  per-channel kernels.



Normalization after convolution is driven by the *input* dtype: we either

clip or rescale into that numeric range. For integer input we use dtype

limits, for floats we derive min/max from the input.



#### Gamma correction



We provide an `apply_gamma` helper in `conv2d.color` and expose it via

`pair_convolve(..., gamma=...)` in the per-channel path.



```txt

img    conv2d (per-channel)    normalize  [optional gamma]  out

```



Gamma is applied *after* convolution, on the final 3-channel image.



---



## 6. Cross-modal mapping: sound -> image


Cross-modal processing lives in `exconv.xmodal.sound2image`. The goal is to

sculpt an image using spectral information from an audio signal, in a way

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



### 6.1 Audio  spectra



Steps (conceptually):



1. **Channel prep** via `_prepare_audio_channels`:



   - Start from mono `(N,)` or multi-channel `(N, C)` audio.

   - Convert to stereo deterministically with `to_stereo` if needed, so that

     `L` and `R` always exist.

   - Derive: $\text{mono} = (L + R) / 2$, $\text{mid} = (L + R) / 2$,
     $\text{sideL} = (L - R) / 2$, $\text{sideR} = (R - L) / 2$.


2. **Spectra** via `_rfft_magnitude`:

   - 1D real FFT (`rfft`) of each signal  magnitude  normalized to max=1:
     $S[k] = \frac{|\mathrm{rfft}(x)[k]|}{\max_k |\mathrm{rfft}(x)[k]|}$.
   - Silent input  return an all-ones curve (identity filter).



3. **Radial filters** via `_radial_filter_from_curve`:
   - Construct a 2D radius grid `rho` in `[0,1]` using `radial_grid_2d(H, W)`.  
   - Map `curve` indices along `rho`:
     $\text{idx} = \lfloor \rho (F - 1) \rfloor$,
     $H_2 = \text{curve}[\text{idx}]$.
   - This yields isotropic ring patterns that encode the audio spectrum.



ASCII sketch of `rho`:


```txt

 = 0      0.25     0.5     0.75      1.0

  +-------------------------------------

  |                         

  |                            

  |                         

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
  $y_{\text{luma}} = \mathcal{F}^{-1}\left(\mathcal{F}(\text{luma}) \cdot H_{\text{luma}}\right)$.


- If original image was color, we expand `y_luma` back to RGB by channel

  replication via `luma_to_rgb`.



Normalization:



- If `normalize=True`, we globally rescale to `[0,1]` via `_normalize_01_global`.



#### `colorspace="color"`



- We first ensure a 3-channel RGB representation (2D  replicated RGB).  

- Convert RGB  YCbCr using `_rgb_to_ycbcr` / `_ycbcr_to_rgb`.



```txt
RGB  Y, Cb, Cr
```



(approx. BT.601)

$Y = 0.299 R + 0.587 G + 0.114 B$,

$Cb = -0.168736 R - 0.331264 G + 0.5 B + 0.5$,

$Cr = 0.5 R - 0.418688 G - 0.081312 B + 0.5$.

- Filters are assigned as:



  | `mode`     | Y filter        | Cb filter      | Cr filter      |

  |-----------|-----------------|----------------|----------------|

  | `"mono"`  | `H_mono`        | identity       | identity       |

  | `"stereo"`| `H_mono`        | `H_L`          | `H_R`          |

  | `"mid-side"` | `H_mid`     | `H_sideL`      | `H_sideR`      |



- We recombine to YCbCr, clip softly, convert back to RGB and (optionally)
  clip to `[0,1]`.



#### Safe-color controls (s2i-safe-color, s2i-chroma-*)

These only affect `colorspace="color"`:

- `s2i-chroma-strength` blends filtered chroma with the original:
  $Cb = (1-a) Cb_{\text{orig}} + a \, Cb_{\text{filt}}$ (same for `Cr`),
  with $a \in [0,1]$.
- `s2i-safe-color` normalizes Y and squashes chroma extremes before RGB:
  - `Y` is globally rescaled to `[0,1]`.
  - For chroma, compute $d = Cb - 0.5$ (or $Cr - 0.5$), scale by the
    99th percentile of $|d|$ so that $|d|$ maps to $0.25$, then re-center.
- `s2i-chroma-clip` clamps chroma around neutral:
  $Cb, Cr = \mathrm{clip}(Cb, 0.5 - c, 0.5 + c)$, with $c$ the clip value.
- With `s2i-safe-color` disabled, we skip Y/chroma normalization and only
  clip RGB to `[0,1]` when `normalize=True`.
- With `normalize=False`, no rescale or clipping is applied.

This design keeps all the interesting stereo logic in the 2D filter
construction, while the image path is just apply this scalar filter to
chosen channels.

### 6.3 Image -> sound (i2s)

Image->sound lives in `exconv.xmodal.image2sound`. It derives an impulse
response `h[n]` from the image, then applies 1D linear convolution to audio:
$y[n] = \sum_k x[k] \, h[n - k]$.

The pipeline:

1. Convert the image into 1 or 2 scalar channels based on `i2s-colorspace`.
2. Build a 1D impulse per channel according to `i2s-mode`.
3. Convolve audio with the impulse (FFT-based), then crop via `i2s-pad-mode`.
4. Optionally normalize output loudness via `i2s-out-norm`.

#### Colorspace mapping (i2s-colorspace)

These determine whether the impulse is mono or stereo:

- `luma` (mono): Rec.709 luma  
  $Y = 0.2126 R + 0.7152 G + 0.0722 B$
- `rgb-mean` (mono): $Y = (R + G + B) / 3$
- `rgb-stereo` (stereo): left = `R`, right = `B` (if no `B`, repeat `R`)
- `ycbcr-mid-side` (stereo):
  $Y = 0.299 R + 0.587 G + 0.114 B$,
  $Cb = 0.564 (B - Y) + 0.5$,
  $Cr = 0.713 (R - Y) + 0.5$,
  $L = \mathrm{clip}(Y + (Cr - 0.5))$,
  $R = \mathrm{clip}(Y + (Cb - 0.5))$.

#### Impulse modes (i2s-mode)

- `flat`  flatten grayscale values into a 1D impulse  
  $h = \mathrm{ravel}(g)$ (optionally decimated for length control).  
  Optional DC removal: $h = h - \mathrm{mean}(h)$.

- `hist`  histogram of grayscale values  
  $h[b] = |\{ g \in \text{bin } b \}|$ for $b = 0..n\_bins-1$.  
  Impulse length is `n_bins`.

- `radial`  FFT magnitude radial profile:
  $G(u,v) = \mathrm{FFT2}(g)$,
  $M(u,v) = |\mathrm{shift}(G)|$,
  $p[k] = \mathrm{mean}\{ M(u,v) : \mathrm{bin}(u,v) = k \}$,
  with radial binning (linear or log) and optional Hann smoothing.

#### Radial options (i2s-radius-mode, i2s-phase-mode, i2s-smoothing)

- `i2s-radius-mode`:
  - `linear`: $bin = \lfloor r (K - 1) \rfloor$
  - `log`: $bin = \lfloor \log(1 + 9 r) / \log(10) \cdot (K - 1) \rfloor$
  - where $K = n\_bins$
- `i2s-smoothing`:
  - `hann`: multiply `p[k]` by a Hann window.
  - `none`: no smoothing.
- `i2s-phase-mode` (radial only):
  - `zero`  $\phi[k] = 0$.
  - `random`  $\phi[k] \sim U(-\pi, \pi)$ per bin.
  - `image`  circular-mean phase per radial bin:  
    $\phi[k] = \arg(\mathrm{mean}(\exp(i \, \theta(u,v))))$.
  - `spiral`  deterministic spiral walk from center to edges; picks phases
    in that order to form $\phi[k]$.
  - `min-phase`  minimum-phase reconstruction:  
    $c = \mathrm{irfft}(\log |H|)$, then
    $H_{\min} = \exp(\mathrm{rfft}(c_{\min}))$,
    where $c_{\min}$ doubles positive quefrencies.

Then $H_{\text{half}}[k] = p[k] \, \exp(i \, \phi[k])$, and the impulse is
$h = \mathrm{irfft}(H_{\text{half}}, n=\text{impulse\_len})$.

#### Length, padding, normalization (i2s-impulse-len, i2s-pad-mode, i2s-*)

- `i2s-impulse-len`:
  - `int`  fixed impulse length (radial mode).
  - `auto`  set to the audio length `N` (radial mode).
  - `frame`  in video-biconv, $impulse\_len = \mathrm{round}(sr / fps)$.
- `i2s-impulse-norm`:
  - `peak`  divide by max abs.
  - `energy`  divide by L2 norm ($rms \cdot \sqrt{N}$), so $\lVert h \rVert_2 = 1$.
  - `none`  no scaling.
- `i2s-pad-mode`:
  - `full`  length $N + K - 1$.
  - `same-first`  first `N` samples.
  - `same-center`  centered window of length `N`.
- `i2s-out-norm` rescales the output to match input loudness:
  - `match_peak`  $y \leftarrow y \cdot \mathrm{peak}(x) / \mathrm{peak}(y)$.
  - `match_rms`  $y \leftarrow y \cdot \mathrm{rms}(x) / \mathrm{rms}(y)$.
  - `none`  no scaling.

### 6.4 Video bi-conv blocks

`biconv_video_*` groups frames into blocks so audio and video stay aligned.

- Block boundaries are frame indices. In `fixed`, they come from `block_size`

  (or `block_size_div`); in `beats`/`novelty`/`structure`, boundaries are derived

  from audio and mapped to frames.

- For each block, the same audio chunk (matching that block's time range

  after `audio_length_mode`) drives sound->image for all frames in the block.

- Image->sound uses the block mean image to synthesize audio for that block;

  in `serial-image-first`, the mean is taken after sound->image processing.

  Output audio is the concatenation of per-block chunks.

---


## 7. CLI and library boundary



The CLI (`exconv.cli.exconv_cli`) is deliberately thin: it wires argparse

to the high-level APIs:



- `audio-auto`  `conv1d.audio.auto_convolve`

- `img-auto`  `conv2d.image_auto_convolve` / `pair_convolve` with Gaussian

  kernels parsed from strings.

- `sound2image`  `xmodal.sound2image.spectral_sculpt` (soundimage demo).



The design intention is:



- New research experiments should first be prototyped in Python using the

  library API.

- Once they stabilize, a small CLI faade can be added for quick runs and

  batch experiments.



---
