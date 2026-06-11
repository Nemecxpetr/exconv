# Audio Spectral Processing

`exconv` can optionally shape the frequency-domain spectra used by audio
convolution before the inverse FFT. This is a creative mode, not plain
mathematical convolution. With no spectral processor supplied, audio convolution
still computes the normal FFT product:

```text
Y(f) = X(f) * H(f)
```

With spectral processing enabled, `exconv` can blur low-frequency magnitudes,
sharpen high-frequency magnitudes, add high-frequency gain, preserve bass from
the reference input, and reduce low-frequency phase contribution from the
kernel.

## Why Use It

Some sample combinations can make the expected pitch feel unstable because the
convolution product suppresses or phase-rotates important low bins. Spectral
processing gives you controls to keep the low end steadier while making upper
partials more present.

Useful goals:

- keep sub/bass bins closer to the source signal;
- add brightness after a dark convolution;
- sharpen harmonics without globally raising the whole spectrum;
- smooth noisy low bins before they dominate the result;
- experiment with image-filter-like spectrum operations.

## Python API

```python
from exconv.conv1d import Audio, AudioSpectralProcessing, pair_convolve

processing = AudioSpectralProcessing(
    crossover_hz=80.0,
    transition_hz=400.0,
    low_preserve=0.85,
    phase_low=0.15,
    phase_high=1.0,
    bass_blur=0.35,
    treble_sharpen=0.8,
    high_gain_db=3.0,
    contrast=1.15,
    blur_bins=5,
    max_gain_db=18.0,
)

out = pair_convolve(
    Audio(samples=x, sr=sr),
    Audio(samples=h, sr=sr),
    mode="same-center",
    normalize="rms",
    spectral_processing=processing,
)
```

The same `spectral_processing=` argument is available on:

- `auto_convolve`
- `pair_convolve`
- `multi_convolve`
- `convolution_family`

If `AudioSpectralProcessing()` is neutral, the output matches plain
convolution. Set at least one non-neutral control to hear an effect.

## CLI Quick Start

`--spectral` and `--audio-spectral` are enable switches. By themselves they use
neutral defaults, so they should sound like normal convolution apart from tiny
floating-point differences. Add controls such as `--low-preserve`,
`--bass-blur`, `--treble-sharpen`, `--high-gain-db`, or
`--spectral-contrast` to make the processor audible.

Single-file auto-convolution:

```bash
exconv audio-auto \
  --in input.wav \
  --out output_bright.wav \
  --order 2 \
  --spectral \
  --spectral-crossover 80 \
  --spectral-transition 400 \
  --low-preserve 0.8 \
  --phase-low 0.2 \
  --treble-sharpen 0.7 \
  --high-gain-db 3 \
  --spectral-contrast 1.1
```

Folder batch:

```bash
exconv folderbatch my_project \
  --root samples \
  --audio-mode same-center \
  --audio-normalize rms \
  --audio-spectral \
  --audio-spectral-crossover 80 \
  --audio-spectral-transition 400 \
  --audio-low-preserve 0.8 \
  --audio-phase-low 0.2 \
  --audio-bass-blur 0.35 \
  --audio-treble-sharpen 0.7 \
  --audio-high-gain-db 3 \
  --audio-spectral-contrast 1.1
```

## Controls

| Python field | `audio-auto` CLI | `folderbatch` CLI | Meaning |
|--------------|------------------|-------------------|---------|
| enable switch | `--spectral` | `--audio-spectral` | Enables the spectral-processing path. Neutral by itself. |
| `crossover_hz` | `--spectral-crossover` | `--audio-spectral-crossover` | Frequency where bass protection starts fading out. |
| `transition_hz` | `--spectral-transition` | `--audio-spectral-transition` | Frequency where treble processing is fully active. |
| `bass_blur` | `--bass-blur` | `--audio-bass-blur` | 0..1 blur amount for low-frequency magnitudes. |
| `treble_sharpen` | `--treble-sharpen` | `--audio-treble-sharpen` | Unsharp-mask amount for high-frequency magnitudes. |
| `high_gain_db` | `--high-gain-db` | `--audio-high-gain-db` | High-frequency gain ramp in dB. |
| `contrast` | `--spectral-contrast` | `--audio-spectral-contrast` | High-frequency contrast. `1.0` is neutral. |
| `low_preserve` | `--low-preserve` | `--audio-low-preserve` | 0..1 blend low bins back toward the reference input spectrum. |
| `phase_low` | `--phase-low` | `--audio-phase-low` | Kernel phase contribution below crossover. |
| `phase_high` | `--phase-high` | `--audio-phase-high` | Kernel phase contribution above transition. |
| `blur_bins` | `--spectral-blur-bins` | `--audio-spectral-blur-bins` | Gaussian blur radius in rFFT bins. |
| `max_gain_db` | `--spectral-max-gain-db` | `--audio-spectral-max-gain-db` | Magnitude growth clamp. |
| `process_operands` | `--spectral-operands` | `--audio-spectral-operands` | Shape each input spectrum before multiplying instead of shaping the product. |

## Frequency-Only Spectral Blur

The blur in `AudioSpectralProcessing` is a one-dimensional blur over frequency
bins. It does not use an STFT and it does not blur across time frames.

For one audio channel, `exconv` computes a single rFFT:

```text
X[k] = rFFT(x)[k]
H[k] = rFFT(h)[k]
```

Then it either multiplies first:

```text
P[k] = X[k] * H[k]
```

or, if `process_operands=True`, shapes `X[k]` and `H[k]` separately before
multiplication.

The frequency-only blur is applied to the magnitude vector:

```text
M[k] = abs(P[k])
```

with a Gaussian kernel over neighboring frequency bins:

```text
G[i] = exp(-0.5 * (i / sigma)^2)
G[i] = G[i] / sum(G)
```

and:

```text
Blur(M)[k] = sum_i G[i] * M[k - i]
```

In code terms, this is like image blur, but only along one axis:

```text
frequency bins:  [0 Hz, ..., Nyquist]
time frames:     not present
operation:       1D convolution over frequency bins
```

The bass blur control blends the original magnitude and the blurred magnitude
with a frequency-dependent mask:

```text
t[k] = smoothstep(crossover_hz, transition_hz, freq[k])
bass[k] = 1 - t[k]

M_bass[k] = lerp(M[k], Blur(M)[k], bass_blur * bass[k])
```

So with `crossover_hz=80` and `transition_hz=400`, the blur is strongest below
80 Hz, fades out between 80 and 400 Hz, and is inactive above 400 Hz.

After magnitude processing, the phase is restored:

```text
P_blurred[k] = M_bass[k] * exp(j * angle(P[k]))
```

This means spectral blur changes the local energy distribution between nearby
frequency bins, but it does not smear transients forward/backward in time the
way an STFT spectrogram blur would.

### Why It Sounds Different From Time-Frequency Blur

An STFT spectrogram is a matrix:

```text
S[time_frame, frequency_bin]
```

A 2D blur on that matrix can smear energy both sideways in frequency and
forward/backward in time. That can soften attacks and create temporal smearing.

The current `exconv` blur uses one full-signal FFT per convolution channel:

```text
P[frequency_bin]
```

There is no time-frame axis in this representation. The blur only spreads
magnitude between neighboring frequencies. It is closer to smoothing a filter
curve than blurring a spectrogram image.

### Frequency-Only Blur Recipe

Use this when you want low bins to become less spiky while leaving timing and
transient placement to the normal convolution result:

```python
AudioSpectralProcessing(
    crossover_hz=60,
    transition_hz=300,
    bass_blur=0.75,
    blur_bins=9,
    low_preserve=0.35,
    phase_low=0.4,
)
```

CLI equivalent:

```bash
exconv audio-auto \
  --in input.wav \
  --out output_freq_blur.wav \
  --spectral \
  --spectral-crossover 60 \
  --spectral-transition 300 \
  --bass-blur 0.75 \
  --spectral-blur-bins 9 \
  --low-preserve 0.35 \
  --phase-low 0.4
```

For folder batch, prefix the same controls with `--audio-`:

```bash
exconv folderbatch my_project \
  --audio-spectral \
  --audio-spectral-crossover 60 \
  --audio-spectral-transition 300 \
  --audio-bass-blur 0.75 \
  --audio-spectral-blur-bins 9 \
  --audio-low-preserve 0.35 \
  --audio-phase-low 0.4
```

## Suggested Recipes

### Bass-Stable Bright Convolution

Keeps lows close to the source while adding upper partials:

```python
AudioSpectralProcessing(
    crossover_hz=80,
    transition_hz=400,
    low_preserve=0.85,
    phase_low=0.2,
    treble_sharpen=0.6,
    high_gain_db=3.0,
    contrast=1.1,
)
```

### Sub-Bass Smoothing

Useful when low bins jump or dominate:

```python
AudioSpectralProcessing(
    crossover_hz=50,
    transition_hz=250,
    bass_blur=0.6,
    low_preserve=0.5,
    blur_bins=7,
)
```

### Harmonic Edge

More aggressive high-frequency partial emphasis:

```python
AudioSpectralProcessing(
    crossover_hz=250,
    transition_hz=1200,
    treble_sharpen=1.2,
    contrast=1.35,
    high_gain_db=2.0,
    max_gain_db=12.0,
)
```

### Pre-Multiply Spectrum Filtering

This is closest to "image filtering before multiplication":

```python
AudioSpectralProcessing(
    process_operands=True,
    bass_blur=0.35,
    treble_sharpen=0.8,
    high_gain_db=2.0,
)
```

Use `process_operands=False` when you want to shape the final convolution
product. Use `process_operands=True` when you want each sample's spectrum to be
filtered before the multiplication interaction happens.

## Practical Notes

- `crossover_hz=50` only protects deep sub-bass. For perceived pitch stability,
  try a wider range such as `80-400 Hz`.
- `low_preserve=1.0` below the crossover makes lows much closer to the
  reference input and can reduce the characteristic convolution effect there.
- `phase_low=0.0` keeps low-frequency phase from the reference input while
  still allowing high-frequency convolution phase.
- High gain, sharpening, and contrast can raise noise. Use `max_gain_db` as a
  limiter for spectral magnitude growth.
- These tools operate on the convolution FFT, not on an STFT spectrogram over
  time. They are image-filter-like along the frequency axis, but they do not
  blur across time frames.
