# src/exconv/conv1d/audio.py
"""1D audio convolution via FFT: auto- and pair-convolution."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np

from exconv.core.fft import fftnd, ifftnd, linear_freq_multiply, next_fast_len_ge
from exconv.core.norms import apply_normalize

__all__ = [
    "Audio",
    "AudioConvolutionResult",
    "AudioSpectralProcessing",
    "auto_convolve",
    "pair_convolve",
    "multi_convolve",
    "convolution_family",
]


@dataclass
class Audio:
    """
    Simple audio container.

    Attributes
    ----------
    samples : np.ndarray
        Shape (N,) for mono or (N, C) for multi-channel. Time axis is always 0.
    sr : int
        Sampling rate in Hz.
    """

    samples: np.ndarray
    sr: int

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_channels(self) -> int:
        if self.samples.ndim == 1:
            return 1
        return int(self.samples.shape[1])

    @property
    def is_mono(self) -> bool:
        return self.n_channels == 1

    def copy(self) -> "Audio":
        return Audio(samples=np.array(self.samples, copy=True), sr=int(self.sr))


@dataclass(frozen=True)
class AudioConvolutionResult:
    """
    Named result from a generated family of 1D audio convolutions.

    Attributes
    ----------
    kind : {"self","pair","multi"}
        Which subset/order produced the result.
    indices : tuple[int, ...]
        Input indices used for the convolution.
    names : tuple[str, ...]
        Optional input names, aligned with `indices`.
    audio : Audio
        The convolved audio output.
    """

    kind: str
    indices: tuple[int, ...]
    names: tuple[str, ...]
    audio: Audio


@dataclass(frozen=True)
class AudioSpectralProcessing:
    """
    Creative frequency-domain shaping applied around FFT multiplication.

    Supplying this config makes audio convolution intentionally non-pure:
    spectra can be smoothed, sharpened, tilted, contrast-shaped, phase-limited,
    or blended back to the dry low-frequency reference. When no config is
    supplied, the normal mathematical convolution path is used.
    """

    crossover_hz: float = 80.0
    transition_hz: float = 400.0
    bass_blur: float = 0.0
    treble_sharpen: float = 0.0
    high_gain_db: float = 0.0
    contrast: float = 1.0
    low_preserve: float = 0.0
    phase_low: float = 1.0
    phase_high: float = 1.0
    blur_bins: int = 3
    max_gain_db: float = 24.0
    process_operands: bool = False

    def is_neutral(self) -> bool:
        return (
            abs(float(self.bass_blur)) <= 1e-12
            and abs(float(self.treble_sharpen)) <= 1e-12
            and abs(float(self.high_gain_db)) <= 1e-12
            and abs(float(self.contrast) - 1.0) <= 1e-12
            and abs(float(self.low_preserve)) <= 1e-12
            and abs(float(self.phase_low) - 1.0) <= 1e-12
            and abs(float(self.phase_high) - 1.0) <= 1e-12
        )


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """
    Ensure array is (N, C).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError("Audio samples must be 1D (N,) or 2D (N, C).")


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float64)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gaussian_blur_1d(values: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0 or values.size <= 1:
        return values
    sigma = max(radius / 2.0, 1e-6)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _shape_spectrum_magnitude(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    processing: AudioSpectralProcessing,
    *,
    base_mag: Optional[np.ndarray] = None,
) -> np.ndarray:
    mag = np.abs(spectrum).astype(np.float64, copy=False)
    if mag.size == 0:
        return spectrum

    original_mag = mag.copy()
    treble = _smoothstep(processing.crossover_hz, processing.transition_hz, freqs)
    bass = 1.0 - treble
    blur_radius = max(0, int(processing.blur_bins))

    if processing.bass_blur > 0.0:
        blurred = _gaussian_blur_1d(mag, blur_radius)
        amount = np.clip(float(processing.bass_blur), 0.0, 1.0) * bass
        mag = mag * (1.0 - amount) + blurred * amount

    if processing.treble_sharpen != 0.0:
        blurred = _gaussian_blur_1d(mag, blur_radius)
        sharp = np.maximum(
            mag + float(processing.treble_sharpen) * (mag - blurred), 0.0
        )
        mag = mag * (1.0 - treble) + sharp * treble

    if processing.high_gain_db != 0.0:
        mag *= 10.0 ** ((float(processing.high_gain_db) * treble) / 20.0)

    if abs(float(processing.contrast) - 1.0) > 1e-12:
        floor = np.finfo(np.float64).eps
        local = _gaussian_blur_1d(mag, max(blur_radius, 1))
        contrasted = local * np.power(
            np.maximum(mag, floor) / np.maximum(local, floor),
            float(processing.contrast),
        )
        mag = mag * (1.0 - treble) + contrasted * treble

    if processing.max_gain_db is not None:
        max_gain = 10.0 ** (float(processing.max_gain_db) / 20.0)
        limit_base = (
            original_mag if base_mag is None else np.asarray(base_mag, dtype=np.float64)
        )
        mag = np.minimum(
            mag, np.maximum(limit_base * max_gain, np.finfo(np.float64).eps)
        )

    return mag * np.exp(1j * np.angle(spectrum))


def _processed_frequency_product(
    spectra: list[np.ndarray],
    freqs: np.ndarray,
    processing: AudioSpectralProcessing,
) -> np.ndarray:
    if not spectra:
        raise ValueError("spectra cannot be empty")

    if processing.process_operands:
        shaped = [_shape_spectrum_magnitude(S, freqs, processing) for S in spectra]
        product = np.array(shaped[0], copy=True)
        for S in shaped[1:]:
            product *= S
    else:
        product = np.array(spectra[0], copy=True)
        for S in spectra[1:]:
            product *= S
        product = _shape_spectrum_magnitude(
            product, freqs, processing, base_mag=np.abs(product)
        )

    if (
        abs(float(processing.phase_low) - 1.0) > 1e-12
        or abs(float(processing.phase_high) - 1.0) > 1e-12
    ):
        treble = _smoothstep(processing.crossover_hz, processing.transition_hz, freqs)
        phase_amount = (
            float(processing.phase_low) * (1.0 - treble)
            + float(processing.phase_high) * treble
        )
        phase = np.angle(spectra[0])
        for S in spectra[1:]:
            phase += np.angle(S) * phase_amount
        product = np.abs(product) * np.exp(1j * phase)

    if processing.low_preserve > 0.0:
        low = 1.0 - _smoothstep(
            processing.crossover_hz, processing.transition_hz, freqs
        )
        amount = np.clip(float(processing.low_preserve), 0.0, 1.0) * low
        product = product * (1.0 - amount) + spectra[0] * amount

    return product


def _spectral_convolve_channel(
    signals: list[np.ndarray],
    sr: int,
    *,
    mode: str = "same-center",
    circular: bool = False,
    processing: AudioSpectralProcessing,
) -> np.ndarray:
    lengths = [int(np.asarray(s).shape[0]) for s in signals]
    if any(length <= 0 for length in lengths):
        raise ValueError("All signals must contain at least one sample")

    if circular:
        fft_len = lengths[0]
        true_len = fft_len
        start = 0
        out_len = fft_len
    else:
        if mode not in ("full", "same-first", "same-center"):
            raise ValueError(f"Unknown mode {mode!r}")
        true_len = sum(lengths) - (len(lengths) - 1)
        fft_len = next_fast_len_ge(true_len)
        ref_len = lengths[0]
        if mode == "full":
            start = 0
            out_len = true_len
        elif mode == "same-first":
            start = 0
            out_len = ref_len
        else:
            start = (true_len - ref_len) // 2
            out_len = ref_len

    spectra = [
        fftnd(np.asarray(s, dtype=np.float64), axes=0, real_input=True, n=[fft_len])
        for s in signals
    ]
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / float(sr))
    product = _processed_frequency_product(spectra, freqs, processing)
    y = ifftnd(product, axes=0, real_output=True, n=[fft_len])
    y = np.asarray(y, dtype=np.float64).ravel()
    if circular:
        return y[:out_len]
    return y[:true_len][start : start + out_len]


def _convolve_samples(
    x: np.ndarray,
    h: np.ndarray,
    sr: Optional[int] = None,
    mode: str = "same-center",
    circular: bool = False,
    spectral_processing: Optional[AudioSpectralProcessing] = None,
) -> np.ndarray:
    """
    Core 1D convolution engine on raw sample arrays.

    Parameters
    ----------
    x, h : ndarray
        x: reference signal (defines 'same-*' length / circular length).
        h: kernel signal.
        Shapes: (N,) or (N, C). Time axis is 0.
    mode : {"full","same-first","same-center"}
        Linear convolution mode. Only used if circular == False.
    circular : bool
        If True, perform circular convolution with length == len(x)
        along the time axis. Ignores `mode` for length.

    Returns
    -------
    y : ndarray
        Convolution result. Shape (L, C) or (L, 1) for mono.
    """
    x2 = _ensure_2d(np.asarray(x, dtype=np.float64))
    h2 = _ensure_2d(np.asarray(h, dtype=np.float64))

    Nx, Cx = x2.shape
    Nh, Ch = h2.shape

    # Decide effective fft mode for linear_freq_multiply
    if circular:
        fft_mode = "circular"
    else:
        if mode not in ("full", "same-first", "same-center"):
            raise ValueError(f"Unknown mode {mode!r}")
        fft_mode = mode

    # Channel strategy:
    # - if Cx == Ch: per-channel convolution
    # - else: downmix both to mono, return mono (as (L,1) here)
    if Cx == Ch:
        outs = []
        max_len = 0
        for c in range(Cx):
            x_c = x2[:, c]
            h_c = h2[:, c]
            if spectral_processing is None or spectral_processing.is_neutral():
                y_c = linear_freq_multiply(
                    x_c,
                    h_c,
                    axes=0,
                    mode=fft_mode,
                    use_real_fft=True,
                )
            else:
                if sr is None:
                    raise ValueError("sr is required for spectral_processing")
                y_c = _spectral_convolve_channel(
                    [x_c, h_c],
                    sr,
                    mode=mode,
                    circular=circular,
                    processing=spectral_processing,
                )
            y_c = np.asarray(y_c, dtype=np.float64).ravel()
            outs.append(y_c)
            max_len = max(max_len, y_c.size)

        # Stack and pad per channel to same length
        y = np.zeros((max_len, Cx), dtype=np.float64)
        for c, yc in enumerate(outs):
            y[: yc.size, c] = yc
        return y

    # Mismatched channels: downmix to mono
    x_mono = x2.mean(axis=1)
    h_mono = h2.mean(axis=1)
    if spectral_processing is None or spectral_processing.is_neutral():
        y_mono = linear_freq_multiply(
            x_mono,
            h_mono,
            axes=0,
            mode=fft_mode,
            use_real_fft=True,
        )
    else:
        if sr is None:
            raise ValueError("sr is required for spectral_processing")
        y_mono = _spectral_convolve_channel(
            [x_mono, h_mono],
            sr,
            mode=mode,
            circular=circular,
            processing=spectral_processing,
        )
    y_mono = np.asarray(y_mono, dtype=np.float64).ravel()
    return y_mono[:, None]


def _wrap_audio_output(samples: np.ndarray, sr: int) -> Audio:
    """
    Flatten mono (N,1) → (N,) for public Audio API.
    """
    samples = np.asarray(samples)
    if samples.ndim == 2 and samples.shape[1] == 1:
        samples = samples[:, 0]
    return Audio(samples=samples, sr=sr)


def _multi_convolve_samples(
    samples_list: list[np.ndarray],
    sr: Optional[int] = None,
    mode: str = "same-center",
    circular: bool = False,
    spectral_processing: Optional[AudioSpectralProcessing] = None,
) -> np.ndarray:
    """
    Core multi-signal 1D convolution engine via frequency multiplication.

    Parameters
    ----------
    samples_list : list[ndarray]
        List of sample arrays, each (N,) or (N, C).
    mode : {"full","same-first","same-center"}
        Linear convolution mode. Only used if circular == False.
    circular : bool
        If True, perform circular convolution with length == len(first signal)
        along the time axis. Ignores `mode` for length.

    Returns
    -------
    y : ndarray
        Convolution result. Shape (L, C) or (L, 1) for mono.
    """
    if not samples_list:
        raise ValueError("samples_list cannot be empty")

    samples_2d = [_ensure_2d(np.asarray(s, dtype=np.float64)) for s in samples_list]
    lengths = [s.shape[0] for s in samples_2d]
    channels = [s.shape[1] for s in samples_2d]
    if any(length <= 0 for length in lengths):
        raise ValueError("All signals must contain at least one sample")

    # Match pair_convolve semantics: equal channel counts convolve per channel;
    # mismatched channel counts are downmixed to mono before convolution.
    if len(set(channels)) > 1:
        channel_sources = [s.mean(axis=1, keepdims=True) for s in samples_2d]
        out_channels = 1
    else:
        channel_sources = samples_2d
        out_channels = channels[0]

    n_signals = len(samples_2d)

    if spectral_processing is not None and not spectral_processing.is_neutral():
        if sr is None:
            raise ValueError("sr is required for spectral_processing")
        if circular:
            out_len = lengths[0]
        else:
            if mode not in ("full", "same-first", "same-center"):
                raise ValueError(f"Unknown mode {mode!r}")
            out_len = sum(lengths) - (n_signals - 1) if mode == "full" else lengths[0]

        result = np.empty((out_len, out_channels), dtype=np.float64)
        for c in range(out_channels):
            result[:, c] = _spectral_convolve_channel(
                [s[:, c] for s in channel_sources],
                sr,
                mode=mode,
                circular=circular,
                processing=spectral_processing,
            )
        return result

    if circular:
        fft_len = lengths[0]
        result = np.empty((fft_len, out_channels), dtype=np.float64)
        for c in range(out_channels):
            result_fft = None
            for s in channel_sources:
                x = s[:, c]
                X = fftnd(x, axes=0, real_input=True, n=[fft_len])
                if result_fft is None:
                    result_fft = np.array(X, copy=True)
                else:
                    result_fft *= X
            y = ifftnd(result_fft, axes=0, real_output=True, n=[fft_len])
            result[:, c] = np.asarray(y, dtype=np.float64).ravel()
        return result

    # Linear convolution
    if mode not in ("full", "same-first", "same-center"):
        raise ValueError(f"Unknown mode {mode!r}")

    full_len = sum(lengths) - (n_signals - 1)
    fft_len = next_fast_len_ge(full_len)

    ref_len = lengths[0]
    if mode == "same-first":
        start = 0
    elif mode == "same-center":
        start = (full_len - ref_len) // 2
    else:
        start = 0
    end = full_len if mode == "full" else start + ref_len
    out_len = end - start

    result = np.empty((out_len, out_channels), dtype=np.float64)
    for c in range(out_channels):
        result_fft = None
        for s in channel_sources:
            X = fftnd(s[:, c], axes=0, real_input=True, n=[fft_len])
            if result_fft is None:
                result_fft = np.array(X, copy=True)
            else:
                result_fft *= X

        y_tmp = ifftnd(result_fft, axes=0, real_output=True, n=[fft_len])
        result[:, c] = np.asarray(y_tmp[start:end], dtype=np.float64).ravel()

    return result


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def auto_convolve(
    audio: Audio,
    mode: str = "same-center",
    circular: bool = False,
    normalize: Optional[str] = "rms",
    order: int = 2,
    spectral_processing: Optional[AudioSpectralProcessing] = None,
) -> Audio:
    """
    Self-convolution of an Audio object.

    Parameters
    ----------
    audio : Audio
        Input audio.
    mode : {"full","same-first","same-center"}
        Linear convolution mode when circular is False.
    circular : bool
        If True, perform circular convolution (length equal to `audio.n_samples`).
    normalize : {"rms","peak",None,"none"}
        Normalization applied to the *output only*. See `core.norms`.
    order : int, default=2
        n-th order self-convolution:
          - order=1 → returns a copy of the original Audio
          - order=2 → audio * audio
          - order=3 → audio * audio * audio, etc.

    Returns
    -------
    Audio
        Self-convolved audio with the same sample rate.
    """
    if order <= 0:
        raise ValueError("order must be >= 1")

    if order == 1:
        # No change, but return a copy for safety
        out = audio.copy()
        if normalize not in (None, "none"):
            out.samples = apply_normalize(out.samples, normalize)
        return out

    # Repeated convolution at the sample level (no normalization between steps)
    samples = np.asarray(audio.samples, dtype=np.float64)
    for _ in range(order - 1):
        samples = _convolve_samples(
            samples,
            audio.samples,
            sr=audio.sr,
            mode=mode,
            circular=circular,
            spectral_processing=spectral_processing,
        )

    # Final normalization only
    samples = apply_normalize(samples, normalize)
    return _wrap_audio_output(samples, audio.sr)


def pair_convolve(
    x: Audio,
    h: Audio,
    mode: str = "same-center",
    circular: bool = False,
    normalize: Optional[str] = "rms",
    spectral_processing: Optional[AudioSpectralProcessing] = None,
) -> Audio:
    """
    Convolution of two Audio objects.

    Parameters
    ----------
    x : Audio
        Reference signal (defines 'same-*' lengths and circular length).
    h : Audio
        Kernel signal.
    mode : {"full","same-first","same-center"}
        Linear convolution mode when circular is False.
    circular : bool
        If True, perform circular convolution (length equal to `x.n_samples`).
    normalize : {"rms","peak",None,"none"}
        Normalization applied to the *output only*. See `core.norms`.

    Returns
    -------
    Audio
        Convolved result with sample rate x.sr.

    Raises
    ------
    ValueError
        If sample rates do not match.
    """
    if x.sr != h.sr:
        raise ValueError(f"Sample rates must match (got {x.sr} vs {h.sr}).")

    samples = _convolve_samples(
        x.samples,
        h.samples,
        sr=x.sr,
        mode=mode,
        circular=circular,
        spectral_processing=spectral_processing,
    )
    samples = apply_normalize(samples, normalize)
    return _wrap_audio_output(samples, x.sr)


def multi_convolve(
    audios: list[Audio],
    mode: str = "same-center",
    circular: bool = False,
    normalize: Optional[str] = "rms",
    spectral_processing: Optional[AudioSpectralProcessing] = None,
) -> Audio:
    """
    Convolution of multiple Audio objects via frequency-domain multiplication.

    Mathematically, this computes the convolution of all input signals by
    multiplying their FFTs element-wise and inverse transforming.

    Parameters
    ----------
    audios : list[Audio]
        List of audio signals to convolve. Must have at least 1 element.
    mode : {"full","same-first","same-center"}
        Linear convolution mode when circular is False.
    circular : bool
        If True, perform circular convolution (length equal to the first audio's n_samples).
    normalize : {"rms","peak",None,"none"}
        Normalization applied to the *output only*. See `core.norms`.

    Returns
    -------
    Audio
        Convolved result with sample rate of the first audio.

    Raises
    ------
    ValueError
        If no audios provided, or sample rates do not match.
    """
    if not audios:
        raise ValueError("At least one audio must be provided")

    sr = audios[0].sr
    for a in audios[1:]:
        if a.sr != sr:
            raise ValueError(f"All sample rates must match (got {a.sr} vs {sr})")

    samples_list = [np.asarray(a.samples, dtype=np.float64) for a in audios]
    samples = _multi_convolve_samples(
        samples_list,
        sr=sr,
        mode=mode,
        circular=circular,
        spectral_processing=spectral_processing,
    )
    samples = apply_normalize(samples, normalize)
    return _wrap_audio_output(samples, sr)


def convolution_family(
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
    spectral_processing: Optional[AudioSpectralProcessing] = None,
) -> list[AudioConvolutionResult]:
    """
    Generate the usual family of 1D convolutions for a set of audio signals.

    This is useful for treating N audio files as N related 1D signals:
    each signal can be self-convolved, each unordered pair can be convolved,
    and the full set can be combined by one N-fold 1D convolution. The
    all-input result is mathematically equivalent to multiplying all spectra
    at a common FFT length and inverse transforming.

    Parameters
    ----------
    audios : list[Audio]
        Audio inputs. Must contain at least one signal.
    names : list[str] or None
        Optional labels for the inputs. Defaults to "0", "1", ...
    mode : {"full","same-first","same-center"}
        Linear convolution mode when circular is False.
    circular : bool
        If True, perform circular convolution with first-input length.
    normalize : {"rms","peak",None,"none"}
        Normalization applied to each output only. See `core.norms`.
    self_order : int, default=2
        Order for each self-convolution.
    include_self, include_pairs, include_multi : bool
        Select which result groups to produce. `include_multi` only emits an
        all-input result when at least two inputs are present.

    Returns
    -------
    list[AudioConvolutionResult]
        Results in stable order: self results, pair results, then multi.

    Raises
    ------
    ValueError
        If no audios are provided, names length does not match, sample rates
        mismatch, or self_order is invalid.
    """
    if not audios:
        raise ValueError("At least one audio must be provided")
    if self_order <= 0:
        raise ValueError("self_order must be >= 1")

    if names is None:
        result_names = tuple(str(i) for i in range(len(audios)))
    else:
        if len(names) != len(audios):
            raise ValueError("names length must match audios length")
        result_names = tuple(str(name) for name in names)

    sr = audios[0].sr
    for a in audios[1:]:
        if a.sr != sr:
            raise ValueError(f"All sample rates must match (got {a.sr} vs {sr})")

    results: list[AudioConvolutionResult] = []

    if include_self:
        for idx, audio in enumerate(audios):
            out = auto_convolve(
                audio,
                mode=mode,
                circular=circular,
                normalize=normalize,
                order=self_order,
                spectral_processing=spectral_processing,
            )
            results.append(
                AudioConvolutionResult(
                    kind="self",
                    indices=(idx,),
                    names=(result_names[idx],),
                    audio=out,
                )
            )

    if include_pairs:
        for i, j in combinations(range(len(audios)), 2):
            out = pair_convolve(
                audios[i],
                audios[j],
                mode=mode,
                circular=circular,
                normalize=normalize,
                spectral_processing=spectral_processing,
            )
            results.append(
                AudioConvolutionResult(
                    kind="pair",
                    indices=(i, j),
                    names=(result_names[i], result_names[j]),
                    audio=out,
                )
            )

    if include_multi and len(audios) >= 2:
        out = multi_convolve(
            audios,
            mode=mode,
            circular=circular,
            normalize=normalize,
            spectral_processing=spectral_processing,
        )
        results.append(
            AudioConvolutionResult(
                kind="multi",
                indices=tuple(range(len(audios))),
                names=result_names,
                audio=out,
            )
        )

    return results
