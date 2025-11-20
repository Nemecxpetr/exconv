# src/exconv/conv1d/audio.py
"""1D audio convolution via FFT: auto- and pair-convolution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from exconv.core.fft import linear_freq_multiply
from exconv.core.norms import apply_normalize


__all__ = ["Audio", "auto_convolve", "pair_convolve"]


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


def _convolve_samples(
    x: np.ndarray,
    h: np.ndarray,
    mode: str = "same-center",
    circular: bool = False,
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
            # IMPORTANT: use_real_fft=False to match np.convolve semantics
            y_c = linear_freq_multiply(
                x_c,
                h_c,
                axes=0,
                mode=fft_mode,
                use_real_fft=False,
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
    y_mono = linear_freq_multiply(
        x_mono,
        h_mono,
        axes=0,
        mode=fft_mode,
        use_real_fft=False,  # same reasoning as above
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


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def auto_convolve(
    audio: Audio,
    mode: str = "same-center",
    circular: bool = False,
    normalize: Optional[str] = "rms",
    order: int = 2,
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
        samples = _convolve_samples(samples, audio.samples, mode=mode, circular=circular)

    # Final normalization only
    samples = apply_normalize(samples, normalize)
    return _wrap_audio_output(samples, audio.sr)


def pair_convolve(
    x: Audio,
    h: Audio,
    mode: str = "same-center",
    circular: bool = False,
    normalize: Optional[str] = "rms",
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

    samples = _convolve_samples(x.samples, h.samples, mode=mode, circular=circular)
    samples = apply_normalize(samples, normalize)
    return _wrap_audio_output(samples, x.sr)
