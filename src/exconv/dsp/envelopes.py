from __future__ import annotations

from typing import Literal

import numpy as np

EnvelopeCurve = Literal["linear", "equal-energy", "equal-power"]

__all__ = ["EnvelopeCurve", "fade_curve", "adsr_envelope", "apply_adsr"]


def fade_curve(n: int, curve: EnvelopeCurve) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.ones((1,), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
    if curve == "linear":
        out = t
    elif curve == "equal-energy":
        out = np.sqrt(t)
    elif curve == "equal-power":
        out = np.sin(t * (np.pi / 2.0))
    else:
        raise ValueError(f"Unknown envelope curve: {curve}")
    return out.astype(np.float32)


def adsr_envelope(
    n: int,
    sr: int,
    *,
    attack_s: float,
    decay_s: float,
    sustain: float,
    release_s: float,
    curve: EnvelopeCurve,
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if sr <= 0:
        raise ValueError("sr must be > 0 for ADSR envelope")
    if attack_s < 0 or decay_s < 0 or release_s < 0:
        raise ValueError("ADSR times must be >= 0")
    if sustain < 0.0 or sustain > 1.0:
        raise ValueError("ADSR sustain must be in [0, 1]")

    a = int(round(attack_s * sr))
    d = int(round(decay_s * sr))
    r = int(round(release_s * sr))
    if a < 0:
        a = 0
    if d < 0:
        d = 0
    if r < 0:
        r = 0

    total = a + d + r
    if total > n and total > 0:
        scale = n / float(total)
        a = int(round(a * scale))
        d = int(round(d * scale))
        r = int(round(r * scale))
        while a + d + r > n:
            if r > 0:
                r -= 1
            elif d > 0:
                d -= 1
            elif a > 0:
                a -= 1
            else:
                break

    s_len = n - (a + d + r)
    env = np.empty(n, dtype=np.float32)
    idx = 0

    if a > 0:
        env[idx : idx + a] = fade_curve(a, curve)
        idx += a
    if d > 0:
        fade_out = fade_curve(d, curve)[::-1]
        env[idx : idx + d] = sustain + (1.0 - sustain) * fade_out
        idx += d
    if s_len > 0:
        env[idx : idx + s_len] = sustain
        idx += s_len
    if r > 0:
        fade_out = fade_curve(r, curve)[::-1]
        env[idx : idx + r] = sustain * fade_out
        idx += r
    if idx < n:
        env[idx:] = sustain
    return env


def apply_adsr(
    audio: np.ndarray,
    sr: int,
    *,
    attack_s: float,
    decay_s: float,
    sustain: float,
    release_s: float,
    curve: EnvelopeCurve,
) -> np.ndarray:
    if (
        attack_s <= 0.0
        and decay_s <= 0.0
        and release_s <= 0.0
        and abs(sustain - 1.0) < 1e-9
    ):
        return audio
    env = adsr_envelope(
        int(audio.shape[0]),
        sr,
        attack_s=attack_s,
        decay_s=decay_s,
        sustain=sustain,
        release_s=release_s,
        curve=curve,
    )
    if audio.ndim == 1:
        return (audio * env).astype(np.float32, copy=False)
    return (audio * env[:, None]).astype(np.float32, copy=False)
