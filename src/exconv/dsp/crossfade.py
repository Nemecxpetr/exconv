from __future__ import annotations

from typing import Literal

import numpy as np

CrossfadeMode = Literal["none", "lin", "equal", "power"]

__all__ = ["CrossfadeMode", "crossfade_weights"]


def crossfade_weights(n: int, mode: CrossfadeMode) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if mode == "none":
        return np.zeros((n,), dtype=np.float32), np.ones((n,), dtype=np.float32)
    if n == 1:
        t = np.array([0.5], dtype=np.float32)
    else:
        t = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
    if mode == "lin":
        fade_in = t
        fade_out = 1.0 - t
    elif mode == "equal":
        fade_in = np.sqrt(t).astype(np.float32)
        fade_out = np.sqrt(1.0 - t).astype(np.float32)
    elif mode == "power":
        fade_in = np.sin(t * (np.pi / 2.0)).astype(np.float32)
        fade_out = np.cos(t * (np.pi / 2.0)).astype(np.float32)
    else:
        raise ValueError(f"Unknown crossfade mode: {mode}")
    return fade_out.astype(np.float32), fade_in.astype(np.float32)
