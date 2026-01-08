from __future__ import annotations

import numpy as np

from exconv.core.grids import hann as _hann
from exconv.core.grids import tukey as _tukey

__all__ = ["hann", "tukey"]


def hann(n: int) -> np.ndarray:
    return _hann(n)


def tukey(n: int, alpha: float = 0.5) -> np.ndarray:
    return _tukey(n, alpha=alpha)
