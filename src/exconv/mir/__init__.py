"""
Minimal MIR helpers for fast beat/novelty-driven segmentation.
"""

from .blocks import (
    BlockStrategy,
    audio_to_frame_blocks,
    estimate_tempo,
    peak_pick,
    quick_dtw_path,
    spectral_flux_novelty,
    structure_novelty,
)

__all__ = [
    "BlockStrategy",
    "audio_to_frame_blocks",
    "estimate_tempo",
    "peak_pick",
    "quick_dtw_path",
    "spectral_flux_novelty",
    "structure_novelty",
]
