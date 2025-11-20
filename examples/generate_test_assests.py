"""
generate_test_assets.py

Creates tiny synthetic audio + image assets for testing exconv.
Generates:
 - audio_long_sines.wav
 - audio_plucks.wav
 - img_checker.png
 - img_gradients.png
 - img_radial.png
"""

from pathlib import Path

import numpy as np
from PIL import Image
import soundfile as sf


# ------------------------------
# Utility: ADSR envelope
# ------------------------------
def adsr_envelope(n, sr, attack=0.05, decay=0.1, sustain=0.7, release=0.2):
    """Simple ADSR in *seconds* for attack/decay/release."""
    t = np.linspace(0, n / sr, n, endpoint=False)

    env = np.zeros(n, dtype=np.float32)
    a_end = attack
    d_end = attack + decay
    r_start = (n / sr) - release

    # Attack
    mask_attack = t < a_end
    env[mask_attack] = (t[mask_attack] / attack).astype(np.float32)

    # Decay
    mask_decay = (t >= a_end) & (t < d_end)
    env[mask_decay] = (
        1.0 - (1.0 - sustain) * ((t[mask_decay] - a_end) / decay)
    ).astype(np.float32)

    # Sustain
    mask_sustain = (t >= d_end) & (t < r_start)
    env[mask_sustain] = sustain

    # Release
    mask_release = t >= r_start
    if release > 0:
        env[mask_release] = sustain * (
            1 - (t[mask_release] - r_start) / release
        ).astype(np.float32)
    else:
        env[mask_release] = sustain

    return env.astype(np.float32)


# ------------------------------
# Audio generator 1: long drifting sines
# ------------------------------
def generate_long_sines(sr=44100, duration=5.0, mode="mono"):
    """
    Long, slowly drifting sines.
    mode:
        "mono"   -> (N,)
        "stereo" -> (N, 2) with two independent mono versions
    """
    n = int(sr * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)

    # Three slowly drifting sine waves
    freqs = [
        (110.0, 140.0),  # start, end
        (220.0, 200.0),
        (330.0, 360.0),
    ]

    mode = str(mode).lower()
    if mode not in ("mono", "stereo"):
        raise ValueError("mode must be 'mono' or 'stereo'")

    if mode == "mono":
        sig = np.zeros(n, dtype=np.float32)
        for f0, f1 in freqs:
            f_t = f0 + (f1 - f0) * (t / duration)
            phase = np.cumsum(2.0 * np.pi * f_t / sr)
            sig += 0.33 * np.sin(phase).astype(np.float32)

        env = adsr_envelope(n, sr, attack=0.2, decay=0.4, sustain=0.5, release=0.4)
        return (sig * env).astype(np.float32)

    # stereo: two independent mono layers
    left = generate_long_sines(sr=sr, duration=duration, mode="mono")
    right = generate_long_sines(sr=sr, duration=duration, mode="mono")
    return np.stack([left, right], axis=1).astype(np.float32)


# ------------------------------
# Audio generator 2: plucky impulses
# ------------------------------
def generate_plucks(sr=44100, duration=5.0, mode="mono"):
    """
    Sparse plucky impulses, decaying quickly.

    mode:
        "mono"   -> (N,)
        "stereo" -> (N, 2) with two independent mono versions
    """
    n = int(sr * duration)
    mode = str(mode).lower()
    if mode not in ("mono", "stereo"):
        raise ValueError("mode must be 'mono' or 'stereo'")

    if mode == "stereo":
        left = generate_plucks(sr=sr, duration=duration, mode="mono")
        right = generate_plucks(sr=sr, duration=duration, mode="mono")
        return np.stack([left, right], axis=1).astype(np.float32)

    # mono
    sig = np.zeros(n, dtype=np.float32)

    # impulse times (random but spaced)
    rng = np.random.default_rng(0)

    base_step = max(1, sr // 8)      # how often we *could* place an impulse
    length = int(0.2 * sr)           # nominal tail length (20% of a second)

    # Only allow positions where we have enough room for the tail
    valid_positions = np.arange(0, n - length, base_step)
    if valid_positions.size == 0:
        return sig

    positions = rng.choice(
        valid_positions,
        size=min(12, len(valid_positions)),
        replace=False,
    )

    for p in positions:
        # Here we know p + length <= n, so full tail fits
        tail = np.arange(length, dtype=np.float32)
        env = np.exp(-8.0 * tail / length).astype(np.float32)  # quick decay
        amp = float(rng.uniform(0.5, 1.0))
        sig[p : p + length] += env * amp

    return sig.astype(np.float32)


# ------------------------------
# Image generator: checkerboard
# ------------------------------
def generate_checkerboard(W=256, H=256, tiles=8):
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)
    board = ((xx // (W // tiles) + yy // (H // tiles)) % 2) * 255
    return board.astype(np.uint8)


# ------------------------------
# Image generator: gradients
# ------------------------------
def generate_gradients(W=256, H=256):
    x = np.linspace(0, 255, W, dtype=np.float32)
    y = np.linspace(0, 255, H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    img = np.stack(
        [
            xx,          # Red   = horizontal gradient
            yy,          # Green = vertical gradient
            (xx + yy) / 2.0,  # Blue  = diagonal blend
        ],
        axis=-1,
    )

    return img.astype(np.uint8)


# ------------------------------
# Image generator: radial gradient
# ------------------------------
def generate_radial(W=256, H=256):
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    img = (1.0 - np.clip(r, 0.0, 1.0)) * 255.0
    return img.astype(np.uint8)


# ------------------------------
# Main runner
# ------------------------------
def main(out_folder="samples/input/test_assets"):
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)

    # Audio
    print("Generating audio...")
    sr = 44100

    long_sines = generate_long_sines(sr=sr, duration=20.0, mode="mono")
    sf.write(out / "audio_long_sines.wav", long_sines, sr)

    plucks = generate_plucks(sr=sr, duration=13.3, mode="stereo")
    sf.write(out / "audio_plucks.wav", plucks, sr)

    # Images
    print("Generating images...")

    Image.fromarray(generate_checkerboard()).save(out / "img_checker.png")
    Image.fromarray(generate_gradients()).save(out / "img_gradients.png")
    Image.fromarray(generate_radial()).save(out / "img_radial.png")

    print(f"Done! Assets stored in: {out.resolve()}")


if __name__ == "__main__":
    main()
