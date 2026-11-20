"""Fixed C1M low-cut/compressor/high-shelf mastering for reference-free E2b.

The FIR was exported from the audio analysis; it is static and reads no
runtime reference audio. The filter is data/c1_fir.npy.
"""
import math
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import fftconvolve

from postprocess_noref import shelf

PROJ = Path(__file__).resolve().parents[1]
H = np.load(os.environ.get("C1_FIR", str(PROJ / "data" / "c1_fir.npy"))).astype(np.float32)
HIGH_SHELF_DB = 0.75
HIGH_SHELF_HZ = 2000.0
TARGET_RMS = 0.1228
PEAK_CEIL = 0.98
THRESHOLD_DB = -10.0
RATIO = 1.5
ATTACK_MS = 5.0
RELEASE_MS = 100.0
KNEE_DB = 4.0


def _compress(w, sr):
    n = max(1, int(ATTACK_MS / 1000 * sr))
    env = maximum_filter1d(np.abs(w), size=2 * n + 1)
    lvl = 20 * np.log10(np.maximum(env, 1e-9))
    over = lvl - THRESHOLD_DB
    slope = 1.0 - 1.0 / RATIO
    gr = np.where(
        over > KNEE_DB / 2,
        (over - KNEE_DB / 2) * slope,
        np.where(
            over > -KNEE_DB / 2,
            (over + KNEE_DB / 2) ** 2 / (2 * KNEE_DB) * slope,
            0.0,
        ),
    )
    gain = 10 ** (-gr / 20)
    a = math.exp(-1.0 / (RELEASE_MS / 1000 * sr))
    out = np.empty_like(gain)
    cur = 1.0
    for i in range(len(gain)):
        cur = gain[i] if gain[i] < cur else a * cur + (1 - a) * gain[i]
        out[i] = cur
    return w * out


def process(w, sr):
    """Return (wav float32, metadata dict)."""
    w = np.asarray(w, dtype=np.float32)
    y = fftconvolve(w, H, mode="same").astype(np.float32)
    y = _compress(y, int(sr))
    y = shelf(y, int(sr), HIGH_SHELF_HZ, HIGH_SHELF_DB, "high")
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)
    peak = float(np.max(np.abs(y)) + 1e-12)
    gain = min(TARGET_RMS / rms, PEAK_CEIL / peak)
    y = np.clip(y * gain, -0.999, 0.999).astype(np.float32)
    return y, {
        "mode": "C1_lowcut_plus_high_shelf",
        "fir_taps": int(H.size),
        "target_rms": TARGET_RMS,
        "peak_ceil": PEAK_CEIL,
        "high_shelf": {"hz": HIGH_SHELF_HZ, "db": HIGH_SHELF_DB},
        "compressor": {
            "threshold_db": THRESHOLD_DB,
            "ratio": RATIO,
            "attack_ms": ATTACK_MS,
            "release_ms": RELEASE_MS,
            "knee_db": KNEE_DB,
        },
        "gain": float(gain),
    }
