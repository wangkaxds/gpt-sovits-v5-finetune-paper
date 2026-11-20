"""Static post-processing for no-reference output.

Fixed corpus-level EQ/loudness, not a per-utterance reference. Tests how much
of the "muffled / quiet" defect is mastering rather than identity.
"""
import argparse
import json
import math

import numpy as np
import soundfile as sf
from scipy.signal import lfilter


def shelf(x, sr, freq, gain_db, kind="high", q=0.707):
    a = 10 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "high":
        b0 = a * ((a + 1) + (a - 1) * cw + 2 * math.sqrt(a) * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - 2 * math.sqrt(a) * alpha)
        a0 = (a + 1) - (a - 1) * cw + 2 * math.sqrt(a) * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - 2 * math.sqrt(a) * alpha
    else:
        b0 = a * ((a + 1) - (a - 1) * cw + 2 * math.sqrt(a) * alpha)
        b1 = 2 * a * ((a - 1) - (a + 1) * cw)
        b2 = a * ((a + 1) - (a - 1) * cw - 2 * math.sqrt(a) * alpha)
        a0 = (a + 1) + (a - 1) * cw + 2 * math.sqrt(a) * alpha
        a1 = -2 * ((a - 1) + (a + 1) * cw)
        a2 = (a + 1) + (a - 1) * cw - 2 * math.sqrt(a) * alpha
    b = np.array([b0, b1, b2]) / a0
    aa = np.array([1.0, a1 / a0, a2 / a0])
    return lfilter(b, aa, x).astype(np.float32)


def process(w, sr, high_db, low_db, target_rms):
    w = shelf(w, sr, 500.0, low_db, "low")
    w = shelf(w, sr, 2000.0, high_db, "high")
    rms = float(np.sqrt(np.mean(w * w)) + 1e-12)
    gain = target_rms / rms
    peak = float(np.max(np.abs(w)) + 1e-12)
    gain = min(gain, 0.98 / peak)
    return np.clip(w * gain, -1.0, 1.0), gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--high", type=float, default=3.8)
    ap.add_argument("--low", type=float, default=-3.5)
    ap.add_argument("--rms", type=float, default=0.115)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((src / "results.json").read_text(encoding="utf-8"))
    rows = []
    gains = []
    for item in old:
        p = Path(item["out"])
        w, sr = sf.read(str(p))
        if w.ndim > 1:
            w = w.mean(1)
        w, g = process(w.astype(np.float32), int(sr), args.high, args.low, args.rms)
        gains.append(g)
        dst = out / p.name
        sf.write(str(dst), w, int(sr))
        meta = dict(item)
        meta.update(
            {
                "out": str(dst),
                "samples": int(len(w)),
                "duration_sec": round(len(w) / sr, 4),
                "rms": float(np.sqrt(np.mean(w.astype(np.float64) ** 2))),
                "peak": float(np.max(np.abs(w.astype(np.float64)))),
                "postprocess": {
                    "high_db": args.high,
                    "low_db": args.low,
                    "target_rms": args.rms,
                    "applied_gain": round(g, 4),
                },
            }
        )
        rows.append(meta)
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "n": len(rows), "gain_mean": round(float(np.mean(gains)), 3)}))


if __name__ == "__main__":
    main()
