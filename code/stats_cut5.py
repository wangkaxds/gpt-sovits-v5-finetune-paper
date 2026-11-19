import csv, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(BASE / "data" / "cut5_paired_seed1.csv", encoding="utf-8")))
rng = np.random.default_rng(20261120)

for key in ("sv", "asr", "amp"):
    a = np.array([float(r[f"cut0_{key}"]) for r in rows])
    b = np.array([float(r[f"cut5_{key}"]) for r in rows])
    d = b - a
    idx = rng.integers(0, len(d), size=(20000, len(d)))
    ms = d[idx].mean(1)
    lo, hi = np.percentile(ms, [2.5, 97.5])
    print(f"{key}: delta={d.mean():+.4f} CI95=[{lo:+.4f},{hi:+.4f}] P>0={(ms > 0).mean():.3f} P<0={(ms < 0).mean():.3f}")
