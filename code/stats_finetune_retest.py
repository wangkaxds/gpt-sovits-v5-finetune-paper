import csv, numpy as np
from math import sqrt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(BASE / "data" / "finetune_retest_summary.csv", encoding="utf-8")))
d_sv = np.array([float(r["delta_sv"]) for r in rows])
d_asr = np.array([float(r["delta_asr"]) for r in rows])

def report(name, d):
    mean = d.mean()
    sd = d.std(ddof=1)
    t = mean / (sd / sqrt(len(d)))
    print(f"{name}: mean={mean:+.4f} sd={sd:.4f} t={t:.2f} n={len(d)}")

report("SV delta (vo16 - v5dev)", d_sv)
report("ASR delta (vo16 - v5dev)", d_asr)
