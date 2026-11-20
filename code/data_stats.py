"""Compute duration statistics for the local Kan data. Requires GSV_DATA_ROOT."""
import os
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = os.environ.get("GSV_DATA_ROOT")
if not ROOT:
    raise SystemExit("Set GSV_DATA_ROOT to the directory containing kan_exp/, kan_clean_exp/, sentences*.txt")
SRC = Path(ROOT) / "kan_exp" / "5-wav32k"
CLEAN = Path(ROOT) / "kan_clean_exp" / "5-wav32k"


def stats(root, names=None):
    if names is None:
        names = [f for f in os.listdir(root) if f.endswith(".wav")]
    durs = []
    for name in names:
        try:
            durs.append(sf.info(str(Path(root) / name)).duration)
        except Exception:
            pass
    durs = np.array(durs)
    return len(durs), durs.sum() / 60, durs.mean(), np.percentile(durs, [5, 50, 95])


for label, root in [("all", SRC), ("clean", CLEAN)]:
    n, total, mean, pct = stats(root)
    print(f"{label} n={n} total={total:.1f}min mean={mean:.2f}s p5/50/95={pct.round(2)}")

for split, filename in [("test", "sentences_eval30.txt"), ("val", "sentences10.txt")]:
    path = Path(ROOT) / filename
    if path.exists():
        lines = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
        print(split, len(lines), "chars mean", np.mean([len(x) for x in lines]))
