"""Validate the paper's public CSV summaries, figures, and repository hygiene.

Run from the repository root:
    python paper/code/validate_paper.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FAIL = []
PASS = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"PASS {name}")
    else:
        FAIL.append(f"{name}: {detail}")
        print(f"FAIL {name}: {detail}")


def load(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def approx(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


# experiments
ex = load("experiments.csv")
ids = {r["id"] for r in ex}
for required in ["E1", "E2b", "E10", "E11", "E12", "E13", "E14", "E15a", "E15b",
                 "E16", "E17", "E18", "C1M-50", "Open20"]:
    check(f"experiment exists: {required}", required in ids)
by = {r["id"]: r for r in ex}
check("E2b SV", approx(by["E2b"]["sv_mean"], 0.8063, 5e-4), by["E2b"]["sv_mean"])
check("E2b test50 SV", approx(by["E2b-test50x3"]["sv_mean"], 0.8258, 5e-4), by["E2b-test50x3"]["sv_mean"])
check("E2b test50 CER median", approx(by["E2b-test50x3"]["cer_median"], 0.2692, 5e-4), by["E2b-test50x3"]["cer_median"])
check("C1M SV", approx(by["C1M-50"]["sv_mean"], 0.8128, 5e-4), by["C1M-50"]["sv_mean"])
check("C1M CER", approx(by["C1M-50"]["cer_mean"], 0.2874, 5e-4), by["C1M-50"]["cer_mean"])
check("Open20 SV", approx(by["Open20"]["sv_mean"], 0.7332, 5e-4), by["Open20"]["sv_mean"])
check("Open20 CER", approx(by["Open20"]["cer_mean"], 0.0754, 5e-4), by["Open20"]["cer_mean"])
check("E17 SV", approx(by["E17"]["sv_mean"], 0.5410, 5e-4), by["E17"]["sv_mean"])
check("E18 SV", approx(by["E18"]["sv_mean"], 0.5448, 5e-4), by["E18"]["sv_mean"])
check("E15 scale arms", approx(by["E15a"]["sv_mean"], 0.7644, 5e-4) and approx(by["E15b"]["sv_mean"], 0.7484, 5e-4),
      (by["E15a"]["sv_mean"], by["E15b"]["sv_mean"]))

# open domain
op = load("open_domain.csv")
check("open domain n", len(op) == 20, len(op))
cer = np.array([float(r["cer"]) for r in op])
sv = np.array([float(r["sv"]) for r in op])
check("open CER mean", approx(cer.mean(), 0.0754, 5e-4), cer.mean())
check("open SV mean", approx(sv.mean(), 0.7332, 5e-4), sv.mean())
check("open SV min", approx(sv.min(), 0.6660, 5e-4), sv.min())

# pairwise
pw = load("pairwise.csv")
check("pairwise 20x20", len(pw) == 400, len(pw))
m = np.zeros((20, 20))
for r in pw:
    m[int(r["row"]) - 1, int(r["col"]) - 1] = float(r["cos"])
off = m[~np.eye(20, dtype=bool)]
check("pairwise mean", approx(off.mean(), 0.8233, 5e-4), off.mean())
check("pairwise min", approx(off.min(), 0.6342, 5e-4), off.min())
check("pairwise max", approx(off.max(), 0.9146, 5e-4), off.max())
check("pairwise symmetric", np.allclose(m, m.T, atol=1e-5), "not symmetric")
check("pairwise diagonal", np.allclose(np.diag(m), 1.0, atol=1e-4), np.diag(m))

# splits
sp = load("splits.csv")
check("split row count", len(sp) == 3, len(sp))
counts = {r["split"]: int(r["n"]) for r in sp}
check("split total 1200", sum(counts.values()) == 1200, counts)
check("split counts", counts == {"train": 1100, "val": 50, "test": 50}, counts)
total_sec = sum(float(r["duration_sec"]) for r in sp)
check("split duration 102.1 min", approx(total_sec / 60, 102.10, 0.01), total_sec / 60)

# bands
bands = load("bands.csv")
b = {r["system"]: r for r in bands}
check("C1M <500", approx(b["C1M"]["ltas_lt500"], 0.1794, 5e-4), b["C1M"]["ltas_lt500"])
check("C1M 2-8k", approx(b["C1M"]["ltas_2_8k"], 0.1533, 5e-4), b["C1M"]["ltas_2_8k"])
check("C1M CPP", approx(b["C1M"]["cpp"], 0.127, 5e-4), b["C1M"]["cpp"])
check("raw <500", approx(b["raw-E2b"]["ltas_lt500"], 0.2337, 5e-4), b["raw-E2b"]["ltas_lt500"])

# paper content + figures
paper = (BASE / "paper.md").read_text(encoding="utf-8")
for f in ["fig1_pipeline.png", "fig2_ladder.png", "fig3_bands.png", "fig4_open.png", "fig5_decision.png"]:
    check(f"figure exists: {f}", (BASE / "figures" / f).exists())
    check(f"figure referenced: {f}", f"figures/{f}" in paper)
for phrase in ["0.8258", "0.7332", "0.823", "0.634", "0.5410", "0.5448", "0.0754", "0.8938"]:
    check(f"paper contains {phrase}", phrase in paper)

# hygiene: no audio / weights in paper
bad = []
for p in BASE.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".pth", ".pt", ".ckpt", ".safetensors", ".npy"}:
        bad.append(str(p.relative_to(BASE)))
check("no audio or weights", not bad, bad)
check("paper.pdf exists", (BASE / "paper.pdf").exists())
check("paper.html exists", (BASE / "paper.html").exists())

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("FAIL:", f)
    sys.exit(1)
