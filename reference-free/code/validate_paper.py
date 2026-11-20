"""Validate the public paper data and documented paths.

Checks:
  1. documented files exist;
  2. per-sentence CSVs recompute the reported means;
  3. real-speaker baseline / pairwise matrices are consistent;
  4. score JSON -> CSV, provenance, splits;
  5. no absolute Windows paths in published text/data.

It cannot rerun inference or retrain; those need private audio, V5 and weights.
"""
import csv
import json
import re
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


paper = (BASE / "paper.md").read_text(encoding="utf-8")
readme = (BASE / "README.md").read_text(encoding="utf-8")

for f in ["fig1_pipeline.png", "fig2_ladder.png", "fig3_bands.png",
          "fig4_open.png", "fig5_decision.png"]:
    check(f"figure exists: {f}", (BASE / "figures" / f).exists())
    check(f"figure referenced: {f}", f"figures/{f}" in paper)
for f in ["paper.pdf", "paper.html", "infer.py", "code/infer_noref.py",
          "code/c1_master.py", "code/postprocess_noref.py",
          "data/unseen20.txt", "data/split_ids.csv", "data/provenance.csv",
          "data/open20_score.json", "data/c1m_val20_score.json",
          "data/real_baseline.json", "data/real_pairwise20.csv"]:
    check(f"documented file exists: {f}", (BASE / f).exists())

ex = load("experiments.csv")
by = {r["id"]: r for r in ex}
for required in ["E2b", "E10", "E11", "E12", "E13", "E14", "E15a", "E15b",
                 "E16", "E17", "E18", "C1M-val20", "C1M-50", "Open20"]:
    check(f"experiment exists: {required}", required in by)
check("E2b val20 SV", approx(by["E2b"]["sv_mean"], 0.8063, 5e-4), by["E2b"]["sv_mean"])
check("E2b val20 CER", approx(by["E2b"]["cer_mean"], 0.3268, 5e-4), by["E2b"]["cer_mean"])
check("E1 CER", approx(by["E1"]["cer_mean"], 0.8733, 5e-4), by["E1"]["cer_mean"])
check("C1M val20 SV", approx(by["C1M-val20"]["sv_mean"], 0.7909, 5e-4), by["C1M-val20"]["sv_mean"])
check("C1M val20 CER", approx(by["C1M-val20"]["cer_mean"], 0.3523, 5e-4), by["C1M-val20"]["cer_mean"])
check("E2b test50 SV", approx(by["E2b-test50x3"]["sv_mean"], 0.8258, 5e-4), by["E2b-test50x3"]["sv_mean"])
check("C1M-50 SV", approx(by["C1M-50"]["sv_mean"], 0.8128, 5e-4), by["C1M-50"]["sv_mean"])
check("Open20 SV", approx(by["Open20"]["sv_mean"], 0.7332, 5e-4), by["Open20"]["sv_mean"])
check("Open20 CER", approx(by["Open20"]["cer_mean"], 0.0754, 5e-4), by["Open20"]["cer_mean"])
check("E17 SV", approx(by["E17"]["sv_mean"], 0.5410, 5e-4), by["E17"]["sv_mean"])
check("E18 SV", approx(by["E18"]["sv_mean"], 0.5448, 5e-4), by["E18"]["sv_mean"])

op = load("open_domain.csv")
check("open n=20", len(op) == 20, len(op))
cer = np.array([float(r["cer"]) for r in op])
sv = np.array([float(r["sv"]) for r in op])
check("open CER mean", approx(cer.mean(), 0.0754, 5e-4), cer.mean())
check("open SV mean", approx(sv.mean(), 0.7332, 5e-4), sv.mean())
check("open SV range", approx(sv.min(), 0.6660, 5e-4) and approx(sv.max(), 0.8600, 5e-4), (sv.min(), sv.max()))

pw = load("pairwise.csv")
check("pairwise 400", len(pw) == 400, len(pw))
m = np.zeros((20, 20))
for r in pw:
    m[int(r["row"]) - 1, int(r["col"]) - 1] = float(r["cos"])
off = m[~np.eye(20, dtype=bool)]
check("pairwise mean", approx(off.mean(), 0.8233, 5e-4), off.mean())
check("pairwise min", approx(off.min(), 0.6342, 5e-4), off.min())
check("pairwise max", approx(off.max(), 0.9146, 5e-4), off.max())
check("pairwise symmetric", np.allclose(m, m.T, atol=1e-5), "not symmetric")
check("pairwise diagonal", np.allclose(np.diag(m), 1.0, atol=1e-4), np.diag(m))

real = load("real_test50.csv")
check("real n=50", len(real) == 50, len(real))
real_sv = np.array([float(r["target_sv"]) for r in real])
check("real target mean", approx(real_sv.mean(), 0.8398, 5e-4), real_sv.mean())
rp = load("real_pairwise.csv")
check("real pairwise 2500", len(rp) == 2500, len(rp))
ids = [r["id"] for r in real]
idx = {x: i for i, x in enumerate(ids)}
rm = np.zeros((50, 50))
for r in rp:
    rm[idx[r["row"]], idx[r["col"]]] = float(r["cos"])
roff = rm[~np.eye(50, dtype=bool)]
check("real pairwise mean", approx(roff.mean(), 0.7684, 5e-4), roff.mean())
check("real pairwise min", approx(roff.min(), 0.4681, 5e-4), roff.min())
rb = json.loads((DATA / "real_baseline.json").read_text(encoding="utf-8"))
check("real baseline JSON mean", approx(rb.get("pairwise_mean"), 0.7684, 5e-4), rb.get("pairwise_mean"))
check("real baseline n20 mean", approx(rb.get("n20", {}).get("mean"), 0.7530, 5e-4), rb.get("n20"))
rp20 = load("real_pairwise20.csv")
check("real pairwise20 400", len(rp20) == 400, len(rp20))
r20 = np.zeros((20, 20))
for r in rp20:
    r20[idx[r["row"]], idx[r["col"]]] = float(r["cos"])
r20off = r20[~np.eye(20, dtype=bool)]
check("real pairwise20 mean", approx(r20off.mean(), 0.7530, 5e-4), r20off.mean())

sp = load("splits.csv")
check("split rows", len(sp) == 3, len(sp))
counts = {r["split"]: int(r["n"]) for r in sp}
check("split counts", counts == {"train": 1100, "val": 50, "test": 50}, counts)
check("split duration", approx(sum(float(r["duration_sec"]) for r in sp) / 60, 102.10, 0.01),
      sum(float(r["duration_sec"]) for r in sp) / 60)
sid = load("split_ids.csv")
check("split_ids total", len(sid) == 1200, len(sid))
check("split_ids counts", {s: sum(1 for r in sid if r["split"] == s) for s in ("train", "val", "test")}
      == {"train": 1100, "val": 50, "test": 50})
test_ids = {r["id"] for r in sid if r["split"] == "test"}
real_ids = {r["id"] for r in real}
check("real ids = test ids", real_ids == test_ids, len(real_ids ^ test_ids))

prov = load("provenance.csv")
prov_ids = {r["id"] for r in prov}
for required in ["E10-E18", "C1M-val20", "E2b-test50x3", "C1M-50", "Open20", "Real-test50",
                 "Ref-control", "Shelf-B-50"]:
    check(f"provenance row: {required}", required in prov_ids)
c1 = json.loads((DATA / "c1m_val20_score.json").read_text(encoding="utf-8"))
check("c1m val20 score n=20", c1.get("n") == 20 and len(c1.get("rows", [])) == 20, c1.get("n"))
check("c1m val20 score SV", approx(c1.get("sv_mean"), 0.7909, 5e-4), c1.get("sv_mean"))
check("c1m val20 score CER", approx(c1.get("cer_mean"), 0.3523, 5e-4), c1.get("cer_mean"))
op_score = json.loads((DATA / "open20_score.json").read_text(encoding="utf-8"))["rows"]
check("open20 score rows", len(op_score) == 20, len(op_score))
check("open20 CSV/JSON CER", all(approx(float(r["cer"]), s["cer"], 5e-5) for r, s in zip(op, op_score)), "mismatch")
check("open20 CSV/JSON SV", all(approx(float(r["sv"]), s["sv_cos"], 5e-5) for r, s in zip(op, op_score)), "mismatch")

for phrase in ["0.8398", "0.7332", "0.8233", "0.7684", "0.7909", "0.5410", "0.5448"]:
    check(f"paper contains {phrase}", phrase in paper)
check("paper mentions ref_enc", "ref_enc" in paper)
check("paper marks L2 partial", "L2" in paper)
check("README uses reference-free layout", "cd reference-free" in readme)
check("README no paper/ command", "python paper/code/" not in readme)

abs_hits = []
files = ["paper.md", "README.md"]
files += [f"code/{p.name}" for p in (BASE / "code").glob("*.py") if p.name != "validate_paper.py"]
files += [f"data/{p.name}" for p in (BASE / "data").glob("*.json")]
files += [f"data/{p.name}" for p in (BASE / "data").glob("*.csv")]
files += [f"data/{p.name}" for p in (BASE / "data").glob("*.txt")]
for rel in files:
    txt = (BASE / rel).read_text(encoding="utf-8", errors="ignore")
    if re.search(r"[A-Za-z]:\\", txt):
        abs_hits.append(rel)
check("no absolute Windows path in published text/data", not abs_hits, abs_hits)

bad = []
for p in BASE.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".pth", ".pt", ".ckpt", ".safetensors"}:
        bad.append(str(p.relative_to(BASE)))
check("no audio or weights", not bad, bad)
check("no __pycache__", not list(BASE.rglob("__pycache__")), "pycache present")
check("paper.pdf exists", (BASE / "paper.pdf").exists())
check("paper.html exists", (BASE / "paper.html").exists())

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("FAIL:", f)
    sys.exit(1)
