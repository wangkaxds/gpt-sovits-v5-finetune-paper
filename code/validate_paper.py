"""Consistency checks for the public paper repository.

This validates statistical summaries that can be recomputed from public CSVs,
paper/README claims that have known corrected values, figure references, and
repository hygiene. It does not regenerate audio or retrain models.

Run from the repository root:
    python code/validate_paper.py
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FAIL = []
PASS = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"PASS {name}")
    else:
        FAIL.append(f"{name}: {detail}")
        print(f"FAIL {name}: {detail}")


def load_csv(name):
    return list(csv.DictReader(open(DATA / name, encoding="utf-8")))


def approx(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


# --- statistical recomputation -------------------------------------------------
cut5 = load_csv("cut5_paired_seed1.csv")
check("cut5 sentence count", len(cut5) == 30, len(cut5))
d_sv = np.array([float(r["cut5_sv"]) - float(r["cut0_sv"]) for r in cut5])
d_asr = np.array([float(r["cut5_asr"]) - float(r["cut0_asr"]) for r in cut5])
rng = np.random.default_rng(20261120)
idx = rng.integers(0, len(d_sv), size=(20000, len(d_sv)))
ms = d_sv[idx].mean(1)
lo, hi = np.percentile(ms, [2.5, 97.5])
check("cut5 SV mean", approx(d_sv.mean(), 0.0194, 2e-4), d_sv.mean())
check("cut5 SV CI", approx(lo, 0.0097, 5e-4) and approx(hi, 0.0309, 5e-4), (lo, hi))
check("cut5 ASR mean", approx(d_asr.mean(), -0.0051, 2e-4), d_asr.mean())
mde_cut5 = 2.8 * d_sv.std(ddof=1) / np.sqrt(len(d_sv))
check("cut5 MDE", approx(mde_cut5, 0.0156, 5e-4), mde_cut5)

seed_rows = [r for r in load_csv("cut5_seed_summary.csv") if r["cut0_sv"]]
c0 = np.array([float(r["cut0_sv"]) for r in seed_rows])
c5 = np.array([float(r["cut5_sv"]) for r in seed_rows])
cut5_test = stats.ttest_rel(c5, c0)
check("cut5 3-seed paired p", approx(cut5_test.pvalue, 0.0111, 5e-4), cut5_test.pvalue)
check("cut5 3-seed delta", approx((c5 - c0).mean(), 0.0244, 2e-4), (c5 - c0).mean())

fin = load_csv("finetune_retest_summary.csv")
fin_sv = np.array([float(r["delta_sv"]) for r in fin])
fin_asr = np.array([float(r["delta_asr"]) for r in fin])
fin_t = stats.ttest_1samp(fin_sv, 0)
check("vo16 SV mean", approx(fin_sv.mean(), -0.0004, 5e-5), fin_sv.mean())
check("vo16 SV t", approx(fin_t.statistic, -0.75, 0.02), fin_t.statistic)

turbo = load_csv("turbo8_seed_deltas.csv")
t_sv = np.array([float(r["delta_sv"]) for r in turbo])
t_test = stats.ttest_1samp(t_sv, 0)
check("turbo SV mean", approx(t_sv.mean(), 0.0052, 2e-5), t_sv.mean())
check("turbo SV t", approx(t_test.statistic, 4.22, 0.02), t_test.statistic)

loader = load_csv("loader_fix_paired_seed1.csv")
check("loader rows", len(loader) == 90, len(loader))
audit = load_csv("lora_checkpoint_audit.csv")
check("audit rows", len(audit) == 19, len(audit))
check("audit unavailable routes",
      {r["route"]: r["status"] for r in audit}["last5"] == "DELETED_UNVERIFIABLE"
      and {r["route"]: r["status"] for r in audit}["syn"] == "NO_CHECKPOINT_UNVERIFIABLE",
      "status mismatch")
by_tag = {}
for row in loader:
    by_tag.setdefault(row["tag"], []).append(row)
base = {r["sentence"]: r for r in by_tag["v5dev"]}
for tag, expected_p, expected_mean in [("ada8_s300", 0.9055, 0.00018),
                                       ("all8_s1200", 0.0084, 0.00100)]:
    rows = {r["sentence"]: r for r in by_tag[tag]}
    keys = sorted(set(base) & set(rows))
    diff = np.array([float(rows[k]["sv"]) - float(base[k]["sv"]) for k in keys])
    test = stats.ttest_1samp(diff, 0)
    check(f"{tag} SV mean", approx(diff.mean(), expected_mean, 2e-5), diff.mean())
    check(f"{tag} SV p", approx(test.pvalue, expected_p, 5e-4), test.pvalue)
    if tag == "all8_s1200":
        mde = 2.8 * diff.std(ddof=1) / np.sqrt(len(diff))
        check("all8 MDE", approx(mde, 0.0010, 5e-5), mde)
        rng2 = np.random.default_rng(20261120)
        bms = diff[rng2.integers(0, len(diff), size=(20000, len(diff)))].mean(1)
        blo, bhi = np.percentile(bms, [2.5, 97.5])
        check("all8 CI", approx(blo, 0.0003, 5e-5) and approx(bhi, 0.0017, 5e-5), (blo, bhi))

# --- paper / README consistency ------------------------------------------------
paper = (BASE / "paper.md").read_text(encoding="utf-8")
readme = (BASE / "README.md").read_text(encoding="utf-8")
for text, name in [(paper, "paper"), (readme, "README")]:
    check(f"{name}: no unsupported cost claim", "成本不变" not in text)
    check(f"{name}: all8 p present", "p=.0084" in text or "p=.008" in text)
    check(f"{name}: duration confound present", "4.324" in text or "时长" in text)

check("paper: turbo t corrected", "t=4.22" in paper or "4.22" in paper)
check("paper: MDE not threshold", "MDE 不是" in paper)
check("paper: no absolute 30-sentence claim", "30 句能排除" not in paper)
check("paper: last5 unverifiable", "DELETED_UNVERIFIABLE" in paper or "无法验证" in paper)
check("paper: no stale retest status", "修复后的音频级重测未完成" not in paper)

fix_doc = (BASE / "code" / "TTS_LORA_LOADER_FIX.md").read_text(encoding="utf-8")
check("loader doc: retest status updated", "未完成" not in fix_doc)
check("loader doc: all8 target exact", "ff.ff.0.0" in fix_doc)
check("loader doc: last5 unavailable", "last5" in fix_doc and "deleted" in fix_doc.lower())

# --- figures / references ------------------------------------------------------
for rel in sorted(set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", paper))):
    check(f"figure exists: {rel}", (BASE / rel).exists(), rel)

# --- repository hygiene --------------------------------------------------------
bad_ext = {".wav", ".flac", ".mp3", ".pth", ".ckpt", ".safetensors", ".zip"}
bad_files = [p for p in BASE.rglob("*") if p.is_file() and p.suffix.lower() in bad_ext]
check("no audio/weights", not bad_files, [str(p) for p in bad_files[:5]])

code_texts = []
for p in (BASE / "code").glob("*.py"):
    text = p.read_text(encoding="utf-8", errors="replace")
    code_texts.append((p.name, text))
    if re.search(r"[A-Z]:\\", text):
        check(f"{p.name}: no absolute Windows path", False, "found drive path")
        break
else:
    check("all code: no absolute Windows path", True)

# --- output --------------------------------------------------------------------
print(f"\n{PASS.__len__()} passed, {len(FAIL)} failed")
if FAIL:
    for item in FAIL:
        print(" -", item)
    sys.exit(1)
