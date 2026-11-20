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
check("cut5 sign/Wilcoxon sensitivity",
      approx(stats.wilcoxon(c5 - c0).pvalue, 0.25, 1e-3), stats.wilcoxon(c5 - c0).pvalue)

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
check("turbo sign sensitivity",
      approx(stats.binomtest(int((t_sv > 0).sum()), len(t_sv), 0.5).pvalue, 0.0625, 5e-4),
      int((t_sv > 0).sum()))
check("vo16 paired t p", approx(fin_t.pvalue, 0.5336, 5e-4), fin_t.pvalue)

loader = load_csv("loader_fix_paired_seed1.csv")
check("loader rows", len(loader) == 90, len(loader))
audit = load_csv("lora_checkpoint_audit.csv")
check("audit rows", len(audit) == 19, len(audit))
check("audit unavailable routes",
      {r["route"]: r["status"] for r in audit}["last5"] == "DELETED_UNVERIFIABLE"
      and {r["route"]: r["status"] for r in audit}["syn"] == "NO_CHECKPOINT_UNVERIFIABLE",
      "status mismatch")

evidence = load_csv("evidence_matrix.csv")
route_ids = {r["route"] for r in load_csv("route_ledger.csv")}
evidence_ids = {r["route"] for r in evidence}
check("evidence matrix rows", len(evidence) == 19, len(evidence))
check("evidence matrix routes match ledger", route_ids == evidence_ids,
      sorted(route_ids ^ evidence_ids))
ledger = load_csv("route_ledger.csv")
bad_classes = [r["route"] for r in ledger
               if not r.get("result_class") or
               any(c not in {"A", "B", "C", "D", "E", "F"} for c in r["result_class"].split(","))]
check("route classifications present", len(ledger) == 19 and not bad_classes, bad_classes)
retracted = load_csv("retracted_claims.csv")
check("retracted claim classified F",
      len(retracted) == 1 and retracted[0]["claim_id"] == "BEST5"
      and retracted[0]["status"] == "retracted", retracted)
matrix_cols = ["training_log", "full_config", "checkpoint", "runtime_load_verify",
               "val", "test", "multi_seed", "stat_test", "human_listening"]
allowed = {"yes", "partial", "structural", "no"}
for col in matrix_cols:
    bad = [r["route"] for r in evidence if r[col] not in allowed]
    check(f"evidence matrix values: {col}", not bad, bad)
check("evidence matrix last5 not claimed runtime",
      {r["route"]: r["runtime_load_verify"] for r in evidence}["last5"] == "no")
check("evidence matrix syn no checkpoint",
      {r["route"]: r["checkpoint"] for r in evidence}["syn"] == "no")

merge = load_csv("lora_merge_verification.csv")
check("merge summary rows", len(merge) == 5, len(merge))
check("merge summary all pass", all(r["passed"] == "True" for r in merge), merge)
check("merge summary finite", all(r["all_finite"] == "True" for r in merge), merge)
check("merge summary untouched unchanged",
      all(float(r["untouched_max_abs_delta"]) == 0.0 for r in merge), merge)
check("merge summary formula tolerance",
      all(float(r["max_abs_error"]) < 1e-6 and float(r["max_relative_error"]) < 1e-3
          for r in merge), merge)
module = load_csv("lora_merge_module_audit.csv")
check("merge module rows", len(module) == 330, len(module))
check("merge module all pass", all(r["passed"] == "True" for r in module),
      [r for r in module if r["passed"] != "True"][:2])
check("merge module finite", all(r["finite"] == "True" for r in module),
      [r for r in module if r["finite"] != "True"][:2])

index = load_csv("result_index.csv")
check("result index rows", len(index) == 15, len(index))
check("result index unique ids", len({r["result_id"] for r in index}) == len(index))
missing_files = []
missing_scripts = []
for row in index:
    for rel in row["data_files"].split(";"):
        if rel and not (BASE / rel).exists():
            missing_files.append(rel)
    for rel in row["script"].split(";"):
        if rel and not (BASE / rel).exists():
            missing_scripts.append(rel)
check("result index data files exist", not missing_files, missing_files)
check("result index scripts exist", not missing_scripts, missing_scripts)
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
        check("all8 Wilcoxon", approx(stats.wilcoxon(diff).pvalue, 0.0081, 5e-4),
              stats.wilcoxon(diff).pvalue)
        check("all8 sign test", approx(stats.binomtest(int((diff > 0).sum()), len(diff), 0.5).pvalue,
                                       0.0428, 5e-4), int((diff > 0).sum()))
        check("all8 sign count", int((diff > 0).sum()) == 21, int((diff > 0).sum()))

# --- paper / README consistency ------------------------------------------------
paper = (BASE / "paper.md").read_text(encoding="utf-8")
readme = (BASE / "README.md").read_text(encoding="utf-8")
html = (BASE / "paper.html").read_text(encoding="utf-8")
for text, name in [(paper, "paper"), (readme, "README")]:
    check(f"{name}: no unsupported cost claim", "成本不变" not in text)
    check(f"{name}: all8 p present", "p=.0084" in text or "p=.008" in text)
    check(f"{name}: duration confound present", "4.324" in text or "时长" in text)
check("html: version synced", "v2.2" in html and "v2.1" in html)
check("html: appendix F present", "附录 F" in html and "版本、撤回" in html)

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
