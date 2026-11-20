import json, csv, os
from pathlib import Path

BASE = os.environ.get("V5_EVAL_ROOT")
if not BASE:
    raise SystemExit("Set V5_EVAL_ROOT to the local v5_ft_eval directory")
OUT = str(Path(__file__).resolve().parent.parent / "data")
os.makedirs(OUT, exist_ok=True)

def rows(path, tag):
    d = json.load(open(path, encoding="utf-8"))
    return {r["i"]: r for r in d["rows"] if r["tag"] == tag}

# 1. cut5 paired seed1, 30 sentences
cut0 = rows(os.path.join(BASE, "suite_eval30_turbo8", "test_seed1_suite.json"), "turbo8")
cut5 = rows(os.path.join(BASE, "suite_cut5_30", "test_seed1_suite.json"), "turbo8")
with open(os.path.join(OUT, "cut5_paired_seed1.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sentence", "cut0_sv", "cut5_sv", "cut0_asr", "cut5_asr", "cut0_amp", "cut5_amp"])
    for i in sorted(set(cut0) & set(cut5)):
        a, b = cut0[i], cut5[i]
        w.writerow([i, a["sv"], b["sv"], a["asr_sim"], b["asr_sim"], a["amp48"], b["amp48"]])

# 2. cut5 seed summary (5 seeds; cut0 1-3)
with open(os.path.join(OUT, "cut5_seed_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["seed", "cut0_sv", "cut5_sv", "cut0_asr", "cut5_asr"])
    w.writerow([1, 0.8868, 0.9062, 0.9238, 0.9188])
    w.writerow([2, 0.8785, 0.9042, 0.9230, 0.9230])
    w.writerow([3, 0.8779, 0.9060, 0.9140, 0.9210])
    w.writerow([4, "", 0.9024, "", 0.9330])
    w.writerow([5, "", 0.9018, "", 0.9220])

# 3. fine-tune retest seed summary
with open(os.path.join(OUT, "finetune_retest_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["seed", "v5dev_sv", "vo16_sv", "delta_sv", "v5dev_asr", "vo16_asr", "delta_asr"])
    w.writerow([1, 0.8818, 0.8816, -0.0003, 0.9080, 0.9123, 0.0043])
    w.writerow([2, 0.8758, 0.8763, 0.0005, 0.9351, 0.9230, -0.0121])
    w.writerow([3, 0.8744, 0.8729, -0.0015, 0.9204, 0.9191, -0.0013])

# 4. turbo8 deltas
with open(os.path.join(OUT, "turbo8_seed_deltas.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["seed", "delta_sv", "delta_asr"])
    for s, sv, asr in [(1, 0.0044, 0.0000), (2, 0.0013, -0.0104), (3, 0.0089, -0.0027), (4, 0.0061, 0.0007), (5, 0.0053, -0.0103)]:
        w.writerow([s, sv, asr])

print("csv done")
