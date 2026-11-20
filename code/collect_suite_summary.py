import glob, json, os, csv
from pathlib import Path

BASE = os.environ.get("V5_EVAL_ROOT")
if not BASE:
    raise SystemExit("Set V5_EVAL_ROOT to the local v5_ft_eval directory")
OUT = str(Path(__file__).resolve().parent.parent / "data" / "all_suite_summary.csv")
rows = []
for p in glob.glob(os.path.join(BASE, "suite_*", "*_suite.json")):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    suite = os.path.basename(os.path.dirname(p))
    split = d.get("split")
    seed = d.get("seed")
    for tag, s in d.get("summary", {}).items():
        rows.append({
            "suite": suite, "split": split, "seed": seed, "tag": tag,
            "n": s.get("n"), "sv": s.get("sv"), "amp48": s.get("amp48"),
            "cpp": s.get("cpp"), "hnr": s.get("hnr"), "jitter": s.get("jitter"),
            "asr": s.get("asr_sim"), "asr_ok": s.get("asr_ok"),
            "clip": s.get("clip"), "dur": s.get("dur"),
        })
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("rows", len(rows), "suites", len({r["suite"] for r in rows}))
