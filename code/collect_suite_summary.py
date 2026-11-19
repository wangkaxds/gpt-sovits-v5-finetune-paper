import glob, json, os, csv

BASE = r"D:\杂活\甜蜜女友3\v5_ft_eval"
OUT = r"D:\杂活\甜蜜女友3\v5_paper\data\all_suite_summary.csv"
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
