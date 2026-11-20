import csv
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).resolve().parent.parent / "data" / "all_suite_summary.csv", encoding="utf-8")))
for r in rows:
    if r["split"] == "val" and r["seed"] == "1":
        print(f"{r['suite']:28s} {r['tag']:22s} n={r['n']:>2} sv={r['sv']} amp={r['amp48']} cpp={r['cpp']} hnr={r['hnr']} asr={r['asr']}")
