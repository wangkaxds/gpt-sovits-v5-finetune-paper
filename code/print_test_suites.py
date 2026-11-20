import csv
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).resolve().parent.parent / "data" / "all_suite_summary.csv", encoding="utf-8")))
for r in rows:
    if r["split"] == "test":
        print(f"{r['suite']:28s} seed={r['seed']:>2} {r['tag']:22s} n={r['n']:>2} sv={r['sv']} amp={r['amp48']} asr={r['asr']} asr_ok={r['asr_ok']}")
