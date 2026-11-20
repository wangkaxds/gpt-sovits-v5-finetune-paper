import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).resolve().parent.parent / "data" / "all_suite_summary.csv", encoding="utf-8")))
suite = sys.argv[1] if len(sys.argv) > 1 else "suite_qv16b"
for r in rows:
    if r["suite"] == suite:
        print(f"split={r['split']} seed={r['seed']} {r['tag']:20s} n={r['n']:>2} sv={r['sv']} amp={r['amp48']} cpp={r['cpp']} hnr={r['hnr']} asr={r['asr']}")
