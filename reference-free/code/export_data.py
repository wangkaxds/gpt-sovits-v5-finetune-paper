"""Derive open_domain.csv from the recorded open20 score JSON.

This is a data-consistency script. It does not read audio or run a model.
Pairwise.csv is the recorded SV matrix and is not recomputed here.
"""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"


def main():
    score = json.loads((DATA / "open20_score.json").read_text(encoding="utf-8"))
    texts = [x.strip() for x in (DATA / "unseen20.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = score["rows"]
    if len(texts) != 20 or len(rows) != 20:
        raise SystemExit(f"expected 20 rows, got texts={len(texts)} rows={len(rows)}")
    with (DATA / "open_domain.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "cer", "sv", "duration_sec"])
        for i, (text, r) in enumerate(zip(texts, rows), 1):
            w.writerow([i, text, f"{r['cer']:.6f}", f"{r['sv_cos']:.6f}", f"{r['duration_sec']:.2f}"])
    print("written", DATA / "open_domain.csv")


if __name__ == "__main__":
    main()
