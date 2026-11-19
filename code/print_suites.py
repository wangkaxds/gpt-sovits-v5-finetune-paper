import csv
rows = list(csv.DictReader(open(r"D:\杂活\甜蜜女友3\v5_paper\data\all_suite_summary.csv", encoding="utf-8")))
for r in rows:
    if r["split"] == "val" and r["seed"] == "1":
        print(f"{r['suite']:28s} {r['tag']:22s} n={r['n']:>2} sv={r['sv']} amp={r['amp48']} cpp={r['cpp']} hnr={r['hnr']} asr={r['asr']}")
