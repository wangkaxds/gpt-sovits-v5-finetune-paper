import re, ast, json, os
from datetime import datetime

LOG = r"D:\DevCache\temp\dsh-ellEc6\gsv\kan_exp\train.log"
runs = []
cur = None
last_ts = None
for line in open(LOG, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    ts = None
    m = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\t", line)
    if m:
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        last_ts = ts
    m = re.match(r"^.+?\tkan_exp\tINFO\t(\{.*\})$", line)
    if m and "train" in m.group(1):
        if cur:
            cur["end"] = last_ts
            runs.append(cur)
        try:
            cfg = ast.literal_eval(m.group(1))
        except Exception:
            cfg = {}
        tr = cfg.get("train", {})
        cur = {
            "start": ts, "name": cfg.get("name"), "dir": tr.get("save_weight_dir"),
            "lr": tr.get("learning_rate"), "batch": tr.get("batch_size"),
            "rank": tr.get("lora_rank"), "epochs": tr.get("epochs"),
            "segment": tr.get("segment_size"), "losses": [], "ckpts": [],
        }
        continue
    if cur is None:
        continue
    m = re.match(r"^.+?\tkan_exp\tINFO\t\[([^\]]+)\]$", line)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        try:
            cur["losses"].append((float(parts[0]), int(float(parts[1])), float(parts[2])))
        except Exception:
            pass
    m = re.search(r"saving ckpt (\S+):(\S+)", line)
    if m:
        cur["ckpts"].append((m.group(1), m.group(2)))
if cur:
    cur["end"] = last_ts
    runs.append(cur)

print(f"runs {len(runs)}")
for i, r in enumerate(runs):
    ls = r["losses"]
    dur = (r["end"] - r["start"]).total_seconds() / 60 if r["end"] and r["start"] else None
    if ls:
        vals = [x[0] for x in ls if x[0] == x[0]]
        print(f"{i:02d} {r['name']:28s} {str(r['dir']):28s} n={len(ls):3d} "
              f"step={ls[-1][1]:4d} dur={dur:5.1f}m loss={vals[0]:.4f}->{vals[-1]:.4f} "
              f"min={min(vals):.4f} max={max(vals):.4f} ckpts={len(r['ckpts'])}")
    else:
        print(f"{i:02d} {r['name']:28s} {str(r['dir']):28s} NO_LOSS dur={dur} ckpts={len(r['ckpts'])}")
# dump compact for figure
out = []
for r in runs:
    if r["losses"]:
        out.append({"name": r["name"], "dir": r["dir"], "start": str(r["start"]), "end": str(r["end"]),
                    "losses": r["losses"]})
json.dump(out, open(r"D:\杂活\甜蜜女友3\v5_paper\data\train_runs_compact.json", "w", encoding="utf-8"), ensure_ascii=False)
