import json, re, os, ast

LOG = r"D:\DevCache\temp\dsh-ellEc6\gsv\kan_exp\train.log"
OUT = r"D:\杂活\甜蜜女友3\v5_paper\data\train_runs.json"
runs = []
cur = None
for line in open(LOG, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    m = re.match(r"^(.+?)\tkan_exp\tINFO\t(\{.*\})$", line)
    if m and "train" in m.group(2):
        if cur:
            runs.append(cur)
        try:
            cfg = ast.literal_eval(m.group(2))
        except Exception:
            cfg = {}
        tr = cfg.get("train", {})
        cur = {
            "start": m.group(1),
            "name": cfg.get("name"),
            "version": cfg.get("version"),
            "save_weight_dir": tr.get("save_weight_dir") or cfg.get("save_weight_dir"),
            "lr": tr.get("learning_rate"),
            "epochs": tr.get("epochs"),
            "batch_size": tr.get("batch_size"),
            "rank": tr.get("lora_rank"),
            "segment_size": tr.get("segment_size"),
            "losses": [],
            "ckpts": [],
        }
        continue
    if cur is None:
        continue
    m = re.match(r"^.+?\tkan_exp\tINFO\t\[([^\]]+)\]$", line)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 3:
            try:
                cur["losses"].append({"loss": float(parts[0]), "step": int(float(parts[1])), "lr": float(parts[2])})
            except Exception:
                pass
        continue
    m = re.search(r"saving ckpt (\S+):(\S+)", line)
    if m:
        cur["ckpts"].append({"name": m.group(1), "status": m.group(2)})
if cur:
    runs.append(cur)

# merge runs with same save dir? Keep sequential, but print compact
print(f"runs={len(runs)}")
for i, r in enumerate(runs):
    ls = r["losses"]
    if not ls:
        print(i, r["name"], r["save_weight_dir"], "NO_LOSS", len(r["ckpts"]))
        continue
    vals = [x["loss"] for x in ls]
    print(i, r["name"], r["save_weight_dir"], f"n={len(ls)} steps={ls[0]['step']}..{ls[-1]['step']}",
          f"loss={vals[0]:.4f}->{vals[-1]:.4f} min={min(vals):.4f} max={max(vals):.4f}",
          f"ckpts={len(r['ckpts'])}")
json.dump(runs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print("saved", OUT)


