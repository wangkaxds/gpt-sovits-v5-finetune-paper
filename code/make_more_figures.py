import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = json.load(open(os.path.join(BASE, "data", "train_runs_compact.json"), encoding="utf-8"))

def pick(name):
    for r in runs:
        if r["name"] == name:
            return r
    return None

names = [
    ("kan_v5dev_r4", "r4 q,v"),
    ("kan_v5dev_r8", "r8 q,v,out"),
    ("kan_v5dev_r16", "r16 q,v,out"),
    ("kan_v5dev_r32", "r32 q,v,out"),
    ("kan_v5dev_vo16", "vo16 v,out"),
    ("kan_v5dev_qv16b", "qv16b q,v"),
    ("kan_v5dev_refenc", "refenc q + ref_enc"),
    ("kan_v5dev_pair16", "pair16 q,v,out"),
]
fig, ax = plt.subplots(figsize=(8, 4.2))
for key, label in names:
    r = pick(key)
    if not r or not r["losses"]:
        continue
    arr = np.array(r["losses"], dtype=float)
    steps, loss = arr[:, 1], arr[:, 0]
    # running median of 5 log points to suppress the heavy batch noise
    if len(loss) >= 5:
        smooth = np.convolve(loss, np.ones(5) / 5, mode="valid")
        ax.plot(steps[4:], smooth, lw=1.2, label=label)
    else:
        ax.plot(steps, loss, lw=1.2, label=label)
ax.set_xlabel("training step")
ax.set_ylabel("CFM loss (5-point moving average)")
ax.set_title("Training loss by route (single epoch, 1,200 steps)")
ax.set_ylim(0, 0.45)
ax.legend(fontsize=7, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "figures", "fig6_training_loss.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# trainable parameter bar
labels = ["out4", "refenc", "r4", "vo16", "ada8", "r8", "r16/clean/pair/fixed/aug", "all8", "r32", "wns4"]
vals = [180224, 747264, 360448, 1441792, 1261568, 1081344, 2162688, 2523136, 4325376, 29851648]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(labels, np.array(vals) / 1e6, color="#2e75b6")
ax.set_xlabel("trainable parameters (M)")
ax.set_title("Trainable parameter count by route (logged checkpoints)")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "figures", "fig7_trainable_params.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig6/7 done")
