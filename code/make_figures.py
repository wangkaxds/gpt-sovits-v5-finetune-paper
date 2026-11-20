import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fin = list(csv.DictReader(open(os.path.join(BASE, "data", "finetune_retest_summary.csv"), encoding="utf-8")))
turbo = list(csv.DictReader(open(os.path.join(BASE, "data", "turbo8_seed_deltas.csv"), encoding="utf-8")))
cut = list(csv.DictReader(open(os.path.join(BASE, "data", "cut5_seed_summary.csv"), encoding="utf-8")))

# 1. V5 pipeline block diagram
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis("off")
boxes = [
    ("Text", 0.5), ("s1 AR GPT\n(s1v3.ckpt)", 2.0), ("Semantic tokens", 3.7),
    ("enc_p / quantizer\n(frozen in our runs)", 5.5), ("CFM DiT\n(LoRA target)", 7.4),
    ("HiFi-GAN 48k", 9.1),
]
for label, x in boxes:
    ax.add_patch(plt.Rectangle((x - 0.65, 0.55), 1.3, 0.7, fill=False, lw=1.2))
    ax.text(x, 0.9, label, ha="center", va="center", fontsize=9)
for x in [1.15, 2.65, 4.35, 6.15, 8.05]:
    ax.annotate("", xy=(x + 0.25, 0.9), xytext=(x, 0.9), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate("", xy=(7.4, 0.2), xytext=(5.5, 0.2), arrowprops=dict(arrowstyle="->", lw=1, ls="--"))
ax.text(6.45, 0.04, "ref_enc / reference audio (timbre anchor)", ha="center", fontsize=8)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_v5_pipeline.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 2. Fine-tune seed-level deltas (vo16 - v5dev)
seeds = np.array([int(r["seed"]) for r in fin])
d_sv = np.array([float(r["delta_sv"]) for r in fin])
d_asr = np.array([float(r["delta_asr"]) for r in fin])
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, d, title, ylab in [(axes[0], d_sv, "Speaker similarity", "ΔSV"), (axes[1], d_asr, "ASR similarity", "ΔASR")]:
    ax.axhline(0, color="0.5", lw=1)
    ax.plot(seeds, d, "o-", color="#1f4e79")
    ax.set_xticks(seeds)
    ax.set_xlabel("generation seed")
    ax.set_ylabel(ylab)
    ax.set_title(f"{title}\nmean={d.mean():+.4f}")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_finetune_seed_deltas.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 3. turbo8 vs turbo4 seed deltas
seeds = np.array([int(r["seed"]) for r in turbo])
d_sv = np.array([float(r["delta_sv"]) for r in turbo])
fig, ax = plt.subplots(figsize=(6, 3))
ax.axhline(0, color="0.5", lw=1)
ax.bar(seeds, d_sv, color="#2e75b6", width=0.6)
ax.set_xticks(seeds)
ax.set_xlabel("generation seed")
ax.set_ylabel("ΔSV (turbo8 - turbo4)")
ax.set_title(f"Turbo 8 vs 4 steps: mean={d_sv.mean():+.4f}, p=.013")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_turbo8_vs_turbo4.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 4. cut0 vs cut5 across available seeds
seeds = np.array([int(r["seed"]) for r in cut])
cut5 = np.array([float(r["cut5_sv"]) for r in cut])
cut0 = np.array([float(r["cut0_sv"]) if r["cut0_sv"] else np.nan for r in cut])
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(seeds, cut5, "o-", label="cut5", color="#c00000")
ax.plot(seeds[:3], cut0[:3], "s--", label="cut0 (seeds 1-3)", color="#404040")
ax.set_xticks(seeds)
ax.set_xlabel("generation seed")
ax.set_ylabel("SV")
ax.set_ylim(0.87, 0.91)
ax.legend(frameon=False)
ax.set_title("cut5 (5 seeds) vs cut0 (3 seeds); means .9041 vs .8811")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_cut5_vs_cut0.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 5. best-of-k selection bias null
rng = np.random.default_rng(20261120)
# empirical seed SD 0.0179; simulate max-of-5 vs random take
sd = 0.0179
n = 20000
draws = rng.normal(0, sd, size=(n, 7))
gain = draws[:, :5].max(axis=1) - draws[:, 0]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.hist(gain, bins=60, color="#8faadc", alpha=0.85)
ax.axvline(0.0127, color="#c00000", lw=1.8, label="observed +0.0127")
ax.axvline(gain.mean(), color="#404040", lw=1.2, ls="--", label=f"null mean {gain.mean():+.4f}")
ax.set_xlabel("SV gain of max-of-5 over a random take")
ax.set_ylabel("count")
ax.set_title("Selection bias check: P(null >= observed) ≈ .886")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_selection_bias.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("figures done")
