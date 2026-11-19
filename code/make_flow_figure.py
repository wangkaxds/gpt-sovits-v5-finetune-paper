import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.axis("off")
steps = [
    ("1,200 utterances\n+ 30/10 split", 1.0),
    ("zero-shot\nbaseline", 2.8),
    ("rank sweep\nr4/r8/r16/r32", 4.6),
    ("module ablations\nclean / out / vo / ada / all / wns / ref_enc / s1", 6.7),
    ("checkpoint &\nval selection", 8.8),
    ("30-sentence test\n+ seed / bootstrap", 10.7),
    ("audit\nloader / selection bias", 12.6),
]
for label, x in steps:
    ax.add_patch(plt.Rectangle((x - 0.75, 0.4), 1.5, 0.8, fill=False, lw=1.2))
    ax.text(x, 0.8, label, ha="center", va="center", fontsize=8)
for x in [1.8, 3.6, 5.4, 7.5, 9.6, 11.5]:
    ax.annotate("", xy=(x + 0.2, 0.8), xytext=(x, 0.8), arrowprops=dict(arrowstyle="->", lw=1))
ax.text(6.8, 0.15, "test set reused for model selection: reported as effect size, not unbiased generalization", ha="center", fontsize=8, style="italic")
ax.set_xlim(0, 13.6)
ax.set_ylim(0, 1.5)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "figures", "fig8_experiment_flow.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig8 done")
