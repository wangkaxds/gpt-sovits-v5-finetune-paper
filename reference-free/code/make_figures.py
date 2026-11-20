"""Generate the paper figures from CSV data. No audio or model weights are read."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 220,
})


def read_csv(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def box(ax, xy, w, h, text, fc="#eef2f7", ec="#344054", fs=8.2):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       linewidth=0.9, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, xy1, xy2, style="-|>", color="#344054"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=9,
                                 linewidth=0.9, color=color))


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    y = 3.15
    boxes = [
        (0.15, 1.0, 0.55, "Japanese\ntext"),
        (1.45, 0.95, 0.6, "S1 AR\nGPT"),
        (3.0, 1.05, 0.62, "semantic\ntokens"),
        (4.3, 1.35, 0.7, "S2: enc_p +\nwns1"),
        (5.95, 0.9, 0.72, "CFM DiT\n22 blocks"),
        (7.35, 1.0, 0.65, "HiFi-GAN\n48 kHz"),
        (8.65, 1.1, 0.6, "waveform"),
    ]
    for x, w, h, t in boxes:
        box(ax, (x, y), w, h, t)
    for (x1, w1, _, _), (x2, _, _, _) in zip(boxes[:-1], boxes[1:]):
        arrow(ax, (x1 + w1, y + 0.3), (x2, y + 0.3))
    # ge path: learned vector feeds both S2 consumers; ref_enc is fallback only.
    box(ax, (3.75, 1.55), 2.55, 0.55, "ge = spk_emb\n(E2b: random-init residual)", fc="#e8f0ff", fs=6.8)
    arrow(ax, (5.0, 2.10), (4.65, y - 0.02), color="#2f6fed")
    arrow(ax, (5.15, 2.10), (5.0, y - 0.02), color="#2f6fed")
    box(ax, (0.2, 1.55), 3.15, 0.55, "reference audio → ref_enc\nfallback only when ge is None", fc="#fff4e5", fs=6.6)
    arrow(ax, (3.35, 1.82), (3.75, 1.82), style="<|-|>", color="#b54708")
    box(ax, (6.95, 1.55), 2.6, 0.55, "CFM spk_film\nE2b: zero-init = identity", fc="#f2f4f7", fs=6.8)
    arrow(ax, (6.95, 1.82), (6.3, y - 0.02), style="-|>", color="#667085")
    ax.text(0.15, 4.35, "V5 inference chain and the two audited ge consumers", fontsize=9.0, weight="bold")
    ax.text(0.15, 0.55, "The reference path is a fallback, not the strict no-ref path.  "
                         "C1M is a static wrapper, not a speaker model.", fontsize=6.8, color="#475467")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def fig2_ladder():
    rows = read_csv("experiments.csv")
    keep = [r for r in rows if r["id"] not in {"E0", "E0b", "E2b-test50x3", "Ref-control", "Raw50",
                                                "C1-50", "C1H-50", "C1M-50", "Shelf-B-50", "Open20",
                                                "C1M-val20"}]
    labels = [r["id"] for r in keep]
    sv = [float(r["sv_mean"]) for r in keep]
    cer = [float(r["cer_mean"]) if r["cer_mean"] else np.nan for r in keep]
    x = np.arange(len(keep))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.scatter(x, sv, s=28, c="#2f6fed", zorder=3)
    ax.axhline(0.7909, color="#b54708", lw=0.9, ls="--", label="C1M-val20 (n=20)")
    ax.axhline(0.8063, color="#667085", lw=0.9, ls=":", label="E2b-val20 (n=20)")
    for xi, yi, lab in zip(x, sv, labels):
        if lab in {"E2b", "E10", "E13", "E15b", "E17", "E18"}:
            ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("SV cosine to target centroid")
    ax.set_title("Speaker similarity on val20 (n=20)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    ax.set_ylim(0.50, 0.86)
    ax = axes[1]
    ax.scatter(x, cer, s=28, c="#b42318", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("CER (Whisper, log)")
    ax.set_title("Content degradation on val20", fontsize=9)
    ax.axhline(0.3523, color="#b54708", lw=0.9, ls="--", label="C1M-val20 CER")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ladder.png", bbox_inches="tight")
    plt.close(fig)


def fig3_bands():
    rows = read_csv("bands.csv")
    names = [r["system"] for r in rows]
    colors = ["#475467", "#2f6fed", "#b54708", "#7a5af8"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [float(r["ltas_lt500"]) for r in rows], width=0.38, label="<500 Hz share", color=colors)
    ax.bar(x + 0.2, [float(r["ltas_2_8k"]) for r in rows], width=0.38, label="2–8 kHz share", color=colors, alpha=0.45)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("LTAS share")
    ax.set_title("Band balance", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    ax.bar(x, [float(r["cpp"]) for r in rows], color=colors, width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("CPP")
    ax.set_title("Cepstral prominence (vocoder bottleneck)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_bands.png", bbox_inches="tight")
    plt.close(fig)


def fig4_open():
    rows = read_csv("open_domain.csv")
    pair = read_csv("pairwise.csv")
    n = 20
    m = np.zeros((n, n))
    for r in pair:
        m[int(r["row"]) - 1, int(r["col"]) - 1] = float(r["cos"])
    sv = np.array([float(r["sv"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.plot(np.arange(1, n + 1), sv, "o-", ms=3, lw=1, color="#2f6fed")
    ax.axhline(0.8398, color="#b54708", ls="--", lw=0.9, label="real test50 mean .840")
    ax.set_ylim(0.62, 0.88)
    ax.set_xlabel("invented sentence index")
    ax.set_ylabel("SV to target centroid")
    ax.set_title("Open-domain identity drift", fontsize=9)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax = axes[1]
    im = ax.imshow(m, cmap="magma", vmin=0.6, vmax=1.0)
    ax.set_title("Pairwise speaker similarity", fontsize=9)
    ax.set_xlabel("sentence"); ax.set_ylabel("sentence")
    ax.set_xticks([0, 4, 9, 14, 19]); ax.set_xticklabels([1, 5, 10, 15, 20])
    ax.set_yticks([0, 4, 9, 14, 19]); ax.set_yticklabels([1, 5, 10, 15, 20])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("cosine", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_open.png", bbox_inches="tight")
    plt.close(fig)


def fig5_decision():
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    stages = [
        (0.3, 5.2, "L1/L2 contract\ncode path + cache", "#e8f0ff"),
        (3.8, 5.2, "E10–E18\nconditioning tests", "#fff4e5"),
        (7.3, 5.2, "point estimates\nSV .541–.778", "#fee4e2"),
        (0.3, 3.2, "C1M wrapper\nSV .813, not a model", "#fff4e5"),
        (3.8, 3.2, "open-domain 20\nCER .075, SV .733", "#fee4e2"),
        (7.3, 3.2, "real pairwise .768\nvs generated .823", "#f2f4f7"),
        (0.3, 1.2, "human judgment\n'not the same person'", "#fee4e2"),
        (3.8, 1.2, "identity fail\nL3/L4 not met", "#fee4e2"),
        (7.3, 1.2, "not a success claim", "#f2f4f7"),
    ]
    for x, y, t, c in stages:
        box(ax, (x, y), 2.3, 0.55, t, fc=c, fs=7.4)
    # horizontal links, one per row; no diagonals
    for y in (5.2, 3.2, 1.2):
        arrow(ax, (2.6, y + 0.27), (3.8, y + 0.27))
        arrow(ax, (6.1, y + 0.27), (7.3, y + 0.27))
    # vertical links down the left and right columns
    for x in (1.45, 8.45):
        arrow(ax, (x, 5.2), (x, 3.75))
        arrow(ax, (x, 3.2), (x, 1.75))
    ax.text(0.3, 0.45, "Falsification chain: path evidence, automatic partial, identity fail, human rejection.",
            fontsize=7.2, color="#475467")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_decision.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline(); fig2_ladder(); fig3_bands(); fig4_open(); fig5_decision()
    print("figures written")
