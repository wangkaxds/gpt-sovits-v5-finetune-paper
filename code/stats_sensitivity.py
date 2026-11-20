"""Sensitivity checks for the paper's main statistical claims.

Reproduces:
- all8: paired t, Wilcoxon signed-rank, sign test, paired bootstrap CI
- turbo8 vs turbo4: paired t, sign test
- cut5: 3-seed paired t, Wilcoxon, sign test
- vo16: 3-seed paired t

These are different questions, not interchangeable p-values. A seed-level test
uses seeds as units; a sentence-level test uses sentences within one seed.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"


def read(name):
    return list(csv.DictReader(open(DATA / name, encoding="utf-8")))


def paired_bootstrap(diff, n=20000, seed=20261120):
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


loader = read("loader_fix_paired_seed1.csv")
by_tag = {}
for row in loader:
    by_tag.setdefault(row["tag"], {})[row["sentence"]] = float(row["sv"])
all8 = np.array([by_tag["all8_s1200"][k] - by_tag["v5dev"][k]
                 for k in sorted(set(by_tag["all8_s1200"]) & set(by_tag["v5dev"]))])
print("all8 n=%d mean=%+.5f" % (len(all8), all8.mean()))
print("  paired t: t=%.3f p=%.5f" % stats.ttest_1samp(all8, 0))
print("  Wilcoxon: p=%.5f" % stats.wilcoxon(all8).pvalue)
print("  sign: %d/%d positive, p=%.5f" % (
    int((all8 > 0).sum()), len(all8), stats.binomtest(int((all8 > 0).sum()), len(all8), 0.5).pvalue))
print("  bootstrap CI: [%+.5f, %+.5f]" % tuple(paired_bootstrap(all8)))

turbo = np.array([float(r["delta_sv"]) for r in read("turbo8_seed_deltas.csv")])
print("\nturbo n=%d mean=%+.5f" % (len(turbo), turbo.mean()))
print("  paired t: t=%.3f p=%.5f" % stats.ttest_1samp(turbo, 0))
print("  sign: %d/%d positive, p=%.5f" % (
    int((turbo > 0).sum()), len(turbo), stats.binomtest(int((turbo > 0).sum()), len(turbo), 0.5).pvalue))

cut = [r for r in read("cut5_seed_summary.csv") if r["cut0_sv"]]
c0 = np.array([float(r["cut0_sv"]) for r in cut])
c5 = np.array([float(r["cut5_sv"]) for r in cut])
d = c5 - c0
print("\ncut5 n=%d mean=%+.5f" % (len(d), d.mean()))
print("  paired t: t=%.3f p=%.5f" % stats.ttest_rel(c5, c0))
print("  Wilcoxon: p=%.5f" % stats.wilcoxon(d).pvalue)
print("  sign: %d/%d positive, p=%.5f" % (
    int((d > 0).sum()), len(d), stats.binomtest(int((d > 0).sum()), len(d), 0.5).pvalue))

fin = np.array([float(r["delta_sv"]) for r in read("finetune_retest_summary.csv")])
print("\nvo16 n=%d mean=%+.5f" % (len(fin), fin.mean()))
print("  paired t: t=%.3f p=%.5f" % stats.ttest_1samp(fin, 0))
