import os, soundfile as sf
import numpy as np

SRC = r"D:\DevCache\temp\dsh-ellEc6\gsv\kan_exp\5-wav32k"
CLEAN = r"D:\DevCache\temp\dsh-ellEc6\gsv\kan_clean_exp\5-wav32k"

def stats(root, names=None):
    if names is None:
        names = [f for f in os.listdir(root) if f.endswith(".wav")]
    durs = []
    for n in names:
        try:
            durs.append(sf.info(os.path.join(root, n)).duration)
        except Exception:
            pass
    durs = np.array(durs)
    return len(durs), durs.sum() / 60, durs.mean(), np.percentile(durs, [5, 50, 95])

n, total, mean, pct = stats(SRC)
print(f"all n={n} total={total:.1f}min mean={mean:.2f}s p5/50/95={pct.round(2)}")
n, total, mean, pct = stats(CLEAN)
print(f"clean n={n} total={total:.1f}min mean={mean:.2f}s p5/50/95={pct.round(2)}")
# test/val sentence lists
for split in ["test", "val"]:
    p = rf"D:\DevCache\temp\dsh-ellEc6\gsv\sentences_eval30.txt" if split == "test" else rf"D:\DevCache\temp\dsh-ellEc6\gsv\sentences10.txt"
    if os.path.exists(p):
        lines = [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
        print(split, len(lines), "chars mean", np.mean([len(x) for x in lines]))
