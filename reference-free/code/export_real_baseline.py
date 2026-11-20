"""Compute the real-speaker SV baseline (optional, requires local audio + V5).

The published CSVs real_test50.csv / real_pairwise.csv were produced by this
script. It is not runnable without the private corpus and V5 package.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "data"


def load16k(path):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-root", default=os.environ.get("V5_ROOT", ""))
    ap.add_argument("--corpus-root", default=os.environ.get("CORPUS_ROOT", ""))
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    if not args.v5_root or not args.corpus_root:
        raise SystemExit("--v5-root and --corpus-root (or env) are required")
    V5 = Path(args.v5_root).resolve()
    corpus = Path(args.corpus_root).resolve()
    sys.path.insert(0, str(V5 / "GPT_SoVITS"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from sv import SV
    sv = SV("cpu", False)
    train = pd.read_csv(corpus / "train.csv").head(20)
    refs = []
    for p in train["path"]:
        e = sv.compute_embedding3(load16k(p)).detach().float().reshape(-1)
        refs.append(e / e.norm())
    c = torch.stack(refs).mean(0)
    c = c / c.norm()

    test = pd.read_csv(corpus / "test.csv")
    embs, svs, ids = [], [], []
    for _, r in test.iterrows():
        e = sv.compute_embedding3(load16k(r["path"])).detach().float().reshape(-1)
        e = e / e.norm()
        embs.append(e)
        svs.append(float(torch.dot(e, c)))
        ids.append(r["id"])
    E = torch.stack(embs)
    S = E @ E.T
    n = len(E)
    off = S[~torch.eye(n, dtype=torch.bool)]
    pd.DataFrame({"id": ids, "target_sv": svs}).to_csv(out / "real_test50.csv", index=False)
    pd.DataFrame([(ids[i], ids[j], float(S[i, j])) for i in range(n) for j in range(n)],
                 columns=["row", "col", "cos"]).to_csv(out / "real_pairwise.csv", index=False)
    summary = {
        "n": n, "target_mean": float(np.mean(svs)), "target_min": float(np.min(svs)),
        "pairwise_mean": float(off.mean()), "pairwise_min": float(off.min()),
        "pairwise_p10": float(np.percentile(off.numpy(), 10)), "pairwise_max": float(off.max()),
    }
    (out / "real_baseline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
