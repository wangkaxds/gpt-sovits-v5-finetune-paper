"""Export open-domain per-sentence scores and the 20x20 speaker similarity matrix."""
import csv
import json
import sys
from pathlib import Path

import torch
import torchaudio

V5 = Path(r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006")
PROJ = Path(r"D:\杂活\甜蜜女友3\gsv-v5-reference-free")
sys.path.insert(0, str(V5 / "GPT_SoVITS"))


def load16k(path):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def main():
    run = PROJ / "logs" / "unseen20_c1m"
    score = json.loads((PROJ / "logs" / "unseen20_c1m_score.json").read_text(encoding="utf-8"))
    texts = [x.strip() for x in (PROJ / "data" / "unseen20.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = score["rows"]
    assert len(texts) == len(rows) == 20

    with (PROJ / "paper" / "data" / "open_domain.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "cer", "sv", "duration_sec"])
        for i, (text, r) in enumerate(zip(texts, rows), 1):
            w.writerow([i, text, f"{r['cer']:.6f}", f"{r['sv_cos']:.6f}", f"{r['duration_sec']:.2f}"])

    from sv import SV
    sv = SV("cpu", False)
    embs = []
    for i in range(1, 21):
        e = sv.compute_embedding3(load16k(run / f"s{i:02d}.wav")).detach().float().reshape(-1)
        embs.append(e / e.norm())
    sim = torch.stack(embs) @ torch.stack(embs).T
    with (PROJ / "paper" / "data" / "pairwise.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "cos"])
        for i in range(20):
            for j in range(20):
                w.writerow([i + 1, j + 1, f"{float(sim[i, j]):.6f}"])
    print("written", len(rows), "sentences; pairwise", float(sim[~torch.eye(20, dtype=torch.bool)].mean()))


if __name__ == "__main__":
    main()
