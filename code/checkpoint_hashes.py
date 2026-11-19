import os, glob, sys, hashlib
import torch
sys.path.insert(0, r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006\GPT_SoVITS")
from process_ckpt import load_sovits_new

ROOT = r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006"

def digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        if torch.is_tensor(v):
            h.update(k.encode())
            h.update(v.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]

for route in ["ada8", "all8", "r16", "refenc"]:
    fs = sorted(glob.glob(os.path.join(ROOT, f"SoVITS_weights_v5dev_{route}", "*.pth")))
    print(f"\n{route} files={len(fs)}")
    for f in fs[::max(1, len(fs)//5)]:
        ck = load_sovits_new(f)
        sd = {k: v for k, v in ck["weight"].items() if "lora_" in k or k.startswith(("ref_enc", "wns1"))}
        print(os.path.basename(f), digest(sd), "keys", len(sd))
