import os, glob, sys
import torch
sys.path.insert(0, r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006\GPT_SoVITS")
from process_ckpt import load_sovits_new

ROOT = r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006"
dirs = [
    "r4", "r8", "r16", "r32", "clean16", "pair16", "ada8", "out4", "vo16",
    "wns4", "fixed16", "all8", "last5", "last5b", "refenc", "qv16b", "aug16",
]
for name in dirs:
    fs = sorted(glob.glob(os.path.join(ROOT, f"SoVITS_weights_v5dev_{name}", "*.pth")))
    if not fs:
        print(f"{name:10s} NO FILES")
        continue
    ck = load_sovits_new(fs[-1])
    sd = ck["weight"]
    lora = {k: v for k, v in sd.items() if "lora_" in k and torch.is_tensor(v)}
    extra = {k: v for k, v in sd.items() if "lora_" not in k and torch.is_tensor(v)}
    # exclude non-trainable buffers/quantizer
    skip = ("quantizer.vq", "rotary_embed", "inited", "cluster_size", "embed_avg")
    extra_train = {k: v for k, v in extra.items() if not any(s in k for s in skip)}
    n_lora = sum(v.numel() for v in lora.values())
    n_extra = sum(v.numel() for v in extra_train.values())
    mods = len({k.split(".lora_")[0] for k in lora})
    print(f"{name:10s} file={os.path.basename(fs[-1]):42s} lora_params={n_lora:>10,d} extra_params={n_extra:>10,d} total={n_lora+n_extra:>10,d} modules={mods}")
