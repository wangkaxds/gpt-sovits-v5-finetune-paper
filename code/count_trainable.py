"""Count saved trainable tensors per route. Requires GPT_SOVITS_ROOT."""
import glob
import os
import re
import sys

import torch

ROOT = os.environ.get("GPT_SOVITS_ROOT")
if not ROOT:
    raise SystemExit("Set GPT_SOVITS_ROOT to the GPT-SoVITS V5 package directory")
sys.path.insert(0, os.path.join(ROOT, "GPT_SoVITS"))
from process_ckpt import load_sovits_new  # noqa: E402

ROUTES = ["r4", "r8", "r16", "r32", "clean16", "pair16", "ada8", "out4",
          "vo16", "wns4", "fixed16", "all8", "last5", "last5b", "refenc",
          "qv16b", "aug16", "full"]
DIRS = {"full": "SoVITS_weights_v5_full"}
for name in ROUTES:
    dirname = DIRS.get(name, f"SoVITS_weights_v5dev_{name}")
    files = sorted(glob.glob(os.path.join(ROOT, dirname, "*.pth")))
    if not files:
        print(f"{name:10s} NO FILES")
        continue
    path = max(files, key=lambda p: int(re.search(r"_s(\d+)", os.path.basename(p)).group(1)))
    ck = load_sovits_new(path)
    sd = ck["weight"]
    lora = {k: v for k, v in sd.items() if "lora_" in k and torch.is_tensor(v)}
    extra = {k: v for k, v in sd.items() if "lora_" not in k and torch.is_tensor(v)}
    skip = ("quantizer.vq", "rotary_embed", "inited", "cluster_size", "embed_avg")
    extra_train = {k: v for k, v in extra.items() if not any(s in k for s in skip)}
    n_lora = sum(v.numel() for v in lora.values())
    n_extra = sum(v.numel() for v in extra_train.values())
    modules = len({k.split(".lora_")[0] for k in lora})
    print(f"{name:10s} file={os.path.basename(path):42s} lora={n_lora:>10,d} "
          f"extra={n_extra:>10,d} total={n_lora + n_extra:>10,d} modules={modules}")
