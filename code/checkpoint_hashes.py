"""Hash adapter/full-condition tensors across checkpoints. Requires GPT_SOVITS_ROOT."""
import glob
import hashlib
import os
import sys

import torch

ROOT = os.environ.get("GPT_SOVITS_ROOT")
if not ROOT:
    raise SystemExit("Set GPT_SOVITS_ROOT to the GPT-SoVITS V5 package directory")
sys.path.insert(0, os.path.join(ROOT, "GPT_SoVITS"))
from process_ckpt import load_sovits_new  # noqa: E402


def digest(sd):
    h = hashlib.sha256()
    for key in sorted(sd):
        value = sd[key]
        if torch.is_tensor(value):
            h.update(key.encode())
            h.update(value.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


for route in ["ada8", "all8", "r16", "refenc", "full"]:
    dirname = "SoVITS_weights_v5_full" if route == "full" else f"SoVITS_weights_v5dev_{route}"
    files = sorted(glob.glob(os.path.join(ROOT, dirname, "*.pth")))
    print(f"\n{route} files={len(files)}")
    for path in files[::max(1, len(files) // 5)]:
        ck = load_sovits_new(path)
        sd = {k: v for k, v in ck["weight"].items()
              if "lora_" in k or k.startswith(("ref_enc", "wns1"))}
        print(os.path.basename(path), digest(sd), "keys", len(sd))
