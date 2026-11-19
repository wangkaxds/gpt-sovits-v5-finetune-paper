"""Verify that the checkpoint's adapter suffixes can be recovered without
hard-coding q/k/v/out. This does not run audio inference.
"""
import os, re, sys, glob
sys.path.insert(0, r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006\GPT_SoVITS")
from process_ckpt import load_sovits_new

ROOT = r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006"
for route in ["r16", "ada8", "all8", "vo16", "refenc"]:
    files = sorted(glob.glob(os.path.join(ROOT, f"SoVITS_weights_v5dev_{route}", "*.pth")))
    ck = load_sovits_new(files[-1])
    targets = set()
    for key in ck["weight"]:
        if ".lora_A." not in key:
            continue
        module = key.split(".lora_A.")[0].split("estimator.", 1)[-1]
        module = re.sub(r"transformer_blocks\.\d+\.", "", module)
        targets.add(module)
    print(route, "rank", ck.get("lora_rank"), "targets", sorted(targets))
