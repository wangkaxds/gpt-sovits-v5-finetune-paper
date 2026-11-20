"""Verify that checkpoint adapter suffixes can be recovered without hard-coded
q/k/v/out targets. This does not load the audio model or generate speech.

Set GPT_SOVITS_ROOT to the GPT-SoVITS V5 package directory.
"""
import glob
import os
import re
import sys

ROOT = os.environ.get("GPT_SOVITS_ROOT")
if not ROOT:
    raise SystemExit("Set GPT_SOVITS_ROOT to the GPT-SoVITS V5 package directory")
sys.path.insert(0, os.path.join(ROOT, "GPT_SoVITS"))
from process_ckpt import load_sovits_new  # noqa: E402

for route in ["r16", "ada8", "all8", "vo16", "refenc", "full"]:
    dirname = "SoVITS_weights_v5_full" if route == "full" else f"SoVITS_weights_v5dev_{route}"
    files = sorted(glob.glob(os.path.join(ROOT, dirname, "*.pth")))
    if not files:
        print(route, "NO FILES")
        continue
    path = max(files, key=lambda p: int(re.search(r"_s(\d+)", os.path.basename(p)).group(1)))
    ck = load_sovits_new(path)
    targets = set()
    for key in ck["weight"]:
        if ".lora_A." not in key:
            continue
        module = key.split(".lora_A.")[0].split("estimator.", 1)[-1]
        module = re.sub(r"transformer_blocks\.\d+\.", "", module)
        targets.add(module)
    print(route, "rank", ck.get("lora_rank"), "targets", sorted(targets))
