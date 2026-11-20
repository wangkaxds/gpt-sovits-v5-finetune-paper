"""Print LoRA modules saved in each checkpoint. Requires GPT_SOVITS_ROOT."""
import glob
import os
import re
import sys

ROOT = os.environ.get("GPT_SOVITS_ROOT")
if not ROOT:
    raise SystemExit("Set GPT_SOVITS_ROOT to the GPT-SoVITS V5 package directory")
sys.path.insert(0, os.path.join(ROOT, "GPT_SoVITS"))
from process_ckpt import load_sovits_new  # noqa: E402

ROUTE_DIRS = {
    "full": "SoVITS_weights_v5_full",
}
routes = ["r4", "r8", "r16", "r32", "clean16", "pair16", "ada8", "out4",
          "vo16", "wns4", "fixed16", "all8", "last5b", "refenc", "qv16b", "aug16", "full"]
for route in routes:
    dirname = ROUTE_DIRS.get(route, f"SoVITS_weights_v5dev_{route}")
    files = sorted(glob.glob(os.path.join(ROOT, dirname, "*.pth")))
    if not files:
        print(route, "NO FILES")
        continue
    path = max(files, key=lambda p: int(re.search(r"_s(\d+)", os.path.basename(p)).group(1)))
    ck = load_sovits_new(path)
    sd = ck.get("weight", ck)
    keys = [k for k in sd if "lora_" in k]
    modules = []
    for key in keys:
        module = re.sub(r"\.lora_[AB]\.default\.(weight|bias)$", "", key)
        if module not in modules:
            modules.append(module)
    print(f"\n{route} | file={os.path.basename(path)} | total_keys={len(sd)} | "
          f"lora_keys={len(keys)} | modules={len(modules)}")
    print("  " + "\n  ".join(modules[:16]))
    if len(modules) > 16:
        print("  ...")
