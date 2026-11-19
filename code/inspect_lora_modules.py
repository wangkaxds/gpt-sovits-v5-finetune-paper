import os, torch, glob, re, sys
sys.path.insert(0, r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006\GPT_SoVITS")
from process_ckpt import load_sovits_new

ROOT = r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006"
dirs = [
    "SoVITS_weights_v5dev_r4", "SoVITS_weights_v5dev_r8", "SoVITS_weights_v5dev_r16",
    "SoVITS_weights_v5dev_r32", "SoVITS_weights_v5dev_clean16", "SoVITS_weights_v5dev_pair16",
    "SoVITS_weights_v5dev_ada8", "SoVITS_weights_v5dev_out4", "SoVITS_weights_v5dev_vo16",
    "SoVITS_weights_v5dev_wns4", "SoVITS_weights_v5dev_fixed16", "SoVITS_weights_v5dev_all8",
    "SoVITS_weights_v5dev_last5b", "SoVITS_weights_v5dev_refenc", "SoVITS_weights_v5dev_qv16b",
    "SoVITS_weights_v5dev_aug16",
]
for d in dirs:
    files = sorted(glob.glob(os.path.join(ROOT, d, "*.pth")))
    if not files:
        print(d, "NO FILES")
        continue
    chosen = None
    for f in reversed(files):
        try:
            ck = load_sovits_new(f)
            chosen = (f, ck)
            break
        except Exception:
            continue
    if chosen is None:
        print(d, "ALL UNREADABLE")
        continue
    f, ck = chosen
    sd = ck.get("weight", ck)
    keys = [k for k in sd if "lora_" in k]
    mods = []
    for k in keys:
        m = re.sub(r"\.lora_[AB]\.default\.(weight|bias)$", "", k)
        if m not in mods:
            mods.append(m)
    print(f"\n{d} | file={os.path.basename(f)} | total_keys={len(sd)} | lora_keys={len(keys)} | modules={len(mods)}")
    print("  " + "\n  ".join(mods[:16]))
    if len(mods) > 16:
        print("  ...")
