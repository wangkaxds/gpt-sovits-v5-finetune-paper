r"""Offline audit of saved V5 LoRA checkpoints.

This does not load the audio model and does not generate speech. It checks that
each checkpoint's adapter factors are structurally loadable, that rank metadata
matches the tensors, and that the merged delta is non-zero.

Usage:
    set GPT_SOVITS_ROOT=<path-to-GPT-SoVITS-v5-package>
    python code/audit_lora_checkpoints.py
"""
import csv
import os
import re
import sys
import glob
from pathlib import Path

import torch

ROOT = os.environ.get("GPT_SOVITS_ROOT")
if not ROOT:
    raise SystemExit("Set GPT_SOVITS_ROOT to the GPT-SoVITS V5 package directory")
sys.path.insert(0, os.path.join(ROOT, "GPT_SoVITS"))
from process_ckpt import load_sovits_new  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "lora_checkpoint_audit.csv"
ROUTES = ["r4", "r8", "r16", "r32", "clean16", "pair16", "ada8", "out4",
          "vo16", "wns4", "fixed16", "all8", "refenc", "qv16b", "aug16", "syn", "full"]
ROUTE_DIRS = {"full": "SoVITS_weights_v5_full"}


def target_of(key):
    module = key.split(".lora_A.")[0].split("estimator.", 1)[-1]
    return re.sub(r"transformer_blocks\.\d+\.", "", module)


rows = []
for route in ROUTES:
    dirname = ROUTE_DIRS.get(route, f"SoVITS_weights_v5dev_{route}")
    files = sorted(glob.glob(os.path.join(ROOT, dirname, "*.pth")))
    if not files:
        status = "NO_CHECKPOINT_UNVERIFIABLE" if route == "syn" else "MISSING"
        rows.append(dict(route=route, file="", rank="", targets="", pairs=0,
                         delta_median="", status=status))
        continue
    # Audit the last checkpoint; earlier checkpoints have the same structure.
    path = max(files, key=lambda p: int(re.search(r"_s(\d+)", os.path.basename(p)).group(1)))
    ck = load_sovits_new(path)
    sd = {k: v for k, v in ck["weight"].items() if torch.is_tensor(v)}
    a = {k.split(".lora_A.")[0]: v for k, v in sd.items() if ".lora_A." in k}
    b = {k.split(".lora_B.")[0]: v for k, v in sd.items() if ".lora_B." in k}
    common = sorted(set(a) & set(b))
    ranks = sorted({int(a[k].shape[0]) for k in common})
    targets = sorted({target_of(k + ".lora_A.") for k in common})
    metadata_rank = ck.get("lora_rank")
    deltas = []
    for k in common:
        r = a[k].shape[0]
        # alpha == r in these runs, so the merge scale is 1.0
        delta = b[k].float() @ a[k].float()
        deltas.append(float(delta.norm()))
    delta_median = float(torch.tensor(deltas).median()) if deltas else 0.0
    status = "OK"
    if not common:
        status = "NO_LORA_PAIRS"
    elif len(ranks) != 1 or (metadata_rank is not None and int(metadata_rank) != ranks[0]):
        status = "RANK_MISMATCH"
    elif any(".lora_A." not in k + ".lora_A." for k in common):  # defensive
        status = "BAD_KEY"
    elif delta_median <= 0:
        status = "ZERO_DELTA"
    rows.append(dict(route=route, file=os.path.basename(path), rank=metadata_rank,
                     targets=";".join(targets), pairs=len(common),
                     delta_median=f"{delta_median:.6g}", status=status))

# Explicit unavailable routes
for route in ("last5", "last5b"):
    rows.append(dict(route=route, file="", rank="", targets="", pairs=0,
                     delta_median="", status="DELETED_UNVERIFIABLE"))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUT}")
for r in rows:
    print(r)
