"""One-off helper used to create the loader-fix candidate list.

Requires GPT_SOVITS_ROOT and V5_EVAL_ROOT. It is not needed to reproduce the
paper's statistical tables.
"""
import json
import os

ROOT = os.environ.get("GPT_SOVITS_ROOT")
EVAL = os.environ.get("V5_EVAL_ROOT")
if not ROOT or not EVAL:
    raise SystemExit("Set GPT_SOVITS_ROOT and V5_EVAL_ROOT")
base = os.path.join(ROOT, "GPT_SoVITS", "pretrained_models",
                    "gsv-v5-pretrained", "s2Gv5dev.pth")
out = {"v5dev": {"path": base, "steps": 32, "cfg": 1.3}}
for tag, route, step, rank in [
    ("ada8_s100", "ada8", 100, 8), ("ada8_s300", "ada8", 300, 8),
    ("ada8_s600", "ada8", 600, 8), ("ada8_s1200", "ada8", 1200, 8),
    ("all8_s600", "all8", 600, 8), ("all8_s1200", "all8", 1200, 8),
    ("r16_s300", "r16", 300, 16),
]:
    name = f"kan_v5dev_{route}_e1_s{step}_l{rank}.pth"
    out[tag] = {"path": os.path.join(ROOT, f"SoVITS_weights_v5dev_{route}", name),
                "steps": 32, "cfg": 1.3}
path = os.path.join(EVAL, "candidates_loader_fix_val.json")
json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(path)
