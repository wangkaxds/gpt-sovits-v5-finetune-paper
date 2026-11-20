"""One-off helper: select loader-fix test candidates. Requires V5_EVAL_ROOT."""
import json
import os

EVAL = os.environ.get("V5_EVAL_ROOT")
if not EVAL:
    raise SystemExit("Set V5_EVAL_ROOT")
src = os.path.join(EVAL, "candidates_loader_fix_val.json")
data = json.load(open(src, encoding="utf-8"))
keep = ["v5dev", "ada8_s300", "all8_s1200"]
out = {key: data[key] for key in keep}
path = os.path.join(EVAL, "candidates_loader_fix_test.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(path)
