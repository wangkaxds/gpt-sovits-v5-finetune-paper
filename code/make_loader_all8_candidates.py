"""One-off helper: select all8/base candidates. Requires V5_EVAL_ROOT."""
import json
import os

EVAL = os.environ.get("V5_EVAL_ROOT")
if not EVAL:
    raise SystemExit("Set V5_EVAL_ROOT")
src = os.path.join(EVAL, "candidates_loader_fix_test.json")
data = json.load(open(src, encoding="utf-8"))
out = {key: data[key] for key in ["v5dev", "all8_s1200"]}
path = os.path.join(EVAL, "candidates_loader_fix_all8.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(path)
