"""Minimal, no-GPU tests for the LoRA loader guards.

The test extracts the actual helper functions from the local TTS.py and executes
their source with only `re` and `torch` in the namespace. This avoids importing
the full TTS stack (which hangs on model/vocoder initialization).

    set GPT_SOVITS_ROOT=<path>
    python code/test_lora_loader_guards.py

It does not load a model or generate audio.
"""
import ast
import os
import re
import sys

import torch

ROOT = os.environ.get("GPT_SOVITS_ROOT")
if not ROOT:
    raise SystemExit("Set GPT_SOVITS_ROOT to the GPT-SoVITS V5 package directory")
TTS_PATH = os.path.join(ROOT, "GPT_SoVITS", "TTS_infer_pack", "TTS.py")
source = open(TTS_PATH, encoding="utf-8").read()
tree = ast.parse(source)
wanted = {"_lora_clean_module_name", "_infer_lora_targets", "_check_lora_load_result",
          "_expected_lora_deltas"}
segments = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        segments[node.name] = ast.get_source_segment(source, node)
missing = wanted - set(segments)
if missing:
    raise SystemExit(f"helper functions not found in TTS.py: {missing}")
ns = {"re": re, "torch": torch}
exec("\n\n".join(segments[name] for name in wanted), ns)
_infer_lora_targets = ns["_infer_lora_targets"]
_check_lora_load_result = ns["_check_lora_load_result"]
_expected_lora_deltas = ns["_expected_lora_deltas"]

FAIL = []


def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
    except Exception as exc:
        FAIL.append(f"{name}: {exc}")
        print(f"FAIL {name}: {exc}")


def raises(fn):
    try:
        fn()
    except Exception:
        return
    raise AssertionError("expected an exception")


def test_targets():
    keys = [
        "cfm.base_model.model.estimator.transformer_blocks.0.attn.to_q.lora_A.default.weight",
        "cfm.base_model.model.estimator.transformer_blocks.0.attn.to_v.lora_A.default.weight",
    ]
    assert _infer_lora_targets(keys) == {"attn.to_q", "attn.to_v"}
    assert _infer_lora_targets(["cfm.base_model.model.estimator.attn_norm.linear.lora_A.default.weight"]) == {"attn_norm.linear"}
    ff = [
        "cfm.base_model.model.estimator.transformer_blocks.0.ff.ff.0.0.lora_A.default.weight",
        "cfm.base_model.model.estimator.transformer_blocks.0.ff.ff.2.lora_A.default.weight",
    ]
    assert _infer_lora_targets(ff) == {"ff.ff.0.0", "ff.ff.2"}


def test_formula():
    a = torch.ones(2, 3)
    b = torch.ones(4, 2)
    sd = {
        "m.lora_A.default.weight": a,
        "m.lora_B.default.weight": b,
    }
    delta, scale = _expected_lora_deltas(sd, rank=2, alpha=2)["m"]
    assert scale == 1.0
    assert tuple(delta.shape) == (4, 3)
    assert torch.allclose(delta, torch.full((4, 3), 2.0))


def test_missing_b():
    raises(lambda: _expected_lora_deltas({"m.lora_A.default.weight": torch.ones(2, 3)}, 2, 2))


def test_bad_shape():
    sd = {
        "m.lora_A.default.weight": torch.ones(2, 3),
        "m.lora_B.default.weight": torch.ones(4, 3),
    }
    raises(lambda: _expected_lora_deltas(sd, 2, 2))


def test_load_result():
    _check_lora_load_result([], [])
    raises(lambda: _check_lora_load_result(["m.lora_A.default.weight"], []))
    raises(lambda: _check_lora_load_result([], ["m.lora_A.default.weight"]))
    raises(lambda: _check_lora_load_result([], ["unexpected.base.weight"]))


check("target inference: q/v, ada, FF", test_targets)
check("formula B@A * alpha/r", test_formula)
check("missing LoRA B", test_missing_b)
check("bad LoRA shape", test_bad_shape)
check("missing/unexpected adapter keys", test_load_result)

if FAIL:
    print(f"\n{len(FAIL)} failed")
    for item in FAIL:
        print(" -", item)
    sys.exit(1)
print("\nall loader guard tests passed")

