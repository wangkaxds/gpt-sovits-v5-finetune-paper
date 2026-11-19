# Inference-side LoRA target fix

The upstream `GPT_SoVITS/TTS_infer_pack/TTS.py` hard-codes:

```python
target_modules=["to_k", "to_q", "to_v", "to_out.0"]
```

This silently ignores checkpoints trained on other modules. In our runs:

- `ada8` trained `attn_norm.linear`; all five measured checkpoints produced identical audio metrics because none of the adapter keys were loaded.
- `all8` trained `ff.0`/`ff.2` in addition to q/k/v/out; the FF adapters were ignored.
- `last5`/`last5b` are full-parameter checkpoints with a v5dev header; their loading path was not verified, and the weights were later deleted.

## Patch

```python
lora_keys = [k for k in dict_s2["weight"] if ".lora_A." in k]
if if_lora_v3 == False or not lora_keys:
    vits_model.load_state_dict(dict_s2["weight"], strict=False)
else:
    targets = set()
    for key in lora_keys:
        module = key.split(".lora_A.")[0]
        module = module.split("estimator.", 1)[-1]
        module = re.sub(r"transformer_blocks\.\d+\.", "", module)
        targets.add(module)
    lora_config = LoraConfig(
        target_modules=sorted(targets),
        r=dict_s2["lora_rank"],
        lora_alpha=dict_s2["lora_rank"],
        init_lora_weights=True,
    )
    vits_model.cfm = get_peft_model(vits_model.cfm, lora_config)
    vits_model.load_state_dict(dict_s2["weight"], strict=False)
    vits_model.cfm = vits_model.cfm.merge_and_unload()
```

Backup: `TTS.py.bak_lora_targets_20261120`.

## Verification

1. Load each checkpoint with `process_ckpt.load_sovits_new` (do not call raw `torch.load`; V5 replaces the ZIP magic with a version byte).
2. Extract target suffixes and assert they equal the training config.
3. Before/after the patch, `ada8` should change from identical metrics to non-identical outputs.

## Status

The patch is saved in the local experimental tree, but the audio-level retest was **not completed** because the local API process failed to bind port 9880 during reload. The paper marks `ada8` and `all8` as invalid/partial, not as model conclusions.
