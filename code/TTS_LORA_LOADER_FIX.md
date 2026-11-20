# Inference-side LoRA target fix

The upstream `GPT_SoVITS/TTS_infer_pack/TTS.py` hard-codes:

```python
target_modules=["to_k", "to_q", "to_v", "to_out.0"]
```

This silently ignores checkpoints trained on other modules. In our runs:

- `ada8` trained `attn_norm.linear`; the five measured checkpoints (s100/300/600/900/1200) produced identical audio metrics because none of the adapter keys were loaded.
- `all8` trained `ff.ff.0.0`/`ff.ff.2` in addition to q/k/v/out; the FF adapters were ignored.
- `last5`/`last5b` are full-parameter block checkpoints with a v5dev header; their loading path was not verified, and the weights were later deleted.
- `full` (`kan_full_e2_s5668_l32.pth`) is **not** full-parameter: it is rank-32 q/k/v/out LoRA trained on the full data set.

## Patch

```python
lora_keys = [k for k in dict_s2["weight"] if ".lora_A." in k]
if if_lora_v3 == False or not lora_keys:
    result = vits_model.load_state_dict(dict_s2["weight"], strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected VITS keys: {result.unexpected_keys[:8]}")
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

    probe_before = sum(p.detach().float().sum() for p in vits_model.cfm.parameters())
    result = vits_model.load_state_dict(dict_s2["weight"], strict=False)

    adapter_missing = [k for k in result.missing_keys if "lora_" in k]
    adapter_unexpected = [k for k in result.unexpected_keys if "lora_" in k]
    other_unexpected = [k for k in result.unexpected_keys if "lora_" not in k]
    if adapter_missing or adapter_unexpected or other_unexpected:
        raise RuntimeError("LoRA checkpoint was not fully consumed: "
                           f"missing={adapter_missing[:8]} "
                           f"unexpected={adapter_unexpected[:8]} "
                           f"other={other_unexpected[:8]}")

    vits_model.cfm = vits_model.cfm.merge_and_unload()
    if any("lora_" in k for k in vits_model.cfm.state_dict()):
        raise RuntimeError("LoRA keys remain after merge_and_unload")

    probe_after = sum(p.detach().float().sum() for p in vits_model.cfm.parameters())
    delta = (probe_after - probe_before).abs().item()
    if delta < 1e-8:
        raise RuntimeError("LoRA loaded but merged checksum did not change")
```

Backup: `TTS.py.bak_lora_targets_20261120`.

## Automatic checks

`code/audit_lora_checkpoints.py` is an offline structural audit. It checks:

- each `.lora_A` has a matching `.lora_B`;
- `lora_rank` metadata equals the tensor rank;
- target suffixes are recoverable;
- the merged delta (B @ A, alpha=r) is non-zero;
- missing checkpoints are reported as `DELETED_UNVERIFIABLE`.

`code/test_lora_target_inference.py` additionally prints the inferred targets for
r16, ada8, all8, vo16, refenc, full. It requires `GPT_SOVITS_ROOT`.

## Verification status

- **Target recovery**: verified for r16, ada8, all8, vo16, refenc, full.
- **Loader-level checks**: `r16`, `ada8`, `all8`, `refenc`, and `full` all loaded
  through `/set_sovits_weights` with `adapter_missing=0`,
  `adapter_unexpected=0`, `other_unexpected=0`.
- **Audio retest**: `ada8_s300` and `all8_s1200` were evaluated on 10 val + 30
  test sentences (seed1). ada8 test SV +0.0002 / ASR −0.008; all8 test SV
  +0.0010 (p=.0084, CI +0.0003…+0.0017) / ASR +0.003. all8 is a small,
  seed1-only, exploratory result; it is not a practical or stable gain.
- **Not completed**: multi-seed test for ada8/all8; any retest for last5/last5b
  (weights deleted); human ABX.

## What this does not prove

This patch is local to the experimental tree; it is not an upstream GPT-SoVITS
release. Passing the loader checks proves that the intended adapters are consumed
and merged. It does **not** prove that the resulting audio is better; that is a
separate statistical and listening question.
