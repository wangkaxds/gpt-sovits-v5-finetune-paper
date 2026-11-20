# Data dictionary

These CSVs/JSONs are **derived summaries**, not raw audio or model weights.

| File | Rows | Meaning |
|---|---:|---|
| `all_suite_summary.csv` | 158 (41 suites) | one row per (suite, split, seed, tag); columns: n, SV, amp48, CPP, HNR, jitter, ASR, ASR-ok, clipping, duration |
| `cut5_paired_seed1.csv` | 30 | sentence-level cut0/cut5 pairs for seed1; used by `stats_cut5.py` |
| `cut5_seed_summary.csv` | 5 | per-seed cut0/cut5 means; cut0 is empty for seeds 4–5 |
| `finetune_retest_summary.csv` | 3 | vo16_s300 minus v5dev, per seed |
| `turbo8_seed_deltas.csv` | 5 | turbo8 minus turbo4, per seed |
| `loader_fix_summary.csv` | 10 | corrected-loader ada8/all8 + v5dev, val and 30-sentence test, seed1 |
| `loader_fix_paired_seed1.csv` | 90 | sentence-level corrected-loader rows for the same three tags (30 sentences × 3) |
| `lora_checkpoint_audit.csv` | 19 | offline structural audit; `syn` has no checkpoint, `last5/last5b` are `DELETED_UNVERIFIABLE` |
| `lora_merge_verification.csv` | 5 | per-checkpoint summary of actual vs `(alpha/r) B @ A`; max/mean/relative error, finite, untouched-layer delta |
| `lora_merge_module_audit.csv` | 330 | one row per verified target module across r16/ada8/all8/refenc/full |
| `evidence_matrix.csv` | 19 | route × evidence-type matrix: training log, config, checkpoint, runtime load, val, test, seeds, statistics, human listening; `structural` means checkpoint structure audit only, not model-level load verification |
| `result_index.csv` | 15 | key result → data file → script → paper section → caveat |
| `route_ledger.csv` | 19 | unified failure ledger: motivation, config, observation, anomaly/investigation, confidence, advice, result_class, class_note, train_process, unresolved |
| `retracted_claims.csv` | 1 | explicitly retracted claim (`BEST5`); F class is not a fine-tune route |
| `train_runs.json` | 24 runs | parsed training logs, including loss points and checkpoints |
| `train_runs_compact.json` | loss-bearing runs | compact version used by the loss figure |

## What can be reproduced

- All statistical tables can be recomputed from these files with `numpy`/`scipy`.
- The loss figure can be regenerated from `train_runs_compact.json`.
- Checkpoint target/rank/merge-delta audits can be regenerated if the local V5
  package and checkpoints are available (`GPT_SOVITS_ROOT`).
- `lora_merge_verification.csv` and `lora_merge_module_audit.csv` are actual
  model-level checks; they require the local V5 tree and checkpoints, and are not
  reproducible from the public CSVs alone.
- `result_index.csv` maps each headline result to its source file and script; it
  is a navigation aid, not a substitute for the raw audio.

## What cannot be reproduced

- Audio generation, because the 1,200-sentence Kan corpus, 30/10 evaluation
  sentences, reference audio, base models, and checkpoints are not published.
- Training, because the corpus and V5 pretrained weights are not published and
  the full run used a 6 GB laptop GPU over many hours.
- `last5`/`last5b` behavior, because those checkpoints were deleted.
- The exact best-of-5 selection-bias P value: the figure uses a deterministic
  null simulation with SD=0.0179; the original per-seed candidate values were not
  preserved. The retraction is still valid as a selection-bias argument, but the
  P value is not raw-data reproducible.

`all_suite_summary.csv` mixes pre-fix and post-fix loader results. The paper
labels pre-fix ada8/all8 as invalid/partial; `loader_fix_summary.csv` is the
post-fix set and should not be silently merged with the pre-fix rows.
