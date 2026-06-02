# Results Guide

The final reportable judged dataset is:

- Rows: 22,603 valid judged responses
- Source raw rows: 22,880 Track A/B result rows
- Excluded rows: 277 invalid or unusable response rows
- Judge errors in final valid corpus: 0
- Full valid CSV:
  `results/second_judge/ab_second_judge_full_final_clean_valid.csv`
- Summary CSV:
  `results/second_judge/ab_second_judge_summary_final_clean.csv`
- Cleaning manifest:
  `results/second_judge/quality_manifest/`

The older `ab_second_judge_full_final_clean.csv` is kept as the manually cleaned
input used to build the valid corpus. For reporting and post-hoc analysis, use
`ab_second_judge_full_final_clean_valid.csv`.

## Corpus Breakdown

- Track A: 5,510 rows, 2,868 successes, 52.05% ASR
- Track B init: 5,673 rows, 2,839 successes, 50.04% ASR
- Track B focus: 11,420 rows, 5,551 successes, 48.61% ASR
- Track B total: 17,093 rows, 8,390 successes, 49.08% ASR

Track B `focus` rows are target-model responses to MCTS-guided mutated prompts.
They are included as evaluated fuzzing-stage samples. They are not mutator
internal reasoning or mutator-only text.

## ASR Tables

The table directory is:

```text
results/second_judge/asr_tables_final/
```

Important files:

- `quality_summary.csv` - row counts and quality checks.
- `asr_by_track.csv` - Track A vs Track B attack success rate.
- `asr_by_track_phase.csv` - Track A and Track B phase-level ASR.
- `asr_by_model.csv` - overall model-level ASR.
- `asr_by_track_model.csv` - model ASR split by track.
- `asr_by_track_model_turn.csv` - model and track ASR split by dialogue turn.
- `asr_by_track_model_attack_method.csv` - model and track ASR split by attack method.
- `asr_by_track_model_scenario.csv` - model and track ASR split by scenario.
- `asr_by_scenario.csv` - scenario-level ASR.
- `asr_by_attack_method.csv` - attack-method-level ASR.

## Regeneration

Regenerate the clean manifest:

```powershell
.\.venv312\Scripts\python.exe scripts\build_quality_corpus_manifest.py `
  --results_root results\from_second_model `
  --clean_csv results\second_judge\ab_second_judge_full_final_clean.csv `
  --manifest_dir results\second_judge\quality_manifest `
  --filtered_csv results\second_judge\ab_second_judge_full_final_clean_valid.csv
```

Regenerate ASR outputs from the valid clean full CSV:

```powershell
.\.venv312\Scripts\python.exe scripts\summarize_second_judge_asr.py `
  --input_csv results\second_judge\ab_second_judge_full_final_clean_valid.csv `
  --output_dir results\second_judge\asr_tables_final `
  --summary_csv results\second_judge\ab_second_judge_summary_final_clean.csv
```
