# Results Guide

The final judged dataset is complete and clean:

- Rows: 22,880
- Judge errors: 0
- Full CSV:
  `results/second_judge/ab_second_judge_full_final_clean.csv`
- Summary CSV:
  `results/second_judge/ab_second_judge_summary_final_clean.csv`

## ASR Tables

The table directory is:

```text
results/second_judge/asr_tables_final/
```

Important files:

- `quality_summary.csv` - row counts and quality checks.
- `asr_by_track.csv` - Track A vs Track B attack success rate.
- `asr_by_model.csv` - overall model-level ASR.
- `asr_by_track_model.csv` - model ASR split by track.
- `asr_by_track_model_turn.csv` - model and track ASR split by dialogue turn.
- `asr_by_track_model_attack_method.csv` - model and track ASR split by attack method.
- `asr_by_track_model_scenario.csv` - model and track ASR split by scenario.
- `asr_by_scenario.csv` - scenario-level ASR.
- `asr_by_attack_method.csv` - attack-method-level ASR.

## Regeneration

Regenerate ASR outputs from the final clean full CSV:

```powershell
python scripts/summarize_second_judge_asr.py `
  --input results/second_judge/ab_second_judge_full_final_clean.csv `
  --output_dir results/second_judge/asr_tables_final
```

Regenerate platform-evaluation datasets only when needed. The old platform
exports were derived files and were removed during cleanup.
