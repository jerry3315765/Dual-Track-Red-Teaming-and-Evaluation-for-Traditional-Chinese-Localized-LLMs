# 0520 Experiment Bundle

This folder contains the Track A / Track B experiment code, raw model outputs,
second-judge outputs, and post-hoc analysis helpers for the thesis experiments.

## Reader Entry Points

Use these files first:

- Final valid judged corpus:
  `results/second_judge/ab_second_judge_full_final_clean_valid.csv`
- Final model/track summary:
  `results/second_judge/ab_second_judge_summary_final_clean.csv`
- Final ASR tables:
  `results/second_judge/asr_tables_final/`
- Cleaning manifest:
  `results/second_judge/quality_manifest/`

The current valid judged corpus contains 22,603 rows. It is derived from 22,880
raw Track A/B result rows after excluding 277 rows with invalid or unusable
responses, including API-error, blank, and `nan` response artifacts. The
manifest records the included and excluded row IDs so reruns can apply the same
filter before judging.

## Project Layout

- `track_a/` - Track A multi-turn experiment code and source prompts.
- `track_b/` - Track B PromptFuzz-based experiment code and datasets.
- `scripts/` - second-judge, cleaning, manifest, and ASR tooling.
- `analysis/` - response-only DIVI-SHAP and cross-track analysis helpers.
- `results/from_second_model/` - raw Track A and Track B model outputs.
- `results/second_judge/` - final judged corpus, summaries, and manifests.
- `docs/` - result guide, structure notes, track mapping, and rerun notes.
- `external/` - optional copied third-party tooling.

See `docs/PROJECT_STRUCTURE.md` and `docs/RESULTS.md` for the detailed map.

## Common Commands

Regenerate the clean manifest from raw outputs and the manually cleaned CSV:

```powershell
.\.venv312\Scripts\python.exe scripts\build_quality_corpus_manifest.py `
  --results_root results\from_second_model `
  --clean_csv results\second_judge\ab_second_judge_full_final_clean.csv `
  --manifest_dir results\second_judge\quality_manifest `
  --filtered_csv results\second_judge\ab_second_judge_full_final_clean_valid.csv
```

Run the second judge again while applying the same exclusions:

```powershell
.\.venv312\Scripts\python.exe scripts\rejudge_ab_results.py `
  --exclude_row_ids_csv results\second_judge\quality_manifest\excluded_row_ids.csv `
  --retry_errors --skip_errors --stop_on_daily_limit
```

Regenerate final ASR tables and the model/track summary:

```powershell
.\.venv312\Scripts\python.exe scripts\summarize_second_judge_asr.py `
  --input_csv results\second_judge\ab_second_judge_full_final_clean_valid.csv `
  --output_dir results\second_judge\asr_tables_final `
  --summary_csv results\second_judge\ab_second_judge_summary_final_clean.csv
```

## Notes

- Track B `focus` rows are evaluated target-model responses to mutated prompts,
  not mutator-internal text.
- Track A uses 1-5 second-judge scores; Track B uses binary SAFE/UNSAFE labels.
  Cross-track ASR uses `second_result_binary`.
- Keep API keys in environment variables or local `.env` files only.
