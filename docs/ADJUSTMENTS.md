# Adjustable Settings

This file records the main knobs that still matter after the final second-judge
cleanup.

## Models

- Edit `models.yaml` for target model names, endpoints, and per-model options.
- Keep API keys outside tracked files. Use environment variables such as
  `OPENAI_API_KEY`.

## Track A

- Source code: `track_a/`
- Main entry points:
  - `track_a/src/main.py`
  - `track_a/src/integrate_redteam_divi.py`
- Scenario source currently lives under Track A config/data files rather than a
  root-level `scenarios.json`.

## Track B

- Source code: `track_b/`
- Main PromptFuzz entry point:
  - `track_b/Experiment/run.py`
- Red-team datasets:
  - `track_b/Datasets/redteam_robustness_dataset.jsonl`
  - `track_b/Datasets/redteam_focus_seed*.jsonl`
  - `track_b/Datasets/redteam_focus_defense*.jsonl`

## Second Judge

- Main script: `scripts/rejudge_ab_results.py`
- Response-quality manifest script: `scripts/build_quality_corpus_manifest.py`
- Summary script: `scripts/summarize_second_judge_asr.py`
- Final outputs:
  - `results/second_judge/ab_second_judge_full_final_clean_valid.csv`
  - `results/second_judge/ab_second_judge_summary_final_clean.csv`
  - `results/second_judge/asr_tables_final/`
  - `results/second_judge/quality_manifest/`

Useful second-judge options:

- `--retry_errors` retries rows with previous error labels.
- `--skip_errors` continues past judge failures instead of blocking the run.
- `--api_batch_size` controls request batching.
- `--stop_on_daily_limit` stops cleanly when an API daily limit is detected.
- `--exclude_row_ids_csv` applies the clean-corpus exclusion manifest before
  judging, so invalid source rows are not reintroduced.

## Analysis

- ASR summaries should be regenerated from
  `results/second_judge/ab_second_judge_full_final_clean_valid.csv`.
- DIVI-SHAP helpers remain under `analysis/` and `track_a/analysis/`.
