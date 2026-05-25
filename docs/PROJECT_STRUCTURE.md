# Project Structure

This file documents the cleaned layout after removing obsolete smoke-test and
partial judge artifacts.

## Root

- `README.md` - project entry point and current status.
- `models.yaml` - model configuration used by runners.
- `run_all_experiments.py` - root experiment runner.
- `.env.example` - example environment variable file.
- `.gitignore`, `.gitattributes` - repository hygiene.

The root is intentionally kept small. Supporting notes live in `docs/`.

## Source and Data

- `track_a/` - Track A source, configs, raw prompts, and analysis scripts.
- `track_b/` - Track B PromptFuzz source, scripts, and datasets.
- `Datasets/` - shared or root-level dataset material.
- `external/` - optional copied third-party tooling.
- `scripts/` - repo-level automation and result-processing scripts.
- `analysis/` - repo-level analysis helpers.
- `docs/CLOUD_GPU.md` - optional cloud GPU rerun notes.

## Results

- `results/from_second_model/` - preserved raw model outputs from Track A and
  Track B.
- `results/second_judge/` - final second-judge artifacts.
- `results/second_judge/asr_tables_final/` - success-rate summary tables.
- `results/run_logs/` - historical CSV run log.

## Local and Ignored

- `.venv312/` - local virtual environment.
- `logs/` - local execution logs.
- `__pycache__/` and `scripts/__pycache__/` - Python bytecode cache.

These local/generated directories are not required for reading the final
results.

## Final CSVs

Use these files for reporting:

- `results/second_judge/ab_second_judge_full_final_clean.csv`
- `results/second_judge/ab_second_judge_summary_final_clean.csv`
- `results/second_judge/asr_tables_final/*.csv`

Avoid resurrecting older partial outputs unless you are debugging a historical
run.
