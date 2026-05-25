# 0520 Experiment Bundle

This repository is the cleaned Track A / Track B experiment bundle for the
red-team robustness runs and the completed second-judge pass.

## Current status

- Track A and Track B source code are kept in `track_a/` and `track_b/`.
- Raw model outputs from the second model run are kept in `results/from_second_model/`.
- The final second-judge outputs are kept in `results/second_judge/`.
- Old smoke tests, partial second-judge files, and obsolete summary exports have been moved out of the working tree.

## Main outputs to use

- Full judged rows:
  `results/second_judge/ab_second_judge_full_final_clean.csv`
- Final judge summary:
  `results/second_judge/ab_second_judge_summary_final_clean.csv`
- ASR tables:
  `results/second_judge/asr_tables_final/`

The full clean CSV currently contains 22,880 judged rows with no `ERROR`
judge labels. Use the ASR tables for model, track, scenario, turn, and attack
method success-rate analysis.

## Project layout

- `track_a/` - Track A experiment code and source prompts.
- `track_b/` - Track B PromptFuzz-based experiment code and datasets.
- `scripts/` - Local runners, second-judge tooling, and summary scripts.
- `analysis/` - Cross-track analysis helpers.
- `results/from_second_model/` - Raw Track A and Track B model outputs.
- `results/second_judge/` - Final judged outputs and ASR tables.
- `results/run_logs/` - Historical CSV run log.
- `docs/` - Project notes, structure, run notes, and result documentation.
- `external/` - Optional copied external tooling.

See `docs/PROJECT_STRUCTURE.md` for a more detailed file map.
Cloud GPU rerun notes are in `docs/CLOUD_GPU.md`.

## Common commands

Run the second judge again from existing Track A / Track B results:

```powershell
python scripts/rejudge_ab_results.py --retry_errors --skip_errors --stop_on_daily_limit
```

Regenerate final ASR tables from the clean full CSV:

```powershell
python scripts/summarize_second_judge_asr.py `
  --input results/second_judge/ab_second_judge_full_final_clean.csv `
  --output_dir results/second_judge/asr_tables_final
```

## Notes

- Keep API keys in environment variables or local `.env` files only.
- `.venv312/`, logs, Python caches, and local secrets are ignored by git.
- The previous platform-evaluation export CSVs were treated as disposable
  derived files during cleanup. They can be regenerated from the final clean
  full CSV if needed.
