# Run Notes

These notes are for rerunning or extending the experiment. They replace older
cloud-only reminders.

## Environment

Use Python 3.12 if possible. The local virtual environment used during cleanup
was `.venv312/`, which is intentionally ignored by git.

Set API keys as environment variables before judge or OpenAI-compatible API
runs:

```powershell
$env:OPENAI_API_KEY = "<your key>"
```

Do not commit real keys.

## Track A and Track B

Track A and Track B can still be run through their own project entry points,
but the current preserved experiment outputs are already under:

- `results/from_second_model/track_a/`
- `results/from_second_model/track_b/`

Use those preserved outputs as the input source for second-judge analysis unless
you intentionally rerun model inference.

For cloud GPU reruns, see `docs/CLOUD_GPU.md`.

## Second Judge

The reportable valid second-judge result is:

```text
results/second_judge/ab_second_judge_full_final_clean_valid.csv
```

To rerun the judge with slower, safer behavior while keeping the final clean
corpus filter:

```powershell
.\.venv312\Scripts\python.exe scripts\rejudge_ab_results.py `
  --exclude_row_ids_csv results\second_judge\quality_manifest\excluded_row_ids.csv `
  --retry_errors `
  --skip_errors `
  --stop_on_daily_limit `
  --api_batch_size 1
```

To regenerate ASR tables:

```powershell
.\.venv312\Scripts\python.exe scripts\summarize_second_judge_asr.py `
  --input_csv results\second_judge\ab_second_judge_full_final_clean_valid.csv `
  --output_dir results\second_judge\asr_tables_final `
  --summary_csv results\second_judge\ab_second_judge_summary_final_clean.csv
```

## Historical Log

The root-level `results_log.csv` was moved to:

```text
results/run_logs/results_log.csv
```
