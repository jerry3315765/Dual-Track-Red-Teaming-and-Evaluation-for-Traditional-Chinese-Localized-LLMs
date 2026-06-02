# Second-Judge Outputs

Use this folder for final judged corpus analysis.

## Primary Files

- `ab_second_judge_full_final_clean_valid.csv`
  - Final valid judged corpus for reporting and DIVI-SHAP reruns.
  - Contains 22,603 valid rows.
- `ab_second_judge_summary_final_clean.csv`
  - Model/track/phase summary rebuilt from the valid corpus.
- `asr_tables_final/`
  - ASR tables by track, phase, model, scenario, turn, and mutation/operator.
- `quality_manifest/`
  - Row-level inclusion and exclusion manifest.

## Audit Files

- `ab_second_judge_full_final_clean.csv`
  - Manually cleaned input CSV. It is retained for auditability.
  - Some non-response artifacts such as `nan`/blank response rows are removed
    when building `ab_second_judge_full_final_clean_valid.csv`.

## Response-Quality Manifest

`quality_manifest/excluded_row_ids.csv` records the rows excluded from the raw
Track A/B source outputs before rerunning judging or analysis. Apply this file
with:

```powershell
.\.venv312\Scripts\python.exe scripts\rejudge_ab_results.py `
  --exclude_row_ids_csv results\second_judge\quality_manifest\excluded_row_ids.csv
```

The manifest makes the final 22,603-row corpus reproducible from
`results/from_second_model/` without reintroducing invalid response rows.
