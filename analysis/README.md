# DIVI-SHAP Analysis

Use `run_final_response_only_divi_shap.py` for the thesis response-only
diagnostic pipeline. It embeds model responses only and keeps prompt-side fields
such as track, phase, scenario, turn, and mutation as metadata.

## Run

```powershell
.\.venv312\Scripts\python.exe analysis\run_final_response_only_divi_shap.py `
  --input_csv results\second_judge\ab_second_judge_full_final_clean_valid.csv `
  --output_dir analysis\divi_shap_response_only_current `
  --seed 42 `
  --split_init original
```

## Outputs

- `clustered_traces_response_only.json`
- `cluster_assignments_response_only.csv`
- `cluster_summary_response_only.csv`
- `shap_response_only.json` if SHAP completes

## Notes

- The old `run_divi_shap.py` script embeds prompt and response together and is
  retained only for historical comparison.
- Corpus filtering and SHAP keyword reporting rules are documented in
  `docs/DIVI_SHAP_REPORTING.md`.
- Requires `sentence-transformers` and `torch`.
- SHAP analysis also requires `shap`.
