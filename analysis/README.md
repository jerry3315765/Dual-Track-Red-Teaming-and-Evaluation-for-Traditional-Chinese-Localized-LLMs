# DIVI-SHAP analysis

This folder contains a wrapper script to cluster Track A/B results and produce SHAP-based token restoration.

## Run

```
python analysis/run_divi_shap.py \
  --track_a_results track_a/data/results \
  --track_b_results track_b/Results \
  --output_dir analysis/divi_shap
```

## Outputs

- analysis/divi_shap/clustered_traces.json
- analysis/divi_shap/cluster_assignments.csv
- analysis/divi_shap/shap_restore.json (if SHAP is installed)

## Notes

- Requires `sentence-transformers` and `torch`.
- SHAP analysis also requires `shap`.
