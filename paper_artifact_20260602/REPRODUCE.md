# Reproduction Guide

Run commands from this artifact root:

```powershell
cd C:\Users\jerry\Desktop\lab\0311\0520\paper_artifact_20260602
```

Use a Python environment with the packages listed in `requirements.txt`.
The original run used Python 3.12 and the multilingual sentence encoder
`paraphrase-multilingual-mpnet-base-v2`.

## 1. Rebuild ASR Tables From the Final Second-Judge CSV

```powershell
python scripts\summarize_second_judge_asr.py `
  --input_csv data\final_judged\ab_second_judge_full_final_valid.csv `
  --output_dir reproduced\asr_tables_valid `
  --summary_csv reproduced\asr_tables_valid\summary_by_track_model.csv
```

To reproduce the no-hard-artifact ASR used by the current paper:

```powershell
python scripts\summarize_second_judge_asr.py `
  --input_csv data\final_judged\ab_second_judge_full_final_valid.csv `
  --output_dir reproduced\asr_tables_no_hard `
  --summary_csv reproduced\asr_tables_no_hard\summary_by_track_model_no_hard_artifacts.csv `
  --exclude_row_ids_csv data\divi_shap_input\hard_artifact_excluded_row_ids.csv
```

## 2. Rebuild the DIVI-SHAP Input Corpus

```powershell
python scripts\build_divi_shap_input.py `
  --input_csv data\final_judged\ab_second_judge_full_final_valid.csv `
  --output_dir reproduced\divi_shap_input
```

The paper uses:

```text
data/divi_shap_input/divi_shap_response_only_input_no_hard_artifact_candidates.csv
```

This file excludes only hard tokenizer/byte artifacts from the response-only
DIVI-SHAP corpus. Review/meta-response candidates are retained unless they are
also hard artifacts.

## 3. Re-run DIVI Clustering With Reused Embeddings

The artifact includes the final filtered embeddings in:

```text
results/divi_run_no_hard_20260602_seed42/response_embeddings.npy
results/divi_run_no_hard_20260602_seed42/response_metadata.json
```

To rerun DIVI clustering while reusing those embeddings:

```powershell
python analysis\run_final_response_only_divi_shap.py `
  --input_csv data\divi_shap_input\divi_shap_response_only_input_no_hard_artifact_candidates.csv `
  --output_dir results\divi_run_no_hard_20260602_seed42 `
  --seed 42 `
  --split_init original `
  --split_threshold auto `
  --reuse_embeddings `
  --skip_shap
```

## 4. Re-run SHAP on Existing Cluster Labels

Text SHAP is slow. The paper run used two sampled responses per high-risk
cluster to make the run tractable:

```powershell
python analysis\run_final_response_only_divi_shap.py `
  --input_csv data\divi_shap_input\divi_shap_response_only_input_no_hard_artifact_candidates.csv `
  --output_dir results\divi_run_no_hard_20260602_seed42 `
  --seed 42 `
  --split_init original `
  --split_threshold auto `
  --reuse_embeddings `
  --reuse_clusters `
  --shap_samples_per_cluster 2 `
  --shap_min_cluster_size 100 `
  --shap_high_risk_asr 50.0
```

## 5. Rebuild Compact Reporting Tables

```powershell
python analysis\summarize_latest_divi_shap.py `
  --run_dir results\divi_run_no_hard_20260602_seed42 `
  --track_summary_csv data\divi_shap_input\summary_by_track_phase_no_hard_artifacts.csv `
  --model_summary_csv data\divi_shap_input\summary_by_track_model_no_hard_artifacts.csv `
  --output_dir reproduced\latest_divi_shap_20260602
```

## 6. Recreate the Linear Weighted Cohen Sample

The audit sample is drawn from the final no-hard-artifact corpus. It is Track A
only because Track A retains ordinal 1--5 second-judge scores, while Track B is
represented as binary SAFE/UNSAFE labels in the final corpus.

```powershell
python scripts\sample_linear_weighted_cohen_audit.py `
  --input_csv data\divi_shap_input\divi_shap_response_only_input_no_hard_artifact_candidates.csv `
  --output_csv data\human_audit\linear_weighted_cohen_sample_20260602_seed20260602.csv `
  --summary_csv data\human_audit\linear_weighted_cohen_sample_20260602_seed20260602_summary.csv `
  --per_score 60 `
  --seed 20260602
```

After manually filling the `human_score` column, calculate agreement with:

```powershell
python scripts\calculate_linear_weighted_cohen.py `
  data\human_audit\linear_weighted_cohen_sample_20260602_seed20260602.csv
```

## 7. Rebuild the Paper

```powershell
cd paper
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## Expected Final Counts

- Final DIVI-SHAP input rows: 20,615
- Track A: 5,046 total, 2,802 success, ASR 55.53%
- Track B: 15,569 total, 8,012 success, ASR 51.46%
- DIVI clusters: 21
- SHAP summarized high-risk clusters: 9
- Linear weighted Cohen audit sample: 300 rows, 60 rows per Track A score level
