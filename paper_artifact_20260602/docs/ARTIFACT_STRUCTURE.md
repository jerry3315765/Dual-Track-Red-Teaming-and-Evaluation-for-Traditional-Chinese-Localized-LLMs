# Artifact Structure

This artifact is organized around the data flow used in the paper:

1. Final judged responses.
2. Response-quality manifests and exclusion lists.
3. Response-only DIVI-SHAP input construction.
4. DIVI clustering and SHAP interpretation.
5. Compact reporting tables and manuscript source.

## Directories

- `data/final_judged/`: final second-judge response tables used as the starting point for reported analyses.
- `data/quality_manifest/`: row-level inclusion/exclusion records for response-quality filtering.
- `data/divi_shap_input/`: response-only input and supporting ASR summaries for the DIVI-SHAP run.
- `results/divi_run_no_hard_20260602_seed42/`: full DIVI-SHAP run outputs.
- `results/latest_divi_shap_20260602/`: compact tables used for manuscript reporting.
- `scripts/`: corpus filtering and ASR summarization scripts.
- `analysis/`: DIVI-SHAP execution and reporting scripts.
- `track_a/src/DIVI/`: DIVI implementation required by `analysis/run_final_response_only_divi_shap.py`.
- `paper/`: manuscript source, bibliography, and compiled PDF.

## Reporting Input

The response-only DIVI-SHAP input used by the current manuscript is:

`data/divi_shap_input/divi_shap_response_only_input_no_hard_artifact_candidates.csv`

This corpus contains 20,615 responses.

