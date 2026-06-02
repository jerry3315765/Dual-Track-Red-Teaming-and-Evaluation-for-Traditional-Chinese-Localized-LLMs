# Paper Experiment Artifact 2026-06-02

This folder is the reader-facing artifact for the paper experiment.
It keeps the files needed to inspect the final reported data and reproduce
the response-only DIVI-SHAP analysis used in the current manuscript.

## What Is Included

- `paper/`: the updated manuscript source, bibliography, and compiled PDF.
- `data/final_judged/`: final second-judge CSVs used as the reporting input.
- `data/quality_manifest/`: row-id manifests documenting which raw rows were retained or excluded.
- `data/divi_shap_input/`: the final response-only corpus used for DIVI-SHAP, plus hard-artifact exclusion summaries.
- `results/latest_divi_shap_20260602/`: compact ASR, cluster, and SHAP token tables for paper reporting.
- `results/divi_run_no_hard_20260602_seed42/`: full DIVI-SHAP run outputs, including embeddings, labels, assignments, metadata, and SHAP JSON.
- `scripts/`: data-quality filtering, second-judge summarization, and DIVI-SHAP input construction scripts.
- `analysis/`: embedding filtering, response-only DIVI-SHAP, and reporting-summary scripts.
- `track_a/src/DIVI/`: the DIVI implementation required by the response-only clustering script.
- `config/`: model configuration and environment-template files.

## Final Corpus Definition

The raw execution produced 22,880 judged responses. The current paper reports
the valid-response corpus after excluding:

- API-error rows,
- empty/null responses,
- visible tokenizer/byte artifacts such as hard decoding artifacts.

Meta/self-referential responses were not removed automatically because they can
be genuine model behavior in this experiment.

The final DIVI-SHAP input is:

`data/divi_shap_input/divi_shap_response_only_input_no_hard_artifact_candidates.csv`

Final corpus size: 20,615 responses.

## Main Reporting Files

- `results/latest_divi_shap_20260602/latest_track_phase_asr.csv`
- `results/latest_divi_shap_20260602/latest_track_model_asr.csv`
- `results/latest_divi_shap_20260602/latest_divi_cluster_summary.csv`
- `results/latest_divi_shap_20260602/latest_shap_content_tokens.csv`
- `results/latest_divi_shap_20260602/latest_shap_tokens_raw.csv`
- `results/latest_divi_shap_20260602/latest_divi_shap_reporting_summary.md`

Chinese SHAP tokens are intentionally preserved in their original extracted
form. They should not be translated into English when reported as SHAP output.

## Reproduction

See `REPRODUCE.md` for the exact commands used to rebuild the ASR tables,
DIVI-SHAP input, DIVI cluster assignments, SHAP summary, and reporting tables.
