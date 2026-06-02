# Final Judged Responses

This folder contains the second-judge response tables used as the starting point
for the reported ASR and DIVI-SHAP analyses.

## Files

- `ab_second_judge_full_final_valid.csv`
  - Final valid judged response corpus for reporting and DIVI-SHAP reruns.
  - Contains 22,603 valid rows before the hard-artifact exclusion used by the
    response-only DIVI-SHAP analysis.
- `ab_second_judge_summary_final.csv`
  - Model/track/phase summary rebuilt from the final valid corpus.

For the exact response-only DIVI-SHAP input used by the manuscript, see:

`../divi_shap_input/divi_shap_response_only_input_no_hard_artifact_candidates.csv`
