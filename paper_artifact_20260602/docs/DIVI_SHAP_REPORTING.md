# DIVI-SHAP Corpus and Reporting Notes

This note records the current policy for preparing the response-only DIVI-SHAP
corpus and reporting SHAP keywords in the paper.

## Corpus Filtering

Use `results/second_judge/divi_shap_input/` as the staging area.

Recommended input for the next DIVI-SHAP run:

```text
results/second_judge/divi_shap_input/divi_shap_response_only_input_no_hard_artifact_candidates.csv
```

This file excludes only hard decoding artifacts, currently detected by visible
tokenization or mojibake markers such as `Ġ`, `Ċ`, `ðŁ`, `âĢ`, `èµ`, `æº`, and
`ĺĬ`. These rows are treated as corrupted model-output artifacts rather than
valid safety behavior.

Do not automatically exclude all meta-response or thinking-style outputs.
Reasoning traces, `<think>` blocks, or Chinese self-analysis may be genuine
model behavior for some model families and languages. They should be reviewed
as behavior candidates, not removed by broad keyword rules.

Useful review files:

- `divi_shap_hard_artifact_candidates.csv`
- `hard_artifact_excluded_row_ids.csv`
- `divi_shap_review_candidates.csv`
- `review_candidate_row_ids.csv`
- `divi_shap_manual_review_all_rows.csv`

## Current Counts

Current valid corpus:

- All valid rows: 22,603
- Hard artifact candidates: 1,988
- Rows after excluding hard artifacts: 20,615
- Broad review candidates: 8,089

The hard artifact candidates are concentrated in `ds-llama-8b__bf16` across
Track A, Track B init, and Track B focus.

## Reporting SHAP Keywords

Most responses are in Traditional Chinese. SHAP keywords should therefore be
reported in their original extracted form.

Do:

- Preserve Chinese tokens or phrases exactly as output by the SHAP pipeline.
- Group tokens into English semantic descriptions when writing prose.
- Explain the meaning of Chinese keywords in English narrative text if needed.

Do not:

- Translate extracted Chinese SHAP keywords into English and present the
  translated words as if they were the model-derived keywords.
- Mix translated keywords with original tokens in the same table without clear
  labeling.

Recommended table style:

```text
Cluster | ASR | Original SHAP keywords / phrases | English interpretation
```

The "Original SHAP keywords / phrases" column should remain Chinese when the
extracted tokens are Chinese. The English interpretation can describe the
semantic category, such as concrete procedural steps, cyber tooling, legality
warnings, or refusal language.
