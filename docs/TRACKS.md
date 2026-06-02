# Track Mapping

This bundle is assembled from two experiment tracks plus optional external
tooling.

## Track A

- Directory: `track_a/`
- Original role: baseline thesis experiment flow.
- Preserved outputs: `results/from_second_model/track_a/`
- Final judged Track A rows are included in:
  `results/second_judge/ab_second_judge_full_final_clean_valid.csv`

## Track B

- Directory: `track_b/`
- Original role: PromptFuzz-based red-team robustness flow.
- Preserved outputs: `results/from_second_model/track_b/`
- Final judged Track B rows are included in:
  `results/second_judge/ab_second_judge_full_final_clean_valid.csv`

Track B has two phases:

- `init` - initial seed evaluations.
- `focus` - target-model responses to MCTS-guided mutated prompts.

The `focus` phase is part of the fuzzing evaluation corpus, but it should be
described as evaluated mutation-stage responses rather than final-only selected
attacks.

## External

- Directory: `external/CKA-Agent/`
- Role: optional copied external tooling.
- It is not required for reading the final second-judge CSVs or ASR tables.
