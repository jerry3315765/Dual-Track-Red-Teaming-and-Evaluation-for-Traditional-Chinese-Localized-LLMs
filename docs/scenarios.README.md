# Scenario Guidance

The older root-level `scenarios.json` note is obsolete for the current cleaned
bundle. Scenario and seed data now live inside the track directories.

Useful locations:

- Track A scenarios/config:
  - `track_a/config/`
  - `track_a/data/raw/`
- Track B red-team datasets:
  - `track_b/Datasets/redteam_robustness_dataset.jsonl`
  - `track_b/Datasets/redteam_focus_seed*.jsonl`
  - `track_b/Datasets/redteam_focus_defense*.jsonl`

When adding new scenarios, keep stable IDs and keep prompts single-line where
possible so downstream CSV exports remain easy to join and summarize.
