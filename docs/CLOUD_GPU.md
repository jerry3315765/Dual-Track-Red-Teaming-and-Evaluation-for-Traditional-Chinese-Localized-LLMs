# Cloud GPU Notes

These notes are retained for reruns that use an OpenAI-compatible server on a
cloud GPU machine.

## Basic flow

1. Start the model server with `scripts/cloud_gpu_launch.sh`.
2. Update `models.yaml` so each model points to the server `base_url`.
3. Run one of the repo-level runners:
   - `scripts/run_single_model.sh`
   - `scripts/run_all_models.sh`

## Configuration

- Put model files under the `MODEL_PATH` expected by
  `scripts/cloud_gpu_launch.sh`.
- Adjust tensor parallelism and maximum model length for the available GPU.
- Keep real keys in environment variables, not in tracked config files.

## Current result state

The cleaned repository already contains preserved model outputs under
`results/from_second_model/` and final judge outputs under
`results/second_judge/`. Use this cloud workflow only when intentionally
rerunning model inference.
