# models.yaml Guidance

`models.yaml` stays at the repository root because runners expect it there.

Common fields:

- `name`: reporting label used in output tables.
- `base_url`: OpenAI-compatible endpoint when using a remote server.
- `model_path`: model id or local path.
- `api_key_env`: environment variable name for the API key.
- `max_tokens`, `temperature`: optional generation overrides.
- `quantization`, `device`: optional local-loading hints for Track A.

Keep real credentials out of this file.
