import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clean_text(value: str) -> str:
    text = (value or "").replace("\x00", " ").strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def load_input_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            response = clean_text(row.get("response", ""))
            if not response:
                continue
            rows.append({
                "row_id": row.get("row_id", ""),
                "track": row.get("track", ""),
                "source_file": row.get("source_file", ""),
                "model_label": row.get("model_label", ""),
                "model_name": row.get("model_name", ""),
                "phase": row.get("phase", ""),
                "system_prompt_name": row.get("system_prompt_name", ""),
                "scenario_id": row.get("scenario_id", ""),
                "template_id": row.get("template_id", ""),
                "turn_index": row.get("turn_index", ""),
                "turn_total": row.get("turn_total", ""),
                "attack_method": row.get("attack_method", ""),
                "mutation": row.get("mutation", ""),
                "query": row.get("query", ""),
                "prompt": row.get("prompt", ""),
                "response": response,
                "second_score": row.get("second_score", ""),
                "second_success": parse_bool(row.get("second_success", "")),
                "second_classification": row.get("second_classification", ""),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a previously embedded response corpus to match a reviewed DIVI-SHAP input CSV."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_metadata_path = source_dir / "response_metadata.json"
    source_embeddings_path = source_dir / "response_embeddings.npy"
    if not source_metadata_path.exists():
        raise FileNotFoundError(source_metadata_path)
    if not source_embeddings_path.exists():
        raise FileNotFoundError(source_embeddings_path)

    input_rows = load_input_rows(input_csv)
    with source_metadata_path.open("r", encoding="utf-8") as f:
        source_rows = json.load(f)
    source_embeddings = np.load(source_embeddings_path, mmap_mode="r")

    if len(source_rows) != source_embeddings.shape[0]:
        raise ValueError(
            f"metadata/embedding row mismatch: {len(source_rows)} vs {source_embeddings.shape[0]}"
        )

    row_id_to_index = {}
    duplicates = []
    for idx, row in enumerate(source_rows):
        row_id = row.get("row_id", "")
        if row_id in row_id_to_index:
            duplicates.append(row_id)
        row_id_to_index[row_id] = idx
    if duplicates:
        raise ValueError(f"duplicate row_id in source metadata, first duplicate: {duplicates[0]}")

    missing = [row["row_id"] for row in input_rows if row["row_id"] not in row_id_to_index]
    if missing:
        raise ValueError(f"{len(missing)} input row_id values were not found, first missing: {missing[0]}")

    indices = np.asarray([row_id_to_index[row["row_id"]] for row in input_rows], dtype=np.int64)
    filtered_embeddings = np.asarray(source_embeddings[indices], dtype=np.float32)

    np.save(output_dir / "response_embeddings.npy", filtered_embeddings)
    with (output_dir / "response_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(input_rows, f, ensure_ascii=False, indent=2)

    summary = {
        "input_csv": str(input_csv),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_rows": len(source_rows),
        "input_rows": len(input_rows),
        "embedding_shape": list(filtered_embeddings.shape),
    }
    with (output_dir / "filtered_embedding_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
