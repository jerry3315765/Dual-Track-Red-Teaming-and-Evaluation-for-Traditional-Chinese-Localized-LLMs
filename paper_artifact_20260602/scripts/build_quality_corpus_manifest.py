import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def load_track_a_rows(results_root: Path) -> Iterable[Dict[str, Any]]:
    pattern = str(results_root / "track_a" / "*" / "results_*.json")
    for path_str in sorted(glob.glob(pattern)):
        path = Path(path_str)
        if path.name.startswith("results_final") or "final_results_aggregated" in path.name:
            continue
        model_label = path.parent.name
        with path.open("r", encoding="utf-8") as f:
            items = json.load(f)
        for idx, item in enumerate(items):
            prompt = str(item.get("prompt", ""))
            response = str(item.get("response", ""))
            row_id = "A::" + "::".join(
                [
                    model_label,
                    str(item.get("template_id", "")),
                    str(item.get("system_prompt_name", "")),
                    str(item.get("turn_index", idx)),
                    sha1_text(prompt + "\n" + response),
                ]
            )
            yield {
                "row_id": row_id,
                "track": "A",
                "phase": "",
                "model_label": model_label,
                "source_file": str(path),
            }


def load_track_b_rows(results_root: Path) -> Iterable[Dict[str, Any]]:
    pattern = str(results_root / "track_b" / "*" / "*" / "redteam" / "baseline" / "*" / "all_results_long.csv")
    for path_str in sorted(glob.glob(pattern)):
        path = Path(path_str)
        parts = path.parts
        try:
            track_b_idx = parts.index("track_b")
            run_label = parts[track_b_idx + 1]
            phase = parts[track_b_idx + 2]
        except Exception:
            run_label = path.parents[4].name
            phase = path.parents[3].name

        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = str(row.get("prompt", ""))
                response = str(row.get("response", ""))
                row_id = "B::" + "::".join(
                    [
                        run_label,
                        phase,
                        str(row.get("defense_id", "")),
                        str(row.get("query", "")),
                        str(row.get("index", "")),
                        sha1_text(prompt + "\n" + response),
                    ]
                )
                yield {
                    "row_id": row_id,
                    "track": "B",
                    "phase": phase,
                    "model_label": run_label,
                    "source_file": str(path),
                }


def has_valid_response(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return text.lower() not in {"nan", "none", "null"}


def is_valid_clean_row(row: Dict[str, str]) -> bool:
    return (
        has_valid_response(row.get("response", ""))
        and bool(str(row.get("second_classification", "")).strip())
        and str(row.get("second_classification", "")).strip() != "ERROR"
        and bool(str(row.get("second_result_binary", "")).strip())
    )


def read_clean_rows(path: Path) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    valid = []
    invalid = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_valid_clean_row(row):
                valid.append(row)
            else:
                invalid.append(row)
    return valid, invalid


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build response-quality row-id manifests from raw Track A/B outputs.")
    parser.add_argument("--results_root", default=str(Path("results") / "from_second_model"))
    parser.add_argument("--clean_csv", default=str(Path("results") / "second_judge" / "ab_second_judge_full_final_clean.csv"))
    parser.add_argument("--manifest_dir", default=str(Path("results") / "second_judge" / "quality_manifest"))
    parser.add_argument(
        "--filtered_csv",
        default="",
        help="Optional path for a valid-only copy of the clean CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    clean_csv = Path(args.clean_csv)
    manifest_dir = Path(args.manifest_dir)

    discovered_rows = list(load_track_a_rows(results_root))
    discovered_rows.extend(load_track_b_rows(results_root))
    discovered_by_id = {row["row_id"]: row for row in discovered_rows}

    clean_valid_rows, clean_invalid_rows = read_clean_rows(clean_csv)
    clean_valid_ids = {row["row_id"] for row in clean_valid_rows}
    clean_invalid_ids = {row.get("row_id", "") for row in clean_invalid_rows if row.get("row_id", "")}

    excluded_rows = []
    for row_id, meta in sorted(discovered_by_id.items()):
        if row_id not in clean_valid_ids:
            row = dict(meta)
            row["reason"] = "invalid_clean_row" if row_id in clean_invalid_ids else "absent_from_clean_csv"
            excluded_rows.append(row)

    unexpected_clean_rows = []
    for row in clean_valid_rows:
        row_id = row.get("row_id", "")
        if row_id and row_id not in discovered_by_id:
            unexpected_clean_rows.append(
                {
                    "row_id": row_id,
                    "track": row.get("track", ""),
                    "phase": row.get("phase", ""),
                    "model_label": row.get("model_label", ""),
                    "source_file": row.get("source_file", ""),
                    "reason": "valid_clean_row_not_found_in_raw_sources",
                }
            )

    write_csv(
        manifest_dir / "included_row_ids.csv",
        ["row_id", "track", "phase", "model_label", "source_file"],
        [
            {
                "row_id": row.get("row_id", ""),
                "track": row.get("track", ""),
                "phase": row.get("phase", ""),
                "model_label": row.get("model_label", ""),
                "source_file": row.get("source_file", ""),
            }
            for row in clean_valid_rows
        ],
    )
    write_csv(
        manifest_dir / "excluded_row_ids.csv",
        ["row_id", "track", "phase", "model_label", "source_file", "reason"],
        excluded_rows,
    )
    write_csv(
        manifest_dir / "unexpected_retained_row_ids.csv",
        ["row_id", "track", "phase", "model_label", "source_file", "reason"],
        unexpected_clean_rows,
    )

    quality_rows = [
        {"metric": "raw_rows_discovered", "value": len(discovered_rows)},
        {"metric": "clean_csv_rows", "value": len(clean_valid_rows) + len(clean_invalid_rows)},
        {"metric": "valid_clean_rows", "value": len(clean_valid_rows)},
        {"metric": "invalid_clean_rows", "value": len(clean_invalid_rows)},
        {"metric": "excluded_rows", "value": len(excluded_rows)},
        {"metric": "unexpected_valid_clean_rows", "value": len(unexpected_clean_rows)},
    ]
    write_csv(manifest_dir / "quality_summary.csv", ["metric", "value"], quality_rows)

    if args.filtered_csv:
        if not clean_valid_rows:
            raise RuntimeError("No valid clean rows found; refusing to write an empty filtered CSV.")
        write_csv(Path(args.filtered_csv), list(clean_valid_rows[0].keys()), clean_valid_rows)

    print(f"[DONE] manifest_dir={manifest_dir}")
    for row in quality_rows:
        print(f"[INFO] {row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()
