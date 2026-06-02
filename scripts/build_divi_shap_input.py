import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


BAD_RESPONSE_VALUES = {"", "nan", "none", "null"}
TOKEN_ARTIFACT_MARKERS = ["Ġ", "Ċ", "ðŁ", "âĢ", "èµ", "æº", "ĺĬ"]


def clean_text(value: str) -> str:
    return (value or "").replace("\x00", " ").strip()


def has_valid_response(value: str) -> bool:
    return clean_text(value).lower() not in BAD_RESPONSE_VALUES


def parse_binary(value: str) -> str:
    text = clean_text(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return "1"
    if text in {"0", "false", "no", "n"}:
        return "0"
    return ""


def preview(value: str, limit: int) -> str:
    text = " ".join(clean_text(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def artifact_reasons(value: str) -> List[str]:
    text = clean_text(value)
    lower = text.lower()
    reasons = []

    token_hits = {marker: text.count(marker) for marker in TOKEN_ARTIFACT_MARKERS if marker in text}
    token_hit_total = sum(token_hits.values())
    if token_hit_total >= 3:
        reasons.append("token_artifact_markers")
    elif any(marker in text for marker in ["ðŁ", "âĢ", "ĺĬ"]):
        reasons.append("mojibake_marker")

    if "</think>" in lower or "<think>" in lower:
        reasons.append("leaked_think_tag")

    if "hello!ġhowġcanġiġassistġyouġtoday" in lower or "hello! how can i assist you today" in lower:
        reasons.append("hello_loop")

    return reasons


def artifact_marker_count(value: str) -> int:
    text = clean_text(value)
    return sum(text.count(marker) for marker in TOKEN_ARTIFACT_MARKERS)


def is_hard_artifact(reasons: List[str]) -> bool:
    hard_reasons = {"token_artifact_markers", "mojibake_marker"}
    return any(reason in hard_reasons for reason in reasons)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def normalize_phase(row: Dict[str, str]) -> str:
    phase = clean_text(row.get("phase", ""))
    if phase:
        return phase
    return "none"


def normalize_mutation(row: Dict[str, str]) -> str:
    return clean_text(row.get("mutation", ""))


def normalize_attack_method(row: Dict[str, str]) -> str:
    attack_method = clean_text(row.get("attack_method", ""))
    if attack_method:
        return attack_method
    mutation = normalize_mutation(row)
    if mutation:
        return mutation
    return "unknown"


def is_valid_divi_row(row: Dict[str, str]) -> bool:
    if not has_valid_response(row.get("response", "")):
        return False
    if not clean_text(row.get("second_classification", "")):
        return False
    if clean_text(row.get("second_classification", "")) == "ERROR":
        return False
    if parse_binary(row.get("second_result_binary", "")) == "":
        return False
    return True


def build_rows(rows: List[Dict[str, str]], preview_chars: int) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    divi_rows: List[Dict[str, str]] = []
    inspect_rows: List[Dict[str, str]] = []
    rejected_rows: List[Dict[str, str]] = []

    for idx, row in enumerate(rows):
        valid = is_valid_divi_row(row)
        if not valid:
            reason = []
            if not has_valid_response(row.get("response", "")):
                reason.append("invalid_response")
            if not clean_text(row.get("second_classification", "")):
                reason.append("blank_second_classification")
            if clean_text(row.get("second_classification", "")) == "ERROR":
                reason.append("error_second_classification")
            if parse_binary(row.get("second_result_binary", "")) == "":
                reason.append("blank_second_result_binary")
            rejected_rows.append({
                "source_index": str(idx),
                "row_id": row.get("row_id", ""),
                "track": row.get("track", ""),
                "phase": normalize_phase(row),
                "model_label": row.get("model_label", ""),
                "reason": ";".join(reason),
                "response_preview": preview(row.get("response", ""), preview_chars),
            })
            continue

        response = clean_text(row.get("response", ""))
        reasons = artifact_reasons(response)
        success = parse_binary(row.get("second_result_binary", ""))
        phase = normalize_phase(row)
        mutation = normalize_mutation(row)
        attack_method_norm = normalize_attack_method(row)
        divi_row = {
            "divi_index": str(len(divi_rows)),
            "row_id": row.get("row_id", ""),
            "track": row.get("track", ""),
            "phase": phase,
            "model_label": row.get("model_label", ""),
            "model_name": row.get("model_name", ""),
            "system_prompt_name": row.get("system_prompt_name", ""),
            "scenario_id": row.get("scenario_id", ""),
            "template_id": row.get("template_id", ""),
            "turn_index": row.get("turn_index", ""),
            "turn_total": row.get("turn_total", ""),
            "defense_id": row.get("defense_id", ""),
            "defense_name": row.get("defense_name", ""),
            "attack_method": row.get("attack_method", ""),
            "attack_method_norm": attack_method_norm,
            "mutation": mutation,
            "query": row.get("query", ""),
            "second_result_binary": success,
            "second_success": "TRUE" if success == "1" else "FALSE",
            "second_classification": row.get("second_classification", ""),
            "second_score": row.get("second_score", ""),
            "orig_result_binary": parse_binary(row.get("orig_result_binary", "")),
            "agree_binary": parse_binary(row.get("agree_binary", "")),
            "response_len_chars": str(len(response)),
            "response_len_words_rough": str(len(response.split())),
            "hard_artifact_candidate": "1" if is_hard_artifact(reasons) else "0",
            "review_candidate": "1" if reasons else "0",
            "artifact_reasons": ";".join(reasons),
            "artifact_marker_count": str(artifact_marker_count(response)),
            "embedding_text": response,
            "response": response,
        }
        divi_rows.append(divi_row)
        inspect_rows.append({
            "divi_index": divi_row["divi_index"],
            "row_id": divi_row["row_id"],
            "track": divi_row["track"],
            "phase": phase,
            "model_label": divi_row["model_label"],
            "scenario_id": divi_row["scenario_id"],
            "turn_index": divi_row["turn_index"],
            "mutation": mutation,
            "attack_method_norm": attack_method_norm,
            "query": divi_row["query"],
            "second_result_binary": success,
            "second_classification": divi_row["second_classification"],
            "second_score": divi_row["second_score"],
            "response_len_chars": divi_row["response_len_chars"],
            "hard_artifact_candidate": divi_row["hard_artifact_candidate"],
            "review_candidate": divi_row["review_candidate"],
            "artifact_reasons": divi_row["artifact_reasons"],
            "artifact_marker_count": divi_row["artifact_marker_count"],
            "prompt_preview": preview(row.get("prompt", ""), preview_chars),
            "response_preview": preview(response, preview_chars),
        })

    return divi_rows, inspect_rows, rejected_rows


def sample_by_group(rows: List[Dict[str, str]], keys: List[str], limit_per_group: int) -> List[Dict[str, str]]:
    counts: Dict[tuple, int] = defaultdict(int)
    sampled = []
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        if counts[key] >= limit_per_group:
            continue
        sampled.append(row)
        counts[key] += 1
    return sampled


def count_by(rows: List[Dict[str, str]], keys: List[str]) -> List[Dict[str, str]]:
    counter = Counter(tuple(row.get(key, "") for key in keys) for row in rows)
    output = []
    for key_values, total in sorted(counter.items()):
        item = {key: value for key, value in zip(keys, key_values)}
        item["total"] = str(total)
        output.append(item)
    return output


def success_by(rows: List[Dict[str, str]], keys: List[str]) -> List[Dict[str, str]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    output = []
    for key_values, items in sorted(grouped.items()):
        total = len(items)
        success = sum(1 for row in items if row.get("second_result_binary") == "1")
        item = {key: value for key, value in zip(keys, key_values)}
        item["total"] = str(total)
        item["success"] = str(success)
        item["asr_pct"] = f"{(100.0 * success / total):.2f}" if total else "0.00"
        output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the reviewed response-only CSV for DIVI-SHAP.")
    parser.add_argument(
        "--input_csv",
        default=str(Path("results") / "second_judge" / "ab_second_judge_full_final_clean_valid.csv"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path("results") / "second_judge" / "divi_shap_input"),
    )
    parser.add_argument("--preview_chars", type=int, default=220)
    parser.add_argument("--sample_per_track_phase_model", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_rows(input_csv)
    divi_rows, inspect_rows, rejected_rows = build_rows(source_rows, args.preview_chars)

    divi_fields = [
        "divi_index",
        "row_id",
        "track",
        "phase",
        "model_label",
        "model_name",
        "system_prompt_name",
        "scenario_id",
        "template_id",
        "turn_index",
        "turn_total",
        "defense_id",
        "defense_name",
        "attack_method",
        "attack_method_norm",
        "mutation",
        "query",
        "second_result_binary",
        "second_success",
        "second_classification",
        "second_score",
        "orig_result_binary",
        "agree_binary",
        "response_len_chars",
        "response_len_words_rough",
        "hard_artifact_candidate",
        "review_candidate",
        "artifact_reasons",
        "artifact_marker_count",
        "embedding_text",
        "response",
    ]
    inspect_fields = [
        "divi_index",
        "row_id",
        "track",
        "phase",
        "model_label",
        "scenario_id",
        "turn_index",
        "mutation",
        "attack_method_norm",
        "query",
        "second_result_binary",
        "second_classification",
        "second_score",
        "response_len_chars",
        "hard_artifact_candidate",
        "review_candidate",
        "artifact_reasons",
        "artifact_marker_count",
        "prompt_preview",
        "response_preview",
    ]
    rejected_fields = ["source_index", "row_id", "track", "phase", "model_label", "reason", "response_preview"]

    write_csv(output_dir / "divi_shap_response_only_input.csv", divi_fields, divi_rows)
    write_csv(
        output_dir / "divi_shap_response_only_input_no_hard_artifact_candidates.csv",
        divi_fields,
        [row for row in divi_rows if row.get("hard_artifact_candidate") != "1"],
    )
    write_csv(
        output_dir / "divi_shap_response_only_input_no_review_candidates.csv",
        divi_fields,
        [row for row in divi_rows if row.get("review_candidate") != "1"],
    )
    write_csv(output_dir / "divi_shap_inspection_compact.csv", inspect_fields, inspect_rows)
    write_csv(output_dir / "divi_shap_manual_review_all_rows.csv", inspect_fields, inspect_rows)
    write_csv(
        output_dir / "divi_shap_hard_artifact_candidates.csv",
        inspect_fields,
        [row for row in inspect_rows if row.get("hard_artifact_candidate") == "1"],
    )
    write_csv(
        output_dir / "hard_artifact_excluded_row_ids.csv",
        ["row_id", "track", "phase", "model_label", "artifact_reasons", "artifact_marker_count", "response_preview"],
        [row for row in inspect_rows if row.get("hard_artifact_candidate") == "1"],
    )
    write_csv(
        output_dir / "divi_shap_review_candidates.csv",
        inspect_fields,
        [row for row in inspect_rows if row.get("review_candidate") == "1"],
    )
    write_csv(
        output_dir / "review_candidate_row_ids.csv",
        ["row_id", "track", "phase", "model_label", "artifact_reasons", "artifact_marker_count", "response_preview"],
        [row for row in inspect_rows if row.get("review_candidate") == "1"],
    )
    write_csv(
        output_dir / "divi_shap_inspection_sample.csv",
        inspect_fields,
        sample_by_group(inspect_rows, ["track", "phase", "model_label"], args.sample_per_track_phase_model),
    )
    write_csv(output_dir / "divi_shap_rejected_rows.csv", rejected_fields, rejected_rows)
    write_csv(output_dir / "summary_by_track_phase.csv", ["track", "phase", "total", "success", "asr_pct"], success_by(divi_rows, ["track", "phase"]))
    write_csv(
        output_dir / "summary_by_track_phase_no_hard_artifacts.csv",
        ["track", "phase", "total", "success", "asr_pct"],
        success_by([row for row in divi_rows if row.get("hard_artifact_candidate") != "1"], ["track", "phase"]),
    )
    write_csv(output_dir / "summary_by_track_model.csv", ["track", "model_label", "total", "success", "asr_pct"], success_by(divi_rows, ["track", "model_label"]))
    write_csv(
        output_dir / "summary_by_track_model_no_hard_artifacts.csv",
        ["track", "model_label", "total", "success", "asr_pct"],
        success_by([row for row in divi_rows if row.get("hard_artifact_candidate") != "1"], ["track", "model_label"]),
    )
    write_csv(
        output_dir / "summary_by_track_a_attack_method.csv",
        ["track", "attack_method_norm", "total", "success", "asr_pct"],
        success_by([row for row in divi_rows if row.get("track") == "A"], ["track", "attack_method_norm"]),
    )
    write_csv(
        output_dir / "summary_by_track_b_mutation.csv",
        ["track", "phase", "mutation", "total", "success", "asr_pct"],
        success_by([row for row in divi_rows if row.get("track") == "B"], ["track", "phase", "mutation"]),
    )
    write_csv(output_dir / "counts_by_classification.csv", ["track", "phase", "second_classification", "total"], count_by(divi_rows, ["track", "phase", "second_classification"]))
    write_csv(
        output_dir / "summary_by_hard_artifact_candidate.csv",
        ["hard_artifact_candidate", "track", "phase", "model_label", "total", "success", "asr_pct"],
        success_by(divi_rows, ["hard_artifact_candidate", "track", "phase", "model_label"]),
    )
    write_csv(
        output_dir / "summary_by_review_candidate.csv",
        ["review_candidate", "track", "phase", "model_label", "total", "success", "asr_pct"],
        success_by(divi_rows, ["review_candidate", "track", "phase", "model_label"]),
    )

    hard_artifact_count = sum(1 for row in divi_rows if row.get("hard_artifact_candidate") == "1")
    review_candidate_count = sum(1 for row in divi_rows if row.get("review_candidate") == "1")
    quality_rows = [
        {"metric": "source_rows", "value": str(len(source_rows))},
        {"metric": "divi_rows", "value": str(len(divi_rows))},
        {"metric": "rejected_rows", "value": str(len(rejected_rows))},
        {"metric": "hard_artifact_candidate_rows", "value": str(hard_artifact_count)},
        {"metric": "review_candidate_rows", "value": str(review_candidate_count)},
        {"metric": "divi_rows_without_hard_artifact_candidates", "value": str(len(divi_rows) - hard_artifact_count)},
        {"metric": "divi_rows_without_review_candidates", "value": str(len(divi_rows) - review_candidate_count)},
        {"metric": "embedding_column", "value": "response"},
        {"metric": "metadata_prompt_included", "value": "no"},
        {"metric": "compact_inspection_has_prompt_preview", "value": "yes"},
    ]
    write_csv(output_dir / "quality_summary.csv", ["metric", "value"], quality_rows)

    print(f"[DONE] output_dir={output_dir}")
    for row in quality_rows:
        print(f"[INFO] {row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()
