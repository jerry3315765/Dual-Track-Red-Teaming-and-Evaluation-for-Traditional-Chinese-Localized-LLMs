#!/usr/bin/env python3
"""Create a stratified human-audit sample for linear weighted Cohen's kappa."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_INPUT = (
    Path("data")
    / "divi_shap_input"
    / "divi_shap_response_only_input_no_hard_artifact_candidates.csv"
)
DEFAULT_OUTPUT = (
    Path("results")
    / "second_judge"
    / "human_audit"
    / "linear_weighted_cohen_sample_20260602_seed20260602.csv"
)
DEFAULT_SUMMARY = (
    Path("results")
    / "second_judge"
    / "human_audit"
    / "linear_weighted_cohen_sample_20260602_seed20260602_summary.csv"
)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_sample(rows: List[Dict[str, str]], per_score: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        score = row.get("second_score", "").strip()
        if row.get("track") == "A" and score in {"1", "2", "3", "4", "5"}:
            groups[score].append(row)

    missing = {score: len(items) for score, items in groups.items() if len(items) < per_score}
    if missing:
        raise ValueError(f"Not enough rows for requested per-score sample: {missing}")

    sampled: List[Dict[str, str]] = []
    for score in ["1", "2", "3", "4", "5"]:
        chosen = rng.sample(groups[score], per_score)
        sampled.extend(chosen)

    rng.shuffle(sampled)
    return sampled


def audit_rows(sampled: List[Dict[str, str]]) -> List[Dict[str, str]]:
    output = []
    for i, row in enumerate(sampled, start=1):
        output.append(
            {
                "audit_id": f"LWK-20260602-{i:03d}",
                "human_score": "",
                "human_notes": "",
                "judge_score": row.get("second_score", ""),
                "judge_classification": row.get("second_classification", ""),
                "judge_reason": row.get("second_reason", ""),
                "row_id": row.get("row_id", ""),
                "track": row.get("track", ""),
                "model_label": row.get("model_label", ""),
                "scenario_id": row.get("scenario_id", ""),
                "template_id": row.get("template_id", ""),
                "turn_index": row.get("turn_index", ""),
                "system_prompt_name": row.get("system_prompt_name", ""),
                "attack_method": row.get("attack_method", ""),
                "prompt": row.get("prompt", ""),
                "response": row.get("response", ""),
                "orig_score": row.get("orig_score", ""),
                "orig_classification": row.get("orig_classification", ""),
                "second_prompt_intent": row.get("second_prompt_intent", ""),
                "second_attack_valid": row.get("second_attack_valid", ""),
                "second_response_harmfulness": row.get("second_response_harmfulness", ""),
            }
        )
    return output


def summary_rows(
    sampled: List[Dict[str, str]], source_rows: List[Dict[str, str]], seed: int, source_file: Path
) -> List[Dict[str, str]]:
    source_scores = Counter(
        row.get("second_score", "").strip()
        for row in source_rows
        if row.get("track") == "A" and row.get("second_score", "").strip() in {"1", "2", "3", "4", "5"}
    )
    sample_scores = Counter(row.get("second_score", "").strip() for row in sampled)
    sample_models = Counter(row.get("model_label", "") for row in sampled)
    sample_scenarios = Counter(row.get("scenario_id", "") for row in sampled)

    rows = [
        {"metric": "seed", "value": str(seed)},
        {"metric": "source_file", "value": str(source_file)},
        {
            "metric": "sample_design",
            "value": "Final no-hard-artifact corpus; Track A only; stratified equally by second_score 1-5",
        },
        {"metric": "sample_size", "value": str(len(sampled))},
    ]
    for score in ["1", "2", "3", "4", "5"]:
        rows.append({"metric": f"source_track_a_score_{score}", "value": str(source_scores[score])})
        rows.append({"metric": f"sample_score_{score}", "value": str(sample_scores[score])})
    for model, count in sorted(sample_models.items()):
        rows.append({"metric": f"sample_model_{model}", "value": str(count)})
    for scenario, count in sorted(sample_scenarios.items()):
        rows.append({"metric": f"sample_scenario_{scenario}", "value": str(count)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary_csv", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--per_score", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260602)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    source_rows = read_rows(input_csv)
    sampled = make_sample(source_rows, per_score=args.per_score, seed=args.seed)
    output_rows = audit_rows(sampled)

    write_csv(
        Path(args.output_csv),
        [
            "audit_id",
            "human_score",
            "human_notes",
            "judge_score",
            "judge_classification",
            "judge_reason",
            "row_id",
            "track",
            "model_label",
            "scenario_id",
            "template_id",
            "turn_index",
            "system_prompt_name",
            "attack_method",
            "prompt",
            "response",
            "orig_score",
            "orig_classification",
            "second_prompt_intent",
            "second_attack_valid",
            "second_response_harmfulness",
        ],
        output_rows,
    )
    write_csv(Path(args.summary_csv), ["metric", "value"], summary_rows(sampled, source_rows, args.seed, input_csv))

    print(f"Wrote {len(output_rows)} audit rows to {args.output_csv}")
    print(f"Wrote summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
