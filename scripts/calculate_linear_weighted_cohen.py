#!/usr/bin/env python3
"""Calculate exact agreement and linear weighted Cohen's kappa from an audit CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def linear_weighted_kappa(a: List[int], b: List[int], min_score: int = 1, max_score: int = 5) -> float:
    labels = list(range(min_score, max_score + 1))
    n = len(a)
    if n == 0:
        raise ValueError("No scored rows found.")

    observed = {(i, j): 0 for i in labels for j in labels}
    a_counts = {i: 0 for i in labels}
    b_counts = {i: 0 for i in labels}

    for x, y in zip(a, b):
        observed[(x, y)] += 1
        a_counts[x] += 1
        b_counts[y] += 1

    max_distance = max_score - min_score

    def weight(i: int, j: int) -> float:
        return abs(i - j) / max_distance

    observed_disagreement = sum(weight(i, j) * observed[(i, j)] / n for i in labels for j in labels)
    expected_disagreement = sum(
        weight(i, j) * (a_counts[i] / n) * (b_counts[j] / n) for i in labels for j in labels
    )

    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1.0 - observed_disagreement / expected_disagreement


def load_scores(path: Path, human_col: str, judge_col: str) -> Tuple[List[int], List[int], int]:
    human: List[int] = []
    judge: List[int] = []
    skipped = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            h = row.get(human_col, "").strip()
            j = row.get(judge_col, "").strip()
            if not h or not j:
                skipped += 1
                continue
            human.append(int(h))
            judge.append(int(j))
    return human, judge, skipped


def score_counts(scores: Iterable[int], min_score: int = 1, max_score: int = 5) -> Dict[int, int]:
    return {score: sum(1 for value in scores if value == score) for score in range(min_score, max_score + 1)}


def confusion_matrix(human: List[int], judge: List[int], min_score: int = 1, max_score: int = 5) -> Dict[Tuple[int, int], int]:
    labels = range(min_score, max_score + 1)
    matrix = {(h, j): 0 for h in labels for j in labels}
    for h, j in zip(human, judge):
        matrix[(h, j)] += 1
    return matrix


def bootstrap_ci(
    human: List[int],
    judge: List[int],
    iterations: int,
    seed: int,
    min_score: int = 1,
    max_score: int = 5,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(human)
    estimates: List[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_human = [human[i] for i in indices]
        sample_judge = [judge[i] for i in indices]
        estimates.append(linear_weighted_kappa(sample_human, sample_judge, min_score, max_score))

    estimates.sort()

    def percentile(p: float) -> float:
        if len(estimates) == 1:
            return estimates[0]
        position = p * (len(estimates) - 1)
        low = int(position)
        high = min(low + 1, len(estimates) - 1)
        fraction = position - low
        return estimates[low] * (1 - fraction) + estimates[high] * fraction

    return percentile(0.025), percentile(0.975)


def write_summary(path: Path, rows: List[Tuple[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_confusion(path: Path, human: List[int], judge: List[int], min_score: int = 1, max_score: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(min_score, max_score + 1))
    matrix = confusion_matrix(human, judge, min_score, max_score)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["human_score\\judge_score", *labels])
        for h in labels:
            writer.writerow([h, *[matrix[(h, j)] for j in labels]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_csv")
    parser.add_argument("--human_col", default="human_score")
    parser.add_argument("--judge_col", default="judge_score")
    parser.add_argument("--output_csv")
    parser.add_argument("--confusion_csv")
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260602)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human, judge, skipped = load_scores(Path(args.audit_csv), args.human_col, args.judge_col)
    if not human:
        print("Scored rows: 0")
        print(f"Skipped rows: {skipped}")
        print(f"Fill the `{args.human_col}` column with 1-5 scores before calculating kappa.")
        return
    exact = sum(h == j for h, j in zip(human, judge)) / len(human)
    kappa = linear_weighted_kappa(human, judge)
    ci_low, ci_high = bootstrap_ci(human, judge, args.bootstrap_iterations, args.seed)
    print(f"Scored rows: {len(human)}")
    print(f"Skipped rows: {skipped}")
    print(f"Exact agreement: {exact:.4f}")
    print(f"Linear weighted Cohen's kappa: {kappa:.4f}")
    print(f"Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    if args.output_csv:
        summary_rows: List[Tuple[str, object]] = [
            ("audit_csv", args.audit_csv),
            ("human_col", args.human_col),
            ("judge_col", args.judge_col),
            ("scored_rows", len(human)),
            ("skipped_rows", skipped),
            ("exact_agreement", f"{exact:.6f}"),
            ("linear_weighted_cohen_kappa", f"{kappa:.6f}"),
            ("bootstrap_iterations", args.bootstrap_iterations),
            ("bootstrap_seed", args.seed),
            ("bootstrap_95ci_low", f"{ci_low:.6f}"),
            ("bootstrap_95ci_high", f"{ci_high:.6f}"),
        ]
        for score, count in score_counts(human).items():
            summary_rows.append((f"human_score_{score}", count))
        for score, count in score_counts(judge).items():
            summary_rows.append((f"judge_score_{score}", count))
        write_summary(Path(args.output_csv), summary_rows)

    if args.confusion_csv:
        write_confusion(Path(args.confusion_csv), human, judge)


if __name__ == "__main__":
    main()
