#!/usr/bin/env python3
"""Calculate exact agreement and linear weighted Cohen's kappa from an audit CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_csv")
    parser.add_argument("--human_col", default="human_score")
    parser.add_argument("--judge_col", default="judge_score")
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
    print(f"Scored rows: {len(human)}")
    print(f"Skipped rows: {skipped}")
    print(f"Exact agreement: {exact:.4f}")
    print(f"Linear weighted Cohen's kappa: {kappa:.4f}")


if __name__ == "__main__":
    main()
