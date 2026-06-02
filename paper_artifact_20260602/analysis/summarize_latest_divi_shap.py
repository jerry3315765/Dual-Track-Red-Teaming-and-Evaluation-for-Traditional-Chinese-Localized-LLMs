import argparse
import csv
import json
import re
from pathlib import Path


PUNCT_OR_MARKUP = re.compile(r"^[\s\W_]+$", re.UNICODE)
COMMON_TOKENS = {
    "的", "了", "和", "與", "及", "或", "在", "是", "我", "你", "他", "她", "它", "們",
    "這", "那", "有", "會", "不", "可以", "可能", "如果", "請", "以下", "因此", "但是",
    "，", "。", "、", "：", "；", "！", "？", "「", "」", "『", "』", "（", "）", "(", ")",
    "**", "*", "-", "—", "\n",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_json_cell(value: str) -> list[dict]:
    if not value:
        return []
    return json.loads(value)


def is_content_token(token: str) -> bool:
    token = str(token).strip()
    if not token:
        return False
    if token in COMMON_TOKENS:
        return False
    if PUNCT_OR_MARKUP.match(token):
        return False
    return True


def top_content_tokens(features: list[dict], limit: int = 12) -> list[dict]:
    impacts: dict[str, float] = {}
    dimensions: dict[str, set[int]] = {}
    for feature in features:
        dim = int(feature.get("dimension", -1))
        for item in feature.get("top_tokens", []):
            token = str(item.get("token", "")).strip()
            if not is_content_token(token):
                continue
            impacts[token] = impacts.get(token, 0.0) + float(item.get("impact", 0.0))
            dimensions.setdefault(token, set()).add(dim)
    ranked = sorted(impacts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:limit]
    return [
        {"token": token, "impact": impact, "dimensions": sorted(dimensions[token])}
        for token, impact in ranked
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact reporting tables for the latest DIVI-SHAP run.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--track_summary_csv", required=True)
    parser.add_argument("--model_summary_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_rows = read_csv(run_dir / "cluster_summary_response_only.csv")
    shap = json.load((run_dir / "shap_response_only.json").open("r", encoding="utf-8"))
    track_rows = read_csv(Path(args.track_summary_csv))
    model_rows = read_csv(Path(args.model_summary_csv))

    cluster_by_id = {int(row["cluster"]): row for row in cluster_rows}

    with (output_dir / "latest_divi_cluster_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster", "total", "success", "asr_pct", "top_tracks", "top_models",
                "top_mutations", "top_classifications", "has_shap_summary", "content_shap_tokens",
            ],
        )
        writer.writeheader()
        for row in sorted(cluster_rows, key=lambda r: float(r["asr_pct"]), reverse=True):
            cid = int(row["cluster"])
            shap_item = shap.get(f"cluster_{cid}")
            tokens = top_content_tokens(shap_item.get("features", [])) if shap_item else []
            writer.writerow({
                "cluster": cid,
                "total": row["total"],
                "success": row["success"],
                "asr_pct": row["asr_pct"],
                "top_tracks": row["top_tracks"],
                "top_models": row["top_models"],
                "top_mutations": row["top_mutations"],
                "top_classifications": row["top_classifications"],
                "has_shap_summary": bool(shap_item),
                "content_shap_tokens": json.dumps(tokens, ensure_ascii=False),
            })

    with (output_dir / "latest_shap_tokens_raw.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cluster", "cluster_total", "cluster_asr_pct", "dimension", "rank", "token", "impact"],
        )
        writer.writeheader()
        for key, item in sorted(shap.items(), key=lambda kv: int(kv[0].split("_")[1])):
            cid = int(key.split("_")[1])
            for feature in item.get("features", []):
                for rank, token_item in enumerate(feature.get("top_tokens", []), 1):
                    writer.writerow({
                        "cluster": cid,
                        "cluster_total": item["total"],
                        "cluster_asr_pct": item["asr_pct"],
                        "dimension": feature["dimension"],
                        "rank": rank,
                        "token": token_item["token"],
                        "impact": token_item["impact"],
                    })

    with (output_dir / "latest_shap_content_tokens.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cluster", "cluster_total", "cluster_asr_pct", "rank", "token", "impact", "dimensions"],
        )
        writer.writeheader()
        for key, item in sorted(shap.items(), key=lambda kv: int(kv[0].split("_")[1])):
            cid = int(key.split("_")[1])
            for rank, token_item in enumerate(top_content_tokens(item.get("features", []), limit=20), 1):
                writer.writerow({
                    "cluster": cid,
                    "cluster_total": item["total"],
                    "cluster_asr_pct": item["asr_pct"],
                    "rank": rank,
                    "token": token_item["token"],
                    "impact": token_item["impact"],
                    "dimensions": json.dumps(token_item["dimensions"], ensure_ascii=False),
                })

    with (output_dir / "latest_track_phase_asr.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=track_rows[0].keys())
        writer.writeheader()
        writer.writerows(track_rows)

    with (output_dir / "latest_track_model_asr.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_rows[0].keys())
        writer.writeheader()
        writer.writerows(model_rows)

    high_risk = [
        row for row in cluster_rows
        if int(row["total"]) >= 100 and float(row["asr_pct"]) >= 50.0
    ]
    high_risk.sort(key=lambda r: float(r["asr_pct"]), reverse=True)

    lines = [
        "# Latest DIVI-SHAP Reporting Summary",
        "",
        "Corpus: `divi_shap_response_only_input_no_hard_artifact_candidates.csv`.",
        "Automatic exclusions include blank/API-error rows and hard tokenizer/byte artifacts only; meta/thinking style responses remain in the corpus.",
        "",
        "## Track/Phase ASR",
        "",
        "| Track | Phase | Total | Success | ASR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in track_rows:
        lines.append(f"| {row['track']} | {row['phase']} | {row['total']} | {row['success']} | {row['asr_pct']}% |")

    lines.extend([
        "",
        "## High-Risk DIVI Clusters",
        "",
        "| Cluster | Total | Success | ASR | Content SHAP tokens |",
        "|---:|---:|---:|---:|---|",
    ])
    for row in high_risk[:8]:
        cid = int(row["cluster"])
        shap_item = shap.get(f"cluster_{cid}")
        tokens = top_content_tokens(shap_item.get("features", []), limit=8) if shap_item else []
        token_text = "、".join(item["token"] for item in tokens)
        lines.append(f"| {cid} | {row['total']} | {row['success']} | {row['asr_pct']}% | {token_text} |")

    lines.extend([
        "",
        "Note: SHAP tokens are kept in their original form. The filtered content-token table removes punctuation and common function words only for readability; the raw token table is retained for audit.",
    ])
    (output_dir / "latest_divi_shap_reporting_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote reporting files to {output_dir}")
    print(f"Clusters: {len(cluster_rows)}; SHAP summarized clusters: {len(shap)}")
    print(f"High-risk clusters (n>=100, ASR>=50): {len(high_risk)}")


if __name__ == "__main__":
    main()
