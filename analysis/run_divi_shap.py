import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_track_a_results(track_a_dir: Path) -> List[Dict]:
    traces = []
    if not track_a_dir.exists():
        return traces

    for json_path in track_a_dir.glob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        for item in data:
            traces.append({
                "source": "Track A",
                "model": item.get("model", "unknown"),
                "prompt": item.get("prompt", ""),
                "response": item.get("response", ""),
                "success": bool(item.get("evaluation", {}).get("success", False)),
            })

    return traces


def parse_track_b_row(row: Dict) -> Optional[List[Dict]]:
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    result = row.get("result", None)

    if result is not None:
        # Long format row
        return [{
            "prompt": prompt,
            "response": response,
            "success": str(result).strip() in {"1", "True", "true"},
        }]

    # Wide format row
    try:
        responses = json.loads(row.get("response", "[]"))
        results = json.loads(row.get("results", "[]"))
    except Exception:
        responses = [row.get("response", "")]
        results = [0]

    traces = []
    for resp, res in zip(responses, results):
        traces.append({
            "prompt": prompt,
            "response": resp,
            "success": bool(res),
        })

    return traces


def load_track_b_results(track_b_dir: Path) -> List[Dict]:
    traces = []
    if not track_b_dir.exists():
        return traces

    csv_files = list(track_b_dir.rglob("*_long.csv"))
    if not csv_files:
        csv_files = list(track_b_dir.rglob("*.csv"))

    for csv_path in csv_files:
        model_name = "unknown"
        parts = csv_path.parts
        if "baseline" in parts:
            idx = parts.index("baseline")
            if idx + 1 < len(parts):
                model_name = parts[idx + 1]

        try:
            with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = parse_track_b_row(row)
                    if not parsed:
                        continue
                    for trace in parsed:
                        trace["source"] = "Track B"
                        trace["model"] = model_name
                        traces.append(trace)
        except Exception:
            continue

    return traces


def run_divi(embeddings: np.ndarray, seed: int, bundle_root: Path):
    divi_path = bundle_root / "track_a" / "src"
    if str(divi_path) not in os.sys.path:
        os.sys.path.append(str(divi_path))

    from DIVI.DIVI_V2 import DIVIClustering, set_seed

    set_seed(seed)
    divi = DIVIClustering(
        split_interval=10,
        split_threshold=-1e9,
        cluster_split_penalty=1.0,
        max_epochs=200,
        verbose=True,
    )
    divi.fit(embeddings)
    return divi


def main():
    parser = argparse.ArgumentParser(description="Run DIVI + SHAP clustering")
    parser.add_argument("--track_a_results", default="track_a/data/results")
    parser.add_argument("--track_b_results", default="track_b/Results")
    parser.add_argument("--output_dir", default="analysis/divi_shap")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shap_samples_per_cluster", type=int, default=10)
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parents[1]
    track_a_dir = bundle_root / args.track_a_results
    track_b_dir = bundle_root / args.track_b_results
    output_dir = bundle_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    track_a_traces = load_track_a_results(track_a_dir)
    track_b_traces = load_track_b_results(track_b_dir)
    combined = track_a_traces + track_b_traces

    if not combined:
        print("No traces found.")
        return

    if args.max_samples and len(combined) > args.max_samples:
        combined = combined[: args.max_samples]

    texts = [f"User: {t['prompt']}\nModel: {t['response']}" for t in combined]

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True)

    divi = run_divi(np.asarray(embeddings), args.seed, bundle_root)

    import torch
    X_tensor = torch.tensor(embeddings, dtype=torch.float32)
    divi.model.eval()
    with torch.no_grad():
        _, _, log_p_x_given_z = divi.model(X_tensor)
        if hasattr(divi.model, "pi_logits"):
            pi = torch.softmax(divi.model.pi_logits, dim=0)
        else:
            pi = torch.ones(divi.model.K) / divi.model.K
        log_joint = log_p_x_given_z + torch.log(pi + 1e-9).unsqueeze(0)
        labels = torch.argmax(log_joint, dim=1).detach().cpu().numpy()

    clustered = []
    for trace, label in zip(combined, labels):
        item = dict(trace)
        item["cluster"] = int(label)
        clustered.append(item)

    clustered_path = output_dir / "clustered_traces.json"
    with clustered_path.open("w", encoding="utf-8") as f:
        json.dump(clustered, f, ensure_ascii=False, indent=2)

    assign_path = output_dir / "cluster_assignments.csv"
    with assign_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "cluster", "source", "model", "success"])
        for idx, item in enumerate(clustered):
            writer.writerow([idx, item["cluster"], item["source"], item["model"], int(item["success"])])

    try:
        import shap

        masker = shap.maskers.Text(embedder.tokenizer)
        shap_summary = {}
        cluster_ids = sorted(set(labels))

        for cid in cluster_ids:
            indices = [i for i, c in enumerate(labels) if c == cid]
            if not indices:
                continue
            sample_indices = indices[: args.shap_samples_per_cluster]
            sample_texts = [texts[i] for i in sample_indices]

            if not sample_texts:
                continue

            dim_means = np.abs(np.asarray(embeddings)[indices].mean(axis=0))
            top_dims = np.argsort(dim_means)[::-1][:2]

            shap_features = []
            for dim_idx in top_dims:
                def predict_dim(text_batch):
                    emb = embedder.encode(text_batch)
                    return emb[:, dim_idx]

                explainer = shap.Explainer(predict_dim, masker)
                shap_values = explainer(sample_texts)

                token_impacts = {}
                for i in range(len(sample_texts)):
                    tokens = shap_values[i].data
                    values = shap_values[i].values
                    for tok, val in zip(tokens, values):
                        token = tok.strip()
                        if not token:
                            continue
                        token_impacts[token] = token_impacts.get(token, 0.0) + float(val)

                top_tokens = sorted(token_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                shap_features.append({
                    "dimension": int(dim_idx),
                    "top_tokens": [{"token": t, "impact": v} for t, v in top_tokens],
                })

            shap_summary[f"cluster_{cid}"] = {
                "sample_count": len(indices),
                "sample_texts": sample_texts,
                "features": shap_features,
            }

        shap_path = output_dir / "shap_restore.json"
        with shap_path.open("w", encoding="utf-8") as f:
            json.dump(shap_summary, f, ensure_ascii=False, indent=2)

        print(f"Saved SHAP restore output to {shap_path}")

    except Exception as e:
        print(f"SHAP analysis skipped: {e}")

    print(f"Saved clustered traces to {clustered_path}")
    print(f"Saved assignments to {assign_path}")


if __name__ == "__main__":
    main()
