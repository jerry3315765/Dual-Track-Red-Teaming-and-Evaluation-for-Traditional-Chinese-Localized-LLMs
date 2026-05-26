import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clean_text(value: str) -> str:
    return (value or "").replace("\x00", " ").strip()


def load_final_rows(csv_path: Path, max_samples: int = 0) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            response = clean_text(row.get("response", ""))
            if not response:
                continue
            item = {
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
            }
            rows.append(item)
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def add_import_paths(bundle_root: Path) -> None:
    divi_path = bundle_root / "track_a" / "src"
    if str(divi_path) not in os.sys.path:
        os.sys.path.append(str(divi_path))


def run_divi(
    embeddings: np.ndarray,
    seed: int,
    bundle_root: Path,
    max_epochs: int,
    split_interval: int,
    split_init: str,
    split_threshold: float | None,
):
    add_import_paths(bundle_root)
    from DIVI.DIVI_V2 import DIVIClustering, DiagnosableGMM, set_seed
    from sklearn.cluster import KMeans
    import torch
    import torch.optim as optim

    set_seed(seed)

    class DataDrivenDIVIClustering(DIVIClustering):
        def _deterministic_assignments(self, x_tensor):
            with torch.no_grad():
                q_phi = torch.sigmoid(self.model.phi_logits).unsqueeze(0)
                x_exp = x_tensor.unsqueeze(1)
                mu_exp = self.model.q_mu.unsqueeze(0)
                logvar = torch.clamp(self.model.q_logvar, min=-5.0, max=5.0).unsqueeze(0)
                log_prob_cluster = -0.5 * (
                    np.log(2 * np.pi) + logvar + (x_exp - mu_exp) ** 2 / torch.exp(logvar)
                )
                log_prob_bg = -0.5 * (
                    np.log(2 * np.pi)
                    + self.model.prior_logvar_0
                    + (x_exp - self.model.prior_mu_0) ** 2 / torch.exp(self.model.prior_logvar_0)
                )
                log_p = (
                    q_phi.unsqueeze(1) * log_prob_cluster
                    + (1 - q_phi.unsqueeze(1)) * log_prob_bg
                ).sum(dim=2)
                return torch.argmax(log_p, dim=1)

        def _expand_model_with_data(self, old_model, split_idx, x_tensor):
            d, k = old_model.D, old_model.K
            z_hard = self._deterministic_assignments(x_tensor)
            split_points = x_tensor[z_hard == split_idx].detach().cpu().numpy()

            if len(split_points) >= 2:
                km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(split_points)
                split_means = torch.tensor(km.cluster_centers_, dtype=torch.float32)
                split_logvars = []
                for local_label in range(2):
                    member_points = split_points[km.labels_ == local_label]
                    if len(member_points) < 2:
                        member_points = split_points
                    var = np.var(member_points, axis=0) + 1e-4
                    split_logvars.append(torch.tensor(np.log(var), dtype=torch.float32))
                split_logvars = torch.stack(split_logvars)
            else:
                target_mu = old_model.q_mu.data[split_idx]
                target_logvar = old_model.q_logvar.data[split_idx]
                split_means = torch.stack([
                    target_mu + torch.randn(d) * 0.2,
                    target_mu - torch.randn(d) * 0.2,
                ])
                split_logvars = torch.stack([target_logvar, target_logvar])

            new_model = DiagnosableGMM(d, k + 1, old_model.prior_phi_probs, temperature=old_model.temperature)
            with torch.no_grad():
                new_model.phi_logits.copy_(old_model.phi_logits)
                keep_idx = [i for i in range(k) if i != split_idx]
                if keep_idx:
                    new_mus = torch.cat([old_model.q_mu.data[keep_idx], split_means], dim=0)
                    new_logvars = torch.cat([old_model.q_logvar.data[keep_idx], split_logvars], dim=0)
                else:
                    new_mus = split_means
                    new_logvars = split_logvars
                new_model.q_mu.copy_(new_mus)
                new_model.q_logvar.copy_(torch.clamp(new_logvars, min=-5.0, max=5.0))
            return new_model

        def fit(self, x_np):
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            _, d = x_tensor.shape

            if self.split_threshold is None:
                self.split_threshold = 0.5 * d * (1 + np.log(2 * np.pi) + np.log(1.0))
                if self.verbose:
                    print(f"Auto-configured Split Threshold: {self.split_threshold:.2f}")

            rho = self._step_a_calculate_prior(x_np)
            global_mean = torch.mean(x_tensor, dim=0, keepdim=True)
            self.model = DiagnosableGMM(d, 1, rho, init_means=global_mean)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            if self.verbose:
                print("Starting Training (Initial K=1)...")

            for epoch in range(1, self.max_epochs + 1):
                self.model.train()
                optimizer.zero_grad()
                loss, q_phi, _ = self.model(x_tensor)
                loss.backward()
                optimizer.step()

                self.model.temperature = max(0.1, self.model.temperature * 0.98)
                self.history["loss"].append(loss.item())
                self.history["k"].append(self.model.K)
                self.history["phi"].append(q_phi.detach().numpy())

                if epoch % self.split_interval == 0:
                    scores = self.model.get_cluster_diagnostics(x_tensor)
                    worst_idx = int(np.argmax(scores))
                    worst_score = scores[worst_idx]
                    current_threshold = self.split_threshold
                    if self.cluster_split_penalty is not None:
                        current_threshold *= self.cluster_split_penalty

                    if self.verbose:
                        print(f"Epoch {epoch}: K={self.model.K}, Max NLL={worst_score:.2f} (Threshold: {current_threshold:.2f})")

                    if worst_score > current_threshold:
                        if self.verbose:
                            print(f"   >>> Splitting Cluster {worst_idx} with data-driven centroids...")
                        self.model = self._expand_model_with_data(self.model, worst_idx, x_tensor)
                        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            if self.verbose:
                print("Training Completed.")
            return self

    effective_epochs = max_epochs
    if split_interval > 0 and max_epochs % split_interval == 0:
        # Avoid ending immediately after a split, before the newly expanded model has
        # any chance to adapt its means and variances.
        effective_epochs = max_epochs + max(5, split_interval // 2)
        print(f"Adjusted max_epochs from {max_epochs} to {effective_epochs} to train after the final split.")
    clustering_cls = DIVIClustering if split_init == "original" else DataDrivenDIVIClustering
    if split_init == "original":
        print("Using original DIVI split initialization: duplicated mean + random perturbation.")
    else:
        print("Using data-driven DIVI split initialization: parent-cluster 2-means centroids.")

    divi = clustering_cls(
        split_interval=split_interval,
        split_threshold=split_threshold,
        cluster_split_penalty=1.0,
        max_epochs=effective_epochs,
        verbose=True,
    )
    divi.fit(embeddings)
    return divi


def prepare_divi_features(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler, normalize

    normalized = normalize(embeddings, norm="l2")
    return StandardScaler().fit_transform(normalized).astype(np.float32)


def infer_labels(divi, embeddings: np.ndarray) -> np.ndarray:
    import torch

    x_tensor = torch.tensor(embeddings, dtype=torch.float32)
    divi.model.eval()
    with torch.no_grad():
        model = divi.model
        q_phi = torch.sigmoid(model.phi_logits).unsqueeze(0)
        x_exp = x_tensor.unsqueeze(1)
        mu_exp = model.q_mu.unsqueeze(0)
        logvar = torch.clamp(model.q_logvar, min=-5.0, max=5.0).unsqueeze(0)

        log_prob_cluster = -0.5 * (
            np.log(2 * np.pi) + logvar + (x_exp - mu_exp) ** 2 / torch.exp(logvar)
        )
        log_prob_bg = -0.5 * (
            np.log(2 * np.pi)
            + model.prior_logvar_0
            + (x_exp - model.prior_mu_0) ** 2 / torch.exp(model.prior_logvar_0)
        )
        log_p_x_given_z = (
            q_phi.unsqueeze(1) * log_prob_cluster
            + (1 - q_phi.unsqueeze(1)) * log_prob_bg
        ).sum(dim=2)
        if hasattr(divi.model, "pi_logits"):
            pi = torch.softmax(divi.model.pi_logits, dim=0)
        else:
            pi = torch.ones(divi.model.K) / divi.model.K
        log_joint = log_p_x_given_z + torch.log(pi + 1e-9).unsqueeze(0)
        labels = torch.argmax(log_joint, dim=1).detach().cpu().numpy()
    return labels


def pct(num: int, den: int) -> float:
    return round((100.0 * num / den), 2) if den else 0.0


def top_counts(values: Iterable[str], top_k: int = 5) -> List[Dict]:
    counter = Counter(v for v in values if str(v).strip())
    return [{"value": k, "count": v} for k, v in counter.most_common(top_k)]


def summarize_clusters(rows: List[Dict], labels: np.ndarray, top_k: int = 5) -> List[Dict]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for row, label in zip(rows, labels):
        grouped[int(label)].append(row)

    summary = []
    for cid in sorted(grouped):
        items = grouped[cid]
        total = len(items)
        success = sum(1 for item in items if item["second_success"])
        summary.append({
            "cluster": cid,
            "total": total,
            "success": success,
            "asr_pct": pct(success, total),
            "track_counts": top_counts([item["track"] for item in items], top_k),
            "model_counts": top_counts([item["model_label"] for item in items], top_k),
            "scenario_counts": top_counts([item["scenario_id"] for item in items], top_k),
            "mutation_counts": top_counts([item["mutation"] or item["attack_method"] for item in items], top_k),
            "classification_counts": top_counts([item["second_classification"] for item in items], top_k),
        })
    return summary


def write_assignments(path: Path, rows: List[Dict], labels: np.ndarray) -> None:
    fieldnames = [
        "index", "cluster", "track", "model_label", "phase", "scenario_id", "turn_index",
        "mutation", "attack_method", "second_score", "second_success", "second_classification", "row_id",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (row, label) in enumerate(zip(rows, labels)):
            writer.writerow({
                "index": idx,
                "cluster": int(label),
                "track": row["track"],
                "model_label": row["model_label"],
                "phase": row["phase"],
                "scenario_id": row["scenario_id"],
                "turn_index": row["turn_index"],
                "mutation": row["mutation"],
                "attack_method": row["attack_method"],
                "second_score": row["second_score"],
                "second_success": int(row["second_success"]),
                "second_classification": row["second_classification"],
                "row_id": row["row_id"],
            })


def write_cluster_summary_csv(path: Path, summary: List[Dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster", "total", "success", "asr_pct",
            "top_tracks", "top_models", "top_scenarios", "top_mutations", "top_classifications",
        ])
        for item in summary:
            writer.writerow([
                item["cluster"],
                item["total"],
                item["success"],
                item["asr_pct"],
                json.dumps(item["track_counts"], ensure_ascii=False),
                json.dumps(item["model_counts"], ensure_ascii=False),
                json.dumps(item["scenario_counts"], ensure_ascii=False),
                json.dumps(item["mutation_counts"], ensure_ascii=False),
                json.dumps(item["classification_counts"], ensure_ascii=False),
            ])


def run_shap_summary(
    output_path: Path,
    rows: List[Dict],
    labels: np.ndarray,
    embeddings: np.ndarray,
    embedder,
    min_cluster_size: int,
    high_risk_asr: float,
    samples_per_cluster: int,
) -> None:
    import shap

    masker = shap.maskers.Text(embedder.tokenizer)
    cluster_ids = sorted(set(int(x) for x in labels))
    shap_summary = {}

    for cid in cluster_ids:
        indices = [i for i, label in enumerate(labels) if int(label) == cid]
        if len(indices) < min_cluster_size:
            continue
        success = sum(1 for i in indices if rows[i]["second_success"])
        asr = pct(success, len(indices))
        if asr < high_risk_asr:
            continue

        sample_indices = indices[:samples_per_cluster]
        sample_texts = [rows[i]["response"] for i in sample_indices]
        dim_means = np.abs(embeddings[indices].mean(axis=0))
        top_dims = np.argsort(dim_means)[::-1][:2]

        features = []
        for dim_idx in top_dims:
            def predict_dim(text_batch, dim_idx=dim_idx):
                emb = embedder.encode(list(text_batch), show_progress_bar=False)
                return emb[:, dim_idx]

            explainer = shap.Explainer(predict_dim, masker)
            shap_values = explainer(sample_texts)

            token_impacts = {}
            for i in range(len(sample_texts)):
                tokens = shap_values[i].data
                values = shap_values[i].values
                for tok, val in zip(tokens, values):
                    token = str(tok).strip()
                    if not token or token in {"User", "Model", "\n"}:
                        continue
                    token_impacts[token] = token_impacts.get(token, 0.0) + float(val)

            top_tokens = sorted(token_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            features.append({
                "dimension": int(dim_idx),
                "top_tokens": [{"token": token, "impact": impact} for token, impact in top_tokens],
            })

        shap_summary[f"cluster_{cid}"] = {
            "cluster": cid,
            "total": len(indices),
            "success": success,
            "asr_pct": asr,
            "sample_row_ids": [rows[i]["row_id"] for i in sample_indices],
            "features": features,
        }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(shap_summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run response-only DIVI-SHAP on final second-judge corpus.")
    parser.add_argument("--input_csv", default="results/second_judge/ab_second_judge_full_final_clean.csv")
    parser.add_argument("--output_dir", default="analysis/divi_shap_response_only")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--split_interval", type=int, default=10)
    parser.add_argument(
        "--split_threshold",
        default="-1e9",
        help="Use a float threshold for deterministic split sweeps, or 'auto' for DIVI's NLL threshold.",
    )
    parser.add_argument("--skip_shap", action="store_true")
    parser.add_argument("--reuse_embeddings", action="store_true")
    parser.add_argument("--reuse_clusters", action="store_true")
    parser.add_argument("--split_init", choices=["original", "data_driven"], default="data_driven")
    parser.add_argument("--shap_samples_per_cluster", type=int, default=8)
    parser.add_argument("--shap_min_cluster_size", type=int, default=100)
    parser.add_argument("--shap_high_risk_asr", type=float, default=50.0)
    args = parser.parse_args()

    split_threshold = None if args.split_threshold == "auto" else float(args.split_threshold)

    bundle_root = Path(__file__).resolve().parents[1]
    input_csv = bundle_root / args.input_csv
    output_dir = bundle_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "response_embeddings.npy"
    metadata_path = output_dir / "response_metadata.json"

    if args.reuse_embeddings and embeddings_path.exists() and metadata_path.exists():
        print(f"Reusing embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        with metadata_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        texts = [row["response"] for row in rows]

        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    else:
        print(f"Loading final corpus from {input_csv}")
        rows = load_final_rows(input_csv, args.max_samples)
        print(f"Loaded {len(rows)} response-only rows")
        if not rows:
            raise SystemExit("No rows loaded.")

        texts = [row["response"] for row in rows]

        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        embeddings = np.asarray(embedder.encode(texts, show_progress_bar=True), dtype=np.float32)

        np.save(embeddings_path, embeddings)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    divi_features = prepare_divi_features(embeddings)
    np.save(output_dir / "response_divi_features.npy", divi_features)

    labels_path = output_dir / "cluster_labels.npy"
    if args.reuse_clusters and labels_path.exists():
        print(f"Reusing cluster labels from {labels_path}")
        labels = np.load(labels_path)
    else:
        divi = run_divi(
            divi_features,
            args.seed,
            bundle_root,
            args.max_epochs,
            args.split_interval,
            args.split_init,
            split_threshold,
        )
        labels = infer_labels(divi, divi_features)
        np.save(labels_path, labels)

    clustered = []
    for row, label in zip(rows, labels):
        item = dict(row)
        item["cluster"] = int(label)
        clustered.append(item)

    with (output_dir / "clustered_traces_response_only.json").open("w", encoding="utf-8") as f:
        json.dump(clustered, f, ensure_ascii=False, indent=2)

    write_assignments(output_dir / "cluster_assignments_response_only.csv", rows, labels)
    summary = summarize_clusters(rows, labels)
    with (output_dir / "cluster_summary_response_only.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_cluster_summary_csv(output_dir / "cluster_summary_response_only.csv", summary)

    if not args.skip_shap:
        try:
            run_shap_summary(
                output_dir / "shap_response_only.json",
                rows,
                labels,
                divi_features,
                embedder,
                args.shap_min_cluster_size,
                args.shap_high_risk_asr,
                args.shap_samples_per_cluster,
            )
        except Exception as exc:
            print(f"SHAP analysis skipped: {exc}")

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
