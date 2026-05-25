import os
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, cohen_kappa_score
from sklearn.preprocessing import normalize, StandardScaler

# Paths
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "experiment_v2", "data", "results")
MERGED_EMBEDDED_CSV = os.path.join(WORKSPACE_ROOT, "code", "merged_results_embedded.csv")
DIVI_RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "code", "divi_results")
HUMAN_AUDIT_FILE = os.path.join(WORKSPACE_ROOT, "code", "human_audit_sample.csv")

OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "experiment_v2", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TURN_ASR_CSV = os.path.join(OUTPUT_DIR, "turn_asr.csv")
CLUSTER_PURITY_CSV = os.path.join(OUTPUT_DIR, "cluster_purity_by_seed.csv")
CLUSTER_QUALITY_CSV = os.path.join(OUTPUT_DIR, "cluster_quality_by_seed.csv")
IRR_STATS_TXT = os.path.join(WORKSPACE_ROOT, "code", "agreement_stats.txt")


def iter_result_items():
    json_pattern = os.path.join(RESULTS_DIR, "redteam_divi_results_*.json")
    files = glob.glob(json_pattern)
    for filepath in files:
        method_name = os.path.basename(filepath).replace("redteam_divi_results_", "").replace(".json", "")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                item["_method"] = method_name
                yield item
        except Exception as e:
            print(f"Error reading {filepath}: {e}")


def compute_turn_asr(items):
    df = pd.DataFrame(items)
    if df.empty:
        print("No result data found for turn-level ASR.")
        return

    if "turn" not in df.columns:
        print("Missing 'turn' field in results.")
        return

    df["success"] = df.get("success", False).fillna(False).astype(bool)
    turn_stats = df.groupby("turn")["success"].agg(["count", "sum", "mean"]).reset_index()
    turn_stats.columns = ["Turn", "Total", "Success", "ASR"]
    turn_stats["ASR(%)"] = (turn_stats["ASR"] * 100).round(2)
    turn_stats = turn_stats.sort_values("Turn")

    turn_stats.to_csv(TURN_ASR_CSV, index=False, encoding="utf-8-sig")
    print("Turn-level ASR saved to:", TURN_ASR_CSV)
    print(turn_stats.to_string(index=False))


def build_id_success_map(items):
    if not os.path.exists(MERGED_EMBEDDED_CSV):
        print(f"Missing file: {MERGED_EMBEDDED_CSV}")
        return {}, 0, 0

    use_cols = ["scenario_desc", "system_prompt_name", "turn", "prompt"]
    df_ref = pd.read_csv(MERGED_EMBEDDED_CSV, usecols=use_cols, encoding="utf-8-sig")
    df_ref["ID"] = df_ref.index

    def make_key(sys_name, scen_desc, turn_num, pmt_text):
        return (
            str(sys_name).strip(),
            str(scen_desc).strip(),
            int(turn_num),
            str(pmt_text).strip(),
        )

    lookup_map = defaultdict(list)
    for idx, row in df_ref.iterrows():
        key = make_key(row["system_prompt_name"], row["scenario_desc"], row["turn"], row["prompt"])
        lookup_map[key].append(idx)

    id_success_map = {}
    matched = 0
    unmatched = 0

    for item in items:
        sys_prompt = item.get("system_prompt_name") or item.get("_method") or "unknown"
        scenario = item.get("scenario_desc", "")
        turn = item.get("turn", 1)
        prompt = item.get("prompt", "")
        key = make_key(sys_prompt, scenario, turn, prompt)

        if key in lookup_map and lookup_map[key]:
            data_id = lookup_map[key].pop(0)
            id_success_map[data_id] = bool(item.get("success", False))
            matched += 1
        else:
            unmatched += 1

    print(f"ID mapping complete. Matched: {matched}, Unmatched: {unmatched}")
    return id_success_map, matched, unmatched


def load_feature_matrix():
    if not os.path.exists(MERGED_EMBEDDED_CSV):
        print(f"Missing file: {MERGED_EMBEDDED_CSV}")
        return None, None

    def use_cols(col_name):
        return col_name == "ID" or col_name.startswith("Feature_")

    df = pd.read_csv(MERGED_EMBEDDED_CSV, usecols=use_cols, encoding="utf-8-sig")
    if "ID" not in df.columns:
        df["ID"] = df.index

    feature_cols = [c for c in df.columns if c.startswith("Feature_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)

    # Match DIVI preprocessing
    X_norm = normalize(X, norm="l2")
    scaler = StandardScaler()
    X_input = scaler.fit_transform(X_norm)

    features_df = pd.DataFrame(X_input, columns=feature_cols)
    features_df["ID"] = df["ID"].values

    return features_df, feature_cols


def compute_cluster_quality(id_success_map):
    cluster_files = glob.glob(os.path.join(DIVI_RESULTS_DIR, "gibbs_cluster_assignments_seed*.csv"))
    if not cluster_files:
        print(f"No cluster assignment files found in {DIVI_RESULTS_DIR}")
        return

    features_df, feature_cols = load_feature_matrix()
    if features_df is None:
        return

    cluster_rows = []
    seed_rows = []

    for c_file in cluster_files:
        seed = os.path.basename(c_file).replace("gibbs_cluster_assignments_seed", "").replace(".csv", "")
        df_assign = pd.read_csv(c_file)
        if "ID" not in df_assign.columns or "Cluster" not in df_assign.columns:
            print(f"Skipping {c_file}: missing ID/Cluster")
            continue

        df_assign["Success"] = df_assign["ID"].map(id_success_map)
        df_valid = df_assign.dropna(subset=["Success"]).copy()
        if df_valid.empty:
            print(f"Seed {seed}: no matched Success labels.")
            continue

        df_valid["Success"] = df_valid["Success"].astype(bool)

        # Cluster purity (success-homogeneity)
        stats = df_valid.groupby("Cluster")["Success"].agg(["count", "sum", "mean"]).reset_index()
        stats["ASR"] = stats["mean"]
        stats["ASR(%)"] = (stats["ASR"] * 100).round(2)
        stats["Purity"] = stats["ASR"].apply(lambda x: max(x, 1 - x))
        stats["Purity(%)"] = (stats["Purity"] * 100).round(2)
        stats["Seed"] = seed

        for _, row in stats.iterrows():
            cluster_rows.append({
                "Seed": seed,
                "Cluster": row["Cluster"],
                "Sample_Count": int(row["count"]),
                "Success_Count": int(row["sum"]),
                "ASR(%)": float(row["ASR(%)"]),
                "Purity(%)": float(row["Purity(%)"]),
            })

        # Weighted average purity per seed
        weighted_purity = np.average(stats["Purity"], weights=stats["count"]) * 100

        # Silhouette score
        df_join = df_valid.merge(features_df, on="ID", how="inner")
        labels = df_join["Cluster"].to_numpy()
        X = df_join[feature_cols].to_numpy()

        if len(np.unique(labels)) < 2 or len(labels) < 2:
            silhouette = np.nan
        else:
            silhouette = float(silhouette_score(X, labels))

        seed_rows.append({
            "Seed": seed,
            "Samples": int(len(df_join)),
            "Clusters": int(len(np.unique(labels))),
            "Silhouette": silhouette,
            "Weighted_Purity(%)": round(weighted_purity, 2),
        })

    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(CLUSTER_PURITY_CSV, index=False, encoding="utf-8-sig")
        print("Cluster purity saved to:", CLUSTER_PURITY_CSV)

    if seed_rows:
        pd.DataFrame(seed_rows).to_csv(CLUSTER_QUALITY_CSV, index=False, encoding="utf-8-sig")
        print("Cluster quality saved to:", CLUSTER_QUALITY_CSV)


def compute_kappa():
    if not os.path.exists(HUMAN_AUDIT_FILE):
        print(f"Missing file: {HUMAN_AUDIT_FILE}")
        return

    df = pd.read_csv(HUMAN_AUDIT_FILE)

    # Support both old and new column names
    human_col = None
    judge_col = None
    for col in ["human_score", "score"]:
        if col in df.columns:
            human_col = col
            break
    for col in ["gpt4_score", "gpt_score"]:
        if col in df.columns:
            judge_col = col
            break

    # If judge score not in file, map from results JSON by (model, prompt)
    if judge_col is None:
        gpt_scores = {}
        for item in iter_result_items():
            model = item.get("model")
            prompt = str(item.get("prompt", "")).strip()
            eval_data = item.get("evaluation")
            score = 0
            if isinstance(eval_data, dict):
                score = eval_data.get("score", 0)
            if model and prompt and score:
                gpt_scores[(model, prompt)] = score

        model_col = "model_name" if "model_name" in df.columns else "model"
        df["_gpt_score"] = np.nan
        for idx, row in df.iterrows():
            model = row.get(model_col)
            prompt = str(row.get("prompt", "")).strip()
            gpt_val = gpt_scores.get((model, prompt))
            if gpt_val is not None:
                df.at[idx, "_gpt_score"] = gpt_val

        judge_col = "_gpt_score"

    if human_col is None or judge_col is None:
        print("Missing human/judge score columns in human audit file.")
        return

    df = df.dropna(subset=[human_col, judge_col])
    if df.empty:
        print("No valid rows for IRR computation.")
        return

    y_human = df[human_col].astype(int).to_numpy()
    y_judge = df[judge_col].astype(int).to_numpy()

    agreement = float(np.mean(y_human == y_judge))
    kappa = float(cohen_kappa_score(y_human, y_judge))
    kappa_weighted = float(cohen_kappa_score(y_human, y_judge, weights="linear"))

    with open(IRR_STATS_TXT, "w", encoding="utf-8") as f:
        f.write(f"Agreement: {agreement:.1%}\n")
        f.write(f"Cohen's Kappa: {kappa:.3f}\n")
        f.write(f"Weighted Kappa (Linear): {kappa_weighted:.3f}\n")

    print("IRR stats saved to:", IRR_STATS_TXT)
    print(f"Agreement: {agreement:.2%}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Weighted Kappa (Linear): {kappa_weighted:.4f}")


def main():
    items = list(iter_result_items())

    print("\n[1] Turn-level ASR")
    compute_turn_asr(items)

    print("\n[2] Human/AI IRR (Cohen's Kappa)")
    compute_kappa()

    print("\n[3] DIVI Cluster Quality (Silhouette & Purity)")
    id_success_map, _, _ = build_id_success_map(items)
    if id_success_map:
        compute_cluster_quality(id_success_map)


if __name__ == "__main__":
    main()
