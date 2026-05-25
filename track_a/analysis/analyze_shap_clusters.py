import shap
import pandas as pd
import numpy as np
import json
import os
import torch
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "merged_results_embedded.csv")
DIVI_RESULTS_DIR = os.path.join(BASE_DIR, "divi_results")
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# Analysis constraints
SAMPLES_PER_CLUSTER = 30   # Set to None to analyze ALL texts in the cluster (Warning: Can be very slow!)
TOP_FEATURES_TO_ANALYZE = 3  # Number of top features (dimensions) to explain per cluster
TOP_TOKENS_TO_RETURN = 5     # Number of top influencing tokens to record

def analyze_shap_for_clusters():
    print("="*60)
    print("Starting SHAP Analysis for DIVI Clusters")
    print("="*60)

    # 1. Load Data
    print(f"Loading data from {DATA_FILE}...")
    try:
        df_main = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error loading main CSV: {e}")
        return

    # Ensure ID map exists
    if 'ID' not in df_main.columns:
        df_main['ID'] = df_main.index
    
    # Map ID to Response Text
    id_to_text = dict(zip(df_main['ID'], df_main['response'].fillna("")))

    # 2. Load Model
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    model.max_seq_length = 512
    # We need the underlying tokenizer for SHAP
    tokenizer = model.tokenizer

    # 3. Find all Seed Results
    json_files = [f for f in os.listdir(DIVI_RESULTS_DIR) if f.startswith("gibbs_cluster_features_seed") and f.endswith(".json")]
    
    if not json_files:
        print("No cluster feature JSON files found in divi_results.")
        return

    print(f"Found {len(json_files)} result sets to analyze.")

    # 4. Iterate Over Each Seed
    for json_file in json_files:
        seed = json_file.replace("gibbs_cluster_features_seed", "").replace(".json", "")
        print(f"\nAnalyzing results for Seed {seed}...")
        
        # Load Features Info
        with open(os.path.join(DIVI_RESULTS_DIR, json_file), 'r', encoding='utf-8') as f:
            features_data = json.load(f)
            
        # Load Assignments
        assignment_file = f"gibbs_cluster_assignments_seed{seed}.csv"
        assignment_path = os.path.join(DIVI_RESULTS_DIR, assignment_file)
        if not os.path.exists(assignment_path):
            print(f"Skipping seed {seed}: Assignment file not found.")
            continue
            
        df_assign = pd.read_csv(assignment_path)
        
        analysis_results = {
            "metadata": features_data.get("metadata", {}),
            "shap_analysis": {}
        }
        
        # Iterate Over Clusters
        for cluster_id, cluster_info in features_data["clusters"].items():
            cid = int(cluster_id.replace("cluster_", ""))
            print(f"  > Processing Cluster {cid} ({cluster_info['percentage']:.1f}% of data)...")
            
            # Get IDs in this cluster
            cluster_doc_ids = df_assign[df_assign['Cluster'] == cid]['ID'].values
            
            # Sample texts
            if SAMPLES_PER_CLUSTER is not None and len(cluster_doc_ids) > SAMPLES_PER_CLUSTER:
                sampled_ids = np.random.choice(cluster_doc_ids, SAMPLES_PER_CLUSTER, replace=False)
            else:
                sampled_ids = cluster_doc_ids
                
            sampled_texts = [str(id_to_text[uid]) for uid in sampled_ids if uid in id_to_text]
            
            if not sampled_texts:
                continue

            # Identify Top Features to Explain
            # cluster_info['top_features'] contains lists of features sorted by importance
            top_features = cluster_info['top_features'][:TOP_FEATURES_TO_ANALYZE]
            
            cluster_shap_data = {
                "sampled_texts_count": len(sampled_texts),
                "features": []
            }

            # Create a masker for text
            masker = shap.maskers.Text(tokenizer)

            for feat_item in top_features:
                dim_idx = feat_item['feature_id'] - 1 # Adjust 1-based to 0-based
                feat_name = feat_item['feature_name']
                
                # Define prediction function for this specific dimension
                def predict_dim(texts):
                    # encode returns [N, 768]
                    # we want return [N] (scalar value of the specific dim)
                    emb = model.encode(texts)
                    return emb[:, dim_idx]

                # Run SHAP
                # We use a Permutation Explainer or simple Explainer with text masker
                try:
                    explainer = shap.Explainer(predict_dim, masker)
                    shap_values = explainer(sampled_texts)
                    
                    # shap_values is an object containing .values [samples, tokens], .data [samples, tokens]
                    
                    # Aggregate impact of tokens across samples
                    token_impacts = {}
                    
                    for i in range(len(sampled_texts)):
                        # For each sample
                        tokens = shap_values[i].data
                        values = shap_values[i].values
                        
                        for tok, val in zip(tokens, values):
                            cleaned_tok = tok.strip()
                            if not cleaned_tok: continue
                            
                            if cleaned_tok not in token_impacts:
                                token_impacts[cleaned_tok] = 0.0
                            # Summing the absolute contribution - magnitude of influence
                            token_impacts[cleaned_tok] += val

                    # Sort by impact value (positive or negative, keeping sign helps understand direction, 
                    # but usually for 'feature importance' in clustering, we care about what pushed it *towards* that value.
                    # Given the DIVI importance is based on |mean| * phi, the feature value might be consistently positive or negative.
                    # Here we simply sort by absolute magnitude of total impact.
                    
                    sorted_tokens = sorted(token_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
                    top_tokens = sorted_tokens[:TOP_TOKENS_TO_RETURN]
                    
                    cluster_shap_data["features"].append({
                        "feature_index": dim_idx,
                        "feature_name": feat_name,
                        "importance_score": feat_item['importance_score'],
                        "top_shap_tokens": [{"token": t, "impact": float(v)} for t, v in top_tokens]
                    })
                    
                except Exception as e:
                    print(f"    ! Error calculating SHAP for feature {dim_idx}: {e}")

            analysis_results["shap_analysis"][cluster_id] = cluster_shap_data

        # Save Result
        output_json = os.path.join(DIVI_RESULTS_DIR, f"shap_analysis_seed{seed}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            
        # Generate Readable Report
        output_txt = os.path.join(DIVI_RESULTS_DIR, f"shap_analysis_report_seed{seed}.txt")
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(f"SHAP Analysis Report for Seed {seed}\n")
            f.write("="*50 + "\n\n")
            
            for cluster_id, data in analysis_results["shap_analysis"].items():
                f.write(f"[{cluster_id}]\n")
                f.write(f"  Sampled Texts: {data['sampled_texts_count']}\n")
                
                for feat in data['features']:
                    f.write(f"  > Feature: {feat['feature_name']} (Idx: {feat['feature_index']})\n")
                    f.write(f"    Importance Score: {feat['importance_score']:.4f}\n")
                    f.write(f"    Key Influencing Tokens:\n")
                    for t in feat['top_shap_tokens']:
                        f.write(f"      - '{t['token']}' : {t['impact']:.4f}\n")
                    f.write("\n")
                f.write("-" * 30 + "\n")

        print(f"✓ Saved SHAP analysis to: {output_json}")
        print(f"✓ Saved Readable Report to: {output_txt}")

if __name__ == "__main__":
    analyze_shap_for_clusters()
