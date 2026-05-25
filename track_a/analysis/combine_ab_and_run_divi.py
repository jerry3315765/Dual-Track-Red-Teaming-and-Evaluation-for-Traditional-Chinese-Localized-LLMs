import os
import json
import pandas as pd
import numpy as np
import sys
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK_A_DIR = os.path.join(BASE_DIR, "results", "raw_traces")
# Note: Adjust relative path to where Results is actually located based on workspace info
# Workspace: c:\Users\jerry\Desktop\lab\0311
# script is in c:\Users\jerry\Desktop\lab\0311\thesis-experiment\analysis
# Results is in c:\Users\jerry\Desktop\lab\0311\Results
# TRACK_B_DIR = os.path.join(BASE_DIR, "..", "Results", "focus", "redteam")
TRACK_B_DIR = os.path.join(BASE_DIR, "..", "PromptFuzz-Thesis", "Results", "focus", "redteam")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "divi_combined")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_analysis():
    # 1. Load Track A
    print("Loading Track A data...")
    track_a_files = [f for f in os.listdir(TRACK_A_DIR) if f.endswith(".json") and "promptfuzz" not in f]
    track_a_data = []
    for f in track_a_files:
        try:
            with open(os.path.join(TRACK_A_DIR, f), "r", encoding="utf-8") as file:
                data = json.load(file)
                # Add trace source
                for d in data:
                    d["source"] = "Track A (Static)"
                track_a_data.extend(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"Loaded {len(track_a_data)} traces from Track A.")

    # 2. Load Track B (CSV) - Recursive Search
    print("Loading Track B data...")
    if os.path.exists(TRACK_B_DIR):
        track_b_files = []
        for root, dirs, files in os.walk(TRACK_B_DIR):
            for file in files:
                if file.endswith(".csv"):
                    track_b_files.append(os.path.join(root, file))
    else:
        print(f"Warning: Track B directory not found at {TRACK_B_DIR}")
        track_b_files = []
        
    track_b_data = []

    for csv_path in track_b_files:
        try:
            # Infer model name from directory structure if possible
            # e.g. baseline/qwen3-8b/all_results.csv -> qwen3-8b
            # e.g. 0.csv -> dynamic_model (fallback)
            rel_path = os.path.relpath(csv_path, TRACK_B_DIR)
            path_parts = rel_path.split(os.sep)
            
            inferred_model = "dynamic_model"
            if len(path_parts) > 1:
                # Assuming structure: model_name/filename.csv
                # possibly baseline/model_name/filename.csv
                if path_parts[0] == "baseline" and len(path_parts) > 2:
                    inferred_model = path_parts[1]
                elif path_parts[0] != "baseline":
                     inferred_model = path_parts[0]
            
            df = pd.read_csv(csv_path)
            
            # Identify model from filename or content if possible, else generic
            # Filenames might be 0.csv, 1.csv... mapping is in a separate file usually 
            # For now, we treat all as "Dynamic Fuzzing" source
            
            for idx, row in df.iterrows():
                prompt = row.get('prompt', str(row.get('attack', '')))
                try:
                    responses = eval(row.get('response', '[]'))
                    results = eval(row.get('results', '[]'))
                except:
                    responses = [row.get('response', '')] if isinstance(row.get('response'), str) else []
                    results = [0]
                
                # Ensure lists
                if not isinstance(responses, list): responses = [str(responses)]
                if not isinstance(results, list): results = [results]

                for i, (resp, res) in enumerate(zip(responses, results)):
                    is_success = bool(res)
                    trace = {
                        "scenario_id": f"B_{os.path.basename(csv_path)}_{idx}",
                        "scenario_desc": "PromptFuzz Dynamic",
                        "attack_type": "promptfuzz",
                        "system_prompt_name": "dynamic_mutation", 
                        "turn": 1,
                        "prompt": prompt,
                        "response": resp,
                        "evaluation": {"score": 5 if is_success else 1, "success": is_success},
                        "model": inferred_model, 
                        "success": is_success,
                        "source": "Track B (Dynamic)"
                    }
                    track_b_data.append(trace)
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")

    print(f"Loaded {len(track_b_data)} traces from Track B.")

    # 3. Merge
    combined_data = track_a_data + track_b_data
    combined_json_path = os.path.join(OUTPUT_DIR, "combined_traces.json")
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"Saved combined data to {combined_json_path}")
    
    if not combined_data:
        print("No data to process.")
        return

    # 4. Embeddings & DIVI
    sys.path.append(os.path.join(BASE_DIR, "src"))
    try:
        from sentence_transformers import SentenceTransformer
        # Attempt to import DIVI_V2
        from DIVI.DIVI_V2 import DIVIClustering, set_seed
        
        print("Imported DIVI. Generating embeddings...")
        
        # We embed the RESPONSES (or Prompt+Response?)
        # Paper says: "embed the multi-turn conversation traces"
        # Usually that means the text content. We will concat prompt + response for richness
        texts = [f"User: {t['prompt']}\nModel: {t['response']}" for t in combined_data]
        
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        embeddings = model.encode(texts, show_progress_bar=True)

        # --------------------------------------------------------------------------
        # Multi-Seed Execution Loop
        # --------------------------------------------------------------------------
        SEEDS = [42, 123, 2025]  # Try multiple seeds to verify stability
        print(f"Starting Multi-Seed Analysis. Seeds: {SEEDS}")

        import shap
        # Masker for SHAP
        masker = shap.maskers.Text(model.tokenizer)
        # Define stopwords
        ignore_tokens = {
            '，', '。', '！', '？', '：', '；', '“', '”', '、', '（', '）', ' ', 
            'User', 'Model', 'User:', 'Model:', '\n', '的', '了', '是', '在', '我',
            ',', '.', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']'
        }

        for seed in SEEDS:
            print(f"\n{'='*40}")
            print(f"Running Experiment with Seed {seed}")
            print(f"{'='*40}")
            
            # Set global seed
            set_seed(seed)
            if hasattr(torch, 'use_deterministic_algorithms'):
                 try: torch.use_deterministic_algorithms(True)
                 except: pass

            print(f"Running DIVI on {len(embeddings)} samples...")
            
            # DIVI Hyperparams (Aggressive splitting adjustment for meaningful separation)
            # split_threshold: Lower = easier to split. Lowering significantly to find distinct behaviors.
            # split_interval: More splits over time.
            # cluster_split_prior_weight: High penalty (default) can prevent splitting. We need to encourage it.
            
            # Using parameters that were successful in finding clusters in similar high-dim NLP tasks
            divi = DIVIClustering(
                split_interval=10,      # Check for splits very frequently (every 10 epochs)
                split_threshold=-1e9,   # Force splits until max K (since NLL can be very negative)
                cluster_split_penalty=1.0, 
                max_epochs=200,         # Ensure convergence with new complexity
                verbose=True
            )
            
            # Run fit
            divi.fit(embeddings)
            
            # Inference
            X_tensor = torch.tensor(embeddings, dtype=torch.float32)
            divi.model.eval()
            with torch.no_grad():
                _, _, log_p_x_given_z = divi.model(X_tensor)
                
                if hasattr(divi.model, 'pi_logits'):
                    pi = torch.softmax(divi.model.pi_logits, dim=0)
                else:
                    pi = torch.ones(divi.model.K) / divi.model.K
                    
                log_joint = log_p_x_given_z + torch.log(pi + 1e-9).unsqueeze(0)
                labels = torch.argmax(log_joint, dim=1).detach().cpu().numpy()
            
            K_found = divi.model.K
            print(f"Seed {seed}: Inferred labels for {len(labels)} samples. Clusters found: {K_found}")

            # Update trace data with this seed's cluster ID
            # We create a copy or just confirm we save separate files
            current_run_data = [d.copy() for d in combined_data]
            for i, label in enumerate(labels):
                current_run_data[i]['cluster'] = int(label)
                
            # Save Clustered Traces for this Seed
            clustered_path = os.path.join(OUTPUT_DIR, f"clustered_traces_seed{seed}.json")
            with open(clustered_path, "w", encoding="utf-8") as f:
                json.dump(current_run_data, f, ensure_ascii=False, indent=2)
            print(f"Saved clustered traces to {clustered_path}")

            # ----------------------------------------------------------------------
            # SHAP Analysis Integration (Per Seed)
            # ----------------------------------------------------------------------
            if K_found < 2:
                print(f"Warning: Only 1 cluster found for Seed {seed}. SHAP analysis might be generic.")
            
            print(f"Starting SHAP Analysis for Seed {seed}...")
            
            shap_results = {}
            cluster_ids = sorted(list(set(labels)))
            
            # Feature weights
            phi_weights = None
            if hasattr(divi.model, 'phi_logits'):
                phi_weights = torch.sigmoid(divi.model.phi_logits).detach().cpu().numpy()
            
            for k in cluster_ids:
                print(f"  Analyzing Cluster {k}...")
                mask = (labels == k)
                cluster_embeddings = embeddings[mask]
                
                if len(cluster_embeddings) == 0: continue
                
                # Identify importance
                mean_vals = np.abs(cluster_embeddings.mean(axis=0))
                
                if phi_weights is not None:
                    if phi_weights.ndim == 2 and k < phi_weights.shape[0]:
                        w = phi_weights[k]
                    else:
                        w = phi_weights.flatten() if phi_weights.ndim == 1 else np.ones_like(mean_vals)
                    importance = w * mean_vals
                else:
                    importance = mean_vals
                    
                top_dims = np.argsort(importance)[::-1][:3] # Top 3 features
                
                # Sampling
                sample_indices = np.where(mask)[0]
                if len(sample_indices) > 10: # Reduce to 10 for speed
                    sample_indices = np.random.choice(sample_indices, 10, replace=False)
                    
                sampled_texts = [texts[i] for i in sample_indices]
                cluster_shap = {"features": []}

                for dim_idx in top_dims:
                    # SHAP Target
                    def predict_dim(txts):
                        return model.encode(txts)[:, dim_idx]

                    try:
                        explainer = shap.Explainer(predict_dim, masker)
                        # Reduced max_evals to speed up
                        shap_values = explainer(sampled_texts, max_evals=500, silent=True)
                        
                        token_impacts = {}
                        for i in range(len(sampled_texts)):
                            toks = shap_values[i].data
                            vals = shap_values[i].values
                            for t, v in zip(toks, vals):
                                t_clean = t.strip()
                                # Filter garbage
                                if not t_clean: continue
                                if t_clean in ignore_tokens: continue
                                token_impacts[t_clean] = token_impacts.get(t_clean, 0) + v 

                        sorted_tokens = sorted(token_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                        
                        cluster_shap["features"].append({
                            "dim": int(dim_idx),
                            "importance": float(importance[dim_idx]),
                            "top_tokens": sorted_tokens
                        })
                        
                    except BaseException as ex: # Catch All including KeyboardInterrupt if needed, but mainly for SHAP internal errors
                        print(f"    SHAP Error on dim {dim_idx}: {ex}")
                        if isinstance(ex, KeyboardInterrupt): raise ex # Re-raise if it's actual interrupt
                
                shap_results[str(k)] = cluster_shap

            # Save SHAP
            shap_path = os.path.join(OUTPUT_DIR, f"divi_shap_analysis_seed{seed}.json")
            with open(shap_path, "w", encoding="utf-8") as f:
                json.dump(shap_results, f, ensure_ascii=False, indent=2)
            print(f"SHAP Analysis Seed {seed} Complete.\n")

    except ImportError as e:
        print(f"Import Error: {e}. Please ensure sentence-transformers and src/DIVI are distinct.")
    except Exception as e:
        print(f"Execution Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_analysis()
