import os
import glob
import json
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
results_dir = os.path.join(workspace_root, "experiment_v2", "data", "results")
divi_results_dir = os.path.join(workspace_root, "code", "divi_results")
merged_csv_path = os.path.join(workspace_root, "code", "merged_results_embedded.csv")
output_csv_path = os.path.join(workspace_root, "experiment_v2", "data", "summary_analysis_full_v2.csv")

def analyze_results():
    print("Starting Analysis (SHAP Integrated)...")

    # 1. Load Reference Data for ID Mapping
    print(f"Loading reference data from {merged_csv_path}...")
    try:
        use_cols = ['scenario_desc', 'system_prompt_name', 'turn', 'prompt']
        df_ref = pd.read_csv(merged_csv_path, usecols=use_cols, encoding='utf-8-sig')
        df_ref['ID'] = df_ref.index
    except Exception as e:
        print(f"Error loading merged CSV: {e}")
        return

    # Create a lookup key function
    def make_key(sys_name, scen_desc, turn_num, pmt_text):
        return (
            str(sys_name).strip(),
            str(scen_desc).strip(),
            int(turn_num),
            str(pmt_text).strip()
        )

    # Dictionary: Key -> List of IDs (handling duplicates)
    print("Building lookup index...")
    lookup_map = defaultdict(list)
    for idx, row in df_ref.iterrows():
        k = make_key(row['system_prompt_name'], row['scenario_desc'], row['turn'], row['prompt'])
        lookup_map[k].append(idx)

    print(f"Lookup index built. Unique keys: {len(lookup_map)}")

    # 2. Load Evaluation Results & Map to IDs
    json_pattern = os.path.join(results_dir, "redteam_divi_results_*.json")
    files = glob.glob(json_pattern)
    
    if not files:
        print(f"No result files found in {results_dir}")
        return

    # ID -> Success (bool)
    id_success_map = {}
    
    print(f"Processing {len(files)} result files...")
    matched_count = 0
    
    # We need to process carefully to pop IDs from lookup_map to handle duplicates correctly
    # Checking if we need to reset lookup map? No, we consume it one-pass.
    # IMPORTANT: The matching order assumes the order in JSONs corresponds somewhat to CSV? 
    # Not necessarily. But if duplicates are identical, any assignment is valid as long as we use all IDs.
    
    for filepath in files:
        filename = os.path.basename(filepath)
        method_name = filename.replace("redteam_divi_results_", "").replace(".json", "")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                scenario = item.get("scenario_desc", "Unknown Scenario")
                sys_prompt = item.get("system_prompt_name", method_name) 
                turn = item.get("turn", 1)
                prompt = item.get("prompt", "")
                is_success = item.get("success", False)
                
                key = make_key(sys_prompt, scenario, turn, prompt)
                
                if key in lookup_map and lookup_map[key]:
                    # Take one ID
                    data_id = lookup_map[key].pop(0)
                    id_success_map[data_id] = is_success
                    matched_count += 1
                else:
                    # Fallback or error?
                    # If empty list, it means we matched more than existed?
                    pass

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Mapped {matched_count} evaluation results to IDs.")

    # 3. Load Cluster Assignments & SHAP Features for Each Seed
    # We will build a Summary List
    summary_rows = []

    # Find all seeds
    cluster_files = glob.glob(os.path.join(divi_results_dir, "gibbs_cluster_assignments_seed*.csv"))
    seeds = []
    for cf in cluster_files:
        bn = os.path.basename(cf)
        s = bn.replace("gibbs_cluster_assignments_seed", "").replace(".csv", "")
        seeds.append(s)
    
    seeds = sorted(seeds, key=lambda x: int(x) if x.isdigit() else x)

    for seed in seeds:
        print(f"Processing Seed {seed}...")
        
        # A. Load Assignments
        assign_file = os.path.join(divi_results_dir, f"gibbs_cluster_assignments_seed{seed}.csv")
        try:
            df_assign = pd.read_csv(assign_file)
            # Filter IDs that have success results
            # We must include only valid IDs
            df_assign['Success'] = df_assign['ID'].map(id_success_map)
            
            # Drop rows where ID was not found in results?
            # User said "All clusters have data".
            # We keep rows where Success is NaN only if we want to debug, but for stats we need Success.
            valid_df = df_assign.dropna(subset=['Success'])
            
            # Calculate Stats per Cluster
            stats = valid_df.groupby('Cluster')['Success'].agg(['count', 'sum', 'mean']).reset_index()
            stats.columns = ['Cluster', 'Sample_Count', 'Success_Count', 'Success_Rate']
            stats['Success_Rate'] = (stats['Success_Rate'] * 100).round(2)
            
        except Exception as e:
            print(f"Error loading assignments for seed {seed}: {e}")
            continue

        # B. Load SHAP Features
        shap_file = os.path.join(divi_results_dir, f"shap_analysis_seed{seed}.json")
        shap_data = {}
        try:
            if os.path.exists(shap_file):
                with open(shap_file, 'r', encoding='utf-8') as f:
                    shap_json = json.load(f)
                    shap_data = shap_json.get("shap_analysis", {})
        except Exception as e:
            print(f"Error loading SHAP for seed {seed}: {e}")

        # C. Merge and Store
        for _, row in stats.iterrows():
            cid = row['Cluster']
            # SHAP key is usually "cluster_X"
            shap_key = f"cluster_{int(cid)}"
            
            features_info = []
            
            if shap_key in shap_data:
                feats = shap_data[shap_key].get("features", [])
                # Get Top 3 features
                for i, ft in enumerate(feats[:3]):
                    fname = ft.get("feature_name", f"F{i}")
                    tokens = [t['token'] for t in ft.get("top_shap_tokens", [])]
                    token_str = ", ".join(tokens)
                    features_info.append(f"{fname}: [{token_str}]")
            
            # Join features into a string or separate columns?
            # User asked to "list key Features, key terms"
            # We'll make a single string for readability in CSV
            features_summary = " | ".join(features_info)

            summary_rows.append({
                "Seed": seed,
                "Cluster": cid,
                "Sample_Count": row['Sample_Count'],
                "Success_Count": row['Success_Count'],
                "Success_Rate(%)": row['Success_Rate'],
                "Key_SHAP_Features": features_summary
            })

    # 4. Save Summary
    if not summary_rows:
        print("No summary data generated.")
        return

    df_summary = pd.DataFrame(summary_rows)
    # Sort by Seed, then Success Rate
    df_summary = df_summary.sort_values(by=['Seed', 'Success_Rate(%)'], ascending=[True, False])
    
    df_summary.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print("="*50)
    print(f"Analysis Complete.")
    print(f"Summary saved to: {output_csv_path}")
    print("Top rows:")
    print(df_summary.head().to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    analyze_results()
