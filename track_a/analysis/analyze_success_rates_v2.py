import os
import glob
import json
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
# Results with evaluation (Success/Fail) from JSON files
results_dir = os.path.join(workspace_root, "experiment_v2", "data", "results")
# Directory with cluster assignments
divi_results_dir = os.path.join(workspace_root, "code", "divi_results")
# Main CSV to map ID (Index)
merged_csv_path = os.path.join(workspace_root, "code", "merged_results_embedded.csv")

def analyze_results():
    print("Starting Analysis (V2)...")

    # 1. Load Reference Data for ID Mapping
    # We need to map (system_prompt_name, scenario_desc, turn, prompt) -> ID (Index in merged_results_embedded.csv)
    print(f"Loading reference data from {merged_csv_path}...")
    try:
        # Only load necessary columns to save memory
        use_cols = ['scenario_desc', 'system_prompt_name', 'turn', 'prompt']
        df_ref = pd.read_csv(merged_csv_path, usecols=use_cols, encoding='utf-8-sig')
        # The index in this CSV is the ID used in clustering
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

    # Build Dictionary: Key -> ID
    print("Building lookup index...")
    lookup_map = {}
    for idx, row in df_ref.iterrows():
        k = make_key(row['system_prompt_name'], row['scenario_desc'], row['turn'], row['prompt'])
        lookup_map[k] = idx

    print(f"Lookup index built with {len(lookup_map)} entries.")

    # 2. Load Cluster Assignments for ALL Seeds
    # Map structure: seed -> { id -> cluster }
    print(f"Loading cluster assignments from {divi_results_dir}...")
    seed_clusters = {}
    # Store all unique cluster IDs per seed to detect missing ones later
    all_cluster_ids = {} 
    
    cluster_files = glob.glob(os.path.join(divi_results_dir, "gibbs_cluster_assignments_seed*.csv"))
    if not cluster_files:
        print(f"No cluster assignment files found in {divi_results_dir}")
        return

    for c_file in cluster_files:
        try:
            # Extract seed from filename "gibbs_cluster_assignments_seed123.csv"
            basename = os.path.basename(c_file)
            seed_part = basename.replace("gibbs_cluster_assignments_seed", "").replace(".csv", "")
            seed = seed_part
            
            # Read CSV
            df_c = pd.read_csv(c_file)
            if 'ID' in df_c.columns and 'Cluster' in df_c.columns:
                # Store as dict for fast lookup
                seed_clusters[seed] = dict(zip(df_c['ID'], df_c['Cluster']))
                # Record all unique clusters found in this seed file
                unique_clusters = sorted(df_c['Cluster'].unique())
                all_cluster_ids[seed] = unique_clusters
            else:
                print(f"Skipping {basename}: Missing ID or Cluster columns")
        except Exception as e:
            print(f"Error loading {c_file}: {e}")

    sorted_seeds = sorted(seed_clusters.keys(), key=lambda x: int(x) if x.isdigit() else x)
    print(f"Loaded cluster data for seeds: {', '.join(sorted_seeds)}")

    # 3. Process Evaluation Results (JSONs)
    json_pattern = os.path.join(results_dir, "redteam_divi_results_*.json")
    files = glob.glob(json_pattern)
    
    if not files:
        print(f"No result files found in {results_dir}")
        return

    all_records = []
    print(f"Processing {len(files)} result files...")

    matched_count = 0
    unmatched_count = 0

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
                
                # Check Success
                is_success = item.get("success", False)
                model = item.get("model", "Unknown Model")

                # Try to find ID to link with Clusters
                key = make_key(sys_prompt, scenario, turn, prompt)
                data_id = lookup_map.get(key)
                
                record = {
                    "Method": method_name,
                    "Scenario": scenario,
                    "Model": model,
                    "Success": 1 if is_success else 0
                }

                if data_id is not None:
                    matched_count += 1
                    # Append Cluster info for EACH seed using the found ID
                    for seed, cluster_map in seed_clusters.items():
                        cluster_id = cluster_map.get(data_id, "Unassigned")
                        record[f"Cluster_Seed_{seed}"] = cluster_id
                else:
                    unmatched_count += 1
                    for seed in seed_clusters.keys():
                        record[f"Cluster_Seed_{seed}"] = "Unknown"

                all_records.append(record)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Data linking complete. Matched: {matched_count}, Unmatched: {unmatched_count}")

    if not all_records:
        print("No records found.")
        return

    df = pd.DataFrame(all_records)

    # ---------------------------------------------------------
    # Analysis 1: Attack Method
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Attack Method Success Rates")
    print("="*50)
    method_stats = df.groupby("Method")["Success"].agg(['count', 'sum', 'mean']).reset_index()
    method_stats.columns = ['Method', 'Total', 'Success', 'Rate(%)']
    method_stats['Rate(%)'] = (method_stats['Rate(%)'] * 100).round(2)
    print(method_stats.sort_values(by="Rate(%)", ascending=False).to_string(index=False))

    # ---------------------------------------------------------
    # Analysis 2: Scenario
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Scenario Success Rates (Lower is safer)")
    print("="*50)
    scenario_stats = df.groupby("Scenario")["Success"].agg(['count', 'sum', 'mean']).reset_index()
    scenario_stats.columns = ['Scenario', 'Total', 'Success', 'Rate(%)']
    scenario_stats['Rate(%)'] = (scenario_stats['Rate(%)'] * 100).round(2)
    
    # Truncate long scenario names for display
    scenario_stats['Scenario'] = scenario_stats['Scenario'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
    print(scenario_stats.sort_values(by="Rate(%)", ascending=False).to_string(index=False))

    # ---------------------------------------------------------
    # Analysis 3: Model
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Model Safety (Success Rate, Lower is better)")
    print("="*50)
    model_stats = df.groupby("Model")["Success"].agg(['count', 'sum', 'mean']).reset_index()
    model_stats.columns = ['Model', 'Total', 'Success', 'Rate(%)']
    model_stats['Rate(%)'] = (model_stats['Rate(%)'] * 100).round(2)
    print(model_stats.sort_values(by="Rate(%)", ascending=True).to_string(index=False))

    # ---------------------------------------------------------
    # Analysis 4: Clusters per Seed
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("SHAP Cluster Analysis (Per Seed)")
    print("="*50)

    for seed in sorted_seeds:
        col_name = f"Cluster_Seed_{seed}"
        print(f"\n--- Seed {seed} ---")
        
        # Original clusters in this seed file
        known_clusters = all_cluster_ids.get(seed, [])
        print(f"Known Cluster IDs in raw file: {known_clusters}")

        # Filter out 'Unknown' or 'Unassigned'
        valid_df = df[~df[col_name].isin(["Unknown", "Unassigned"])]
        
        if valid_df.empty:
            print("No valid cluster data matched.")
            continue
            
        # Calculate stats
        c_stats = valid_df.groupby(col_name)["Success"].agg(['count', 'sum', 'mean']).reset_index()
        c_stats.columns = ['Cluster', 'Total', 'Success', 'Rate(%)']
        c_stats['Rate(%)'] = (c_stats['Rate(%)'] * 100).round(2)

        # Sort by Cluster ID naturally to check for missing ones
        try:
            c_stats['Cluster'] = c_stats['Cluster'].astype(int)
        except ValueError:
            pass # Keep as string if conversion fails
            
        c_stats_id_sorted = c_stats.sort_values(by="Cluster", ascending=True)
        
        print("\n[Sorted by Cluster ID]")
        print(c_stats_id_sorted.to_string(index=False))

        # Check for missing clusters in the matched results
        matched_cluster_ids = c_stats['Cluster'].unique()
        missing = set(known_clusters) - set(matched_cluster_ids)
        if missing:
             print(f"⚠️ Warning: Clusters {sorted(list(missing))} exist in assignment file but have 0 samples in evaluation results.")

        # Also show the sorted by Success Rate version for analysis
        print("\n[Sorted by Success Rate (High -> Low)]")
        print(c_stats.sort_values(by="Rate(%)", ascending=False).to_string(index=False))

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    total = len(df)
    success_count = df["Success"].sum()
    rate = (success_count / total) * 100
    print("\n" + "="*50)
    print("Overall Summary")
    print(f"Total Attempts: {total}")
    print(f"Total Success: {success_count}")
    print(f"Overall Rate: {rate:.2f}%")
    print("="*50)

    # Save to CSV
    output_csv = os.path.join(workspace_root, "experiment_v2", "data", "summary_analysis_full_v2.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nDetailed data saved to: {output_csv}")

if __name__ == "__main__":
    analyze_results()
