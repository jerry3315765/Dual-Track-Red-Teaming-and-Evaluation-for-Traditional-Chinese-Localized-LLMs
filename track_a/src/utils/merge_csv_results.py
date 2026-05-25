import os
import glob
import pandas as pd
from datetime import datetime

# Determine paths
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
results_dir = os.path.join(workspace_root, "experiment_v2", "data", "results")
output_file = os.path.join(current_dir, f"merged_results.csv")

def merge_csv_files():
    print(f"Looking for CSV files in: {results_dir}")
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(results_dir, "*_summary_*.csv"))
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    print(f"Found {len(csv_files)} files:")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")

    # Read and append
    all_dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            # Optionally add a column to specifically identify the source file
            df['source_file'] = os.path.basename(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSuccessfully merged {len(all_dfs)} files.")
        print(f"Saved merged file to: {output_file}")
        print(f"Total rows: {len(merged_df)}")
    else:
        print("No valid data found to merge.")

if __name__ == "__main__":
    merge_csv_files()
