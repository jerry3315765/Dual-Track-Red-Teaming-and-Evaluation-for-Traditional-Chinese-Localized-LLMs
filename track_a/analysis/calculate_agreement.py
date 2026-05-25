
import pandas as pd
import json
import glob
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Paths
HUMAN_AUDIT_FILE = "code/human_audit_sample.csv"
RESULTS_PATTERN = "experiment_v2/data/results/redteam_divi_results_*.json"

def calculate_agreement():
    print("Loading Human Audit Data...")
    try:
        df_human = pd.read_csv(HUMAN_AUDIT_FILE)
    except Exception as e:
        print(f"Error reading human audit file: {e}")
        return

    # Check for 'score' column
    if 'score' not in df_human.columns:
        print("Error: 'score' column not found in human audit file.")
        return

    # We treat 'score' in this file as the HUMAN score
    # We need to find the GPT score from the original results
    
    print("Loading Original Results for Ground Truth (GPT Scores)...")
    results_files = glob.glob(RESULTS_PATTERN)
    gpt_scores = {} # (model, prompt) -> score

    for fpath in results_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    model = item.get('model')
                    prompt = item.get('prompt', '').strip()
                    
                    # Extract score
                    eval_data = item.get('evaluation')
                    score = 0
                    if isinstance(eval_data, dict):
                        score = eval_data.get('score', 0)
                    
                    if model and prompt:
                        gpt_scores[(model, prompt)] = score
        except Exception as e:
            pass

    print(f"Loaded {len(gpt_scores)} reference scores.")

    # Match Human Scores to GPT Scores
    y_human = []
    y_gpt = []
    
    matched_count = 0
    df_human['gpt_score'] = np.nan

    for idx, row in df_human.iterrows():
        model = row.get('model_name') # CSV uses model_name
        prompt = str(row.get('prompt', '')).strip()
        human_score = row.get('score')
        
        # Try to find GPT score
        gpt_val = gpt_scores.get((model, prompt))
        
        if gpt_val is not None:
            y_human.append(int(human_score))
            y_gpt.append(int(gpt_val))
            df_human.at[idx, 'gpt_score'] = gpt_val
            matched_count += 1
        else:
            # Try fuzzy match or debug
            pass

    print(f"\nMatched {matched_count} out of {len(df_human)} samples.")
    
    if matched_count < 10:
        print("Warning: Too few matches found. Aborting metric calculation.")
        return

    # Calculate Metrics
    y_human = np.array(y_human)
    y_gpt = np.array(y_gpt)
    
    # Exact Agreement
    agreement = np.mean(y_human == y_gpt)
    
    # Cohen's Kappa
    # Since scores are ordinal/categorical (1-5), weighted kappa is better, but simple linear kappa is often used.
    # We will compute simple Cohen's Kappa first.
    kappa = cohen_kappa_score(y_human, y_gpt)
    
    # Linear Weighted Kappa (Quadratic is also common)
    kappa_weighted = cohen_kappa_score(y_human, y_gpt, weights='linear')
    
    print("\n--- Agreement Results ---")
    print(f"Exact Agreement: {agreement:.2%}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Weighted Kappa (Linear): {kappa_weighted:.4f}")
    
    # Disagreement Analysis
    diffs = df_human[df_human['score'] != df_human['gpt_score']]
    print(f"\nDisagreements: {len(diffs)}")
    if len(diffs) > 0:
        print(diffs[['model_name', 'score', 'gpt_score']].head().to_string())

    # Save format for Paper
    with open("code/agreement_stats.txt", "w") as f:
        f.write(f"Agreement: {agreement:.1%}\n")
        f.write(f"Kappa: {kappa:.2f}\n")

if __name__ == "__main__":
    calculate_agreement()
