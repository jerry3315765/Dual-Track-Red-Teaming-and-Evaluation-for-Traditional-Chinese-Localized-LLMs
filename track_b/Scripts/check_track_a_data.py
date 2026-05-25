import pandas as pd
import os

TRACK_A_CSV = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\data\processed\merged_results_embedded.csv"

if os.path.exists(TRACK_A_CSV):
    try:
        df = pd.read_csv(TRACK_A_CSV)
        print("Columns:", df.columns.tolist())
        if 'model' in df.columns:
            print("Track A Models:", df['model'].unique())
        if 'success' in df.columns:
            print("Success values:", df['success'].unique())
        if 'turn' in df.columns:
            print("Turn unique:", df['turn'].unique())
    except Exception as e:
        print(f"Error reading CSV: {e}")
else:
    print(f"File not found: {TRACK_A_CSV}")
