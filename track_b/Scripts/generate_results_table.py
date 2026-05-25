import os
import pandas as pd
import ast

def list_string_to_sum(val):
    try:
        if isinstance(val, (int, float)): return val
        if isinstance(val, list): return sum(val)
        if isinstance(val, str):
            val = val.strip()
            if val.startswith('[') and val.endswith(']'):
                li = ast.literal_eval(val)
                return sum(li)
        return 0
    except:
        return 0

def list_string_to_len(val):
    try:
        if isinstance(val, (int, float)): return 1
        if isinstance(val, list): return len(val)
        if isinstance(val, str):
            val = val.strip()
            if val.startswith('[') and val.endswith(']'):
                li = ast.literal_eval(val)
                return len(li)
        return 0
    except:
        return 0

# Set the path to where the results are - using the path verified in previous step
RESULTS_ROOT = r"c:\Users\jerry\Desktop\lab\0311\PromptFuzz-Thesis\Results\focus\redteam\baseline"

model_stats = []

if os.path.exists(RESULTS_ROOT):
    for model_name in os.listdir(RESULTS_ROOT):
        model_path = os.path.join(RESULTS_ROOT, model_name)
        csv_path = os.path.join(model_path, "all_results.csv")
        
        if os.path.isdir(model_path) and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                # Check if 'results' column exists
                if 'results' not in df.columns:
                    # try look for similar columns (case insensitive)
                    cols = [c for c in df.columns if c.lower() == 'results']
                    if cols:
                        df.rename(columns={cols[0]: 'results'}, inplace=True)
                    else:
                        print(f"Skipping {model_name}: No 'results' column found.")
                        continue

                # Calculate metrics
                df['success_count'] = df['results'].apply(list_string_to_sum)
                df['total_attempts'] = df['results'].apply(list_string_to_len)
                
                total_queries = len(df)
                total_attempts = df['total_attempts'].sum()
                total_successes = df['success_count'].sum()
                
                prompts_with_break = len(df[df['success_count'] > 0])
                
                esr = (prompts_with_break / total_queries * 100) if total_queries > 0 else 0
                asr = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
                
                # Clean model name for LaTeX
                display_name = model_name.replace('_', '-').replace('-text', '').replace('-chat', '')
                
                model_stats.append({
                    "Model": display_name,
                    "Total Prompts": total_queries,
                    "True ASR": asr,
                    "Prompt ASR (ESR)": esr
                })
            except Exception as e:
                print(f"Error processing {model_name}: {e}")

    # Create DataFrame
    stats_df = pd.DataFrame(model_stats)
    
    # Sort by ASR
    if not stats_df.empty:
        stats_df = stats_df.sort_values('True ASR', ascending=False)
    
    # Generate LaTeX Table
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Experimental Results: Jailbreak Success Rates on Localized vs. General LLMs}")
    print("\\label{tab:main_results}")
    print("\\begin{tabular}{l c c c}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Prompts Generated} & \\textbf{Prompt Coverage (ESR)} & \\textbf{True ASR} \\\\")
    print("\\midrule")
    
    for _, row in stats_df.iterrows():
        print(f"{row['Model']} & {row['Total Prompts']} & {row['Prompt ASR (ESR)']:.2f}\\% & {row['True ASR']:.2f}\\% \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
else:
    print(f"Directory not found: {RESULTS_ROOT}")
