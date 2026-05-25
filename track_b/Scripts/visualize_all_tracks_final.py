import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# Paths
# Prefer 0115 because embedded lacks valid columns or is tricky
TRACK_A_PATH = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\data\processed\merged_results_0115.csv"
TRACK_B_ROOT = r"c:\Users\jerry\Desktop\lab\0311\PromptFuzz-Thesis\Results\focus\redteam\baseline"
OUTPUT_DIR = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\paper\Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs\figures"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def parse_results_column(val):
    try:
        if isinstance(val, (int, float)):
            return [val]
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            # Handle list strings like "[1, 1, 5]"
            if val.startswith('[') and val.endswith(']'):
                try:
                    return ast.literal_eval(val)
                except:
                    pass
            # Handle bare numbers
            try:
                return [float(val)]
            except:
                pass
        return []
    except:
        return []

def get_track_b_data():
    data = []
    if not os.path.exists(TRACK_B_ROOT):
        return pd.DataFrame()
    
    for model_name in os.listdir(TRACK_B_ROOT):
        model_path = os.path.join(TRACK_B_ROOT, model_name)
        csv_path = os.path.join(model_path, "all_results.csv")
        
        if os.path.isdir(model_path) and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                if 'results' not in df.columns:
                    cols = [c for c in df.columns if c.lower() == 'results']
                    if cols:
                        df.rename(columns={cols[0]: 'results'}, inplace=True)
                    else:
                        continue
                
                # Treat results as Binary (1=Success).
                all_results = []
                for res in df['results']:
                    parsed = parse_results_column(res)
                    all_results.extend(parsed)
                
                # Filter out empty or non-numeric
                all_results = [r for r in all_results if isinstance(r, (int, float))]
                
                total_attempts = len(all_results)
                # Treat positive values as success based on our manual inspection of 100% ASR case
                success_count = sum(1 for r in all_results if r > 0) 
                
                asr = (success_count / total_attempts * 100) if total_attempts > 0 else 0
                
                display_name = model_name # keep raw for now
                
                data.append({
                    "Model": model_name,
                    "ASR": asr,
                    "Track": "Track B (Dynamic Fuzzing)"
                })
            except Exception as e:
                print(f"Error Track B {model_name}: {e}")
                
    return pd.DataFrame(data)

def get_track_a_data():
    if not os.path.exists(TRACK_A_PATH):
        print("Track A file not found")
        return pd.DataFrame()
    
    try:
        # Use merged_results_0115.csv logic
        # Read limited columns
        cols = ['model', 'success', 'scenario_id', 'system_prompt_name']
        try:
            df = pd.read_csv(TRACK_A_PATH, usecols=lambda c: c in cols)
        except:
            df = pd.read_csv(TRACK_A_PATH) # Fallback
            
        if 'model' not in df.columns or 'success' not in df.columns:
            return pd.DataFrame()
        
        # Ensure success is bool
        df['success'] = df['success'].astype(bool)
        
        # Group logic
        grp_keys = ['model']
        if 'scenario_id' in df.columns: grp_keys.append('scenario_id')
        if 'system_prompt_name' in df.columns: grp_keys.append('system_prompt_name')
        
        grp = df.groupby(grp_keys)['success'].max().reset_index()
        model_asrs = grp.groupby('model')['success'].mean() * 100
        
        data = []
        for model, asr in model_asrs.items():
            data.append({
                "Model": model,
                "ASR": asr,
                "Track": "Track A (Static Narrative)"
            })
            
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"Error Track A: {e}")
        return pd.DataFrame()

def clean_model_name(name):
    n = str(name).lower()
    n = n.replace('@q8_0', '').replace('@q8-0', '')
    n = n.replace('oreal-', '') # remove prefix
    n = n.replace('_', '-')
    n = n.replace('-chat', '').replace('-instruct', '').replace('-text', '')
    return n

def beautify_name(name):
    # Use clean name as base
    n = clean_model_name(name)
    
    # Specific beautification
    if 'deepseek-r1-distill-llama-8b' in n: return 'DS-Llama-8B'
    if 'deepseek-r1-distill-qwen-7b' in n: return 'DS-Qwen-7B'
    if 'deepseek-r1-0528-qwen3-8b' in n: return 'DS-R1-Qwen'
    if 'gemma-3-taide-12b' in n: return 'TAIDE-12B'
    if 'llama3-taide-lx-8b' in n: return 'TAIDE-LX-8B'
    if 'llama-breeze2-8b' in n: return 'Breeze2-8B'
    if 'llama-3.2-3b' in n: return 'Llama-3.2'
    if 'gpt-oss-20b' in n: return 'GPT-OSS'
    if 'qwen3-8b' in n: return 'Qwen3-8B'
    if 'gemma-3-4b' in n: return 'Gemma-4B'
    if 'mistral-7b' in n: return 'Mistral-7B'
    
    return n.title() # Fallback

def main():
    print("Loading data...")
    df_a = get_track_a_data()
    print(f"Track A loaded: {len(df_a)} models")
    
    df_b = get_track_b_data()
    print(f"Track B loaded: {len(df_b)} models")
    
    # Normalize names for merging
    df_a['Model_Clean'] = df_a['Model'].apply(clean_model_name)
    df_b['Model_Clean'] = df_b['Model'].apply(clean_model_name)
    
    # Find intersection
    models_a = set(df_a['Model_Clean'])
    models_b = set(df_b['Model_Clean'])
    common_models = models_a.intersection(models_b)
    
    print("Common Models:", common_models)
    
    # Filter
    df_a_filtered = df_a[df_a['Model_Clean'].isin(common_models)].copy()
    df_b_filtered = df_b[df_b['Model_Clean'].isin(common_models)].copy()
    
    # Beautify
    df_a_filtered['Display_Model'] = df_a_filtered['Model'].apply(beautify_name)
    df_b_filtered['Display_Model'] = df_b_filtered['Model'].apply(beautify_name)
    
    # Combine
    df_combined = pd.concat([df_a_filtered, df_b_filtered])
    
    # Sort by Track B ASR
    if not df_b_filtered.empty:
        sort_order = df_b_filtered.sort_values('ASR', ascending=False)['Display_Model'].tolist()
    else:
        sort_order = None
    
    # --- Plot 1: Comparison Bar Chart ---
    if not df_combined.empty:
        plt.figure(figsize=(10, 6))
        
        colors = {"Track A (Static Narrative)": "#3498db", "Track B (Dynamic Fuzzing)": "#e74c3c"}
        
        sns.barplot(
            data=df_combined,
            x="ASR",
            y="Display_Model",
            hue="Track",
            palette=colors,
            order=sort_order
        )
        
        plt.title("Jailbreak Success Rate Comparison", fontsize=16, fontweight='bold')
        plt.xlabel("Attack Success Rate (%)", fontsize=12)
        plt.ylabel("", fontsize=12)
        plt.xlim(0, 105)
        plt.legend(title=None, loc='lower right')
        
        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "comparison_asr.pdf"), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, "comparison_asr.png"), dpi=300)
        plt.close()
        
        # --- Plot 2: Track A Only ---
        plt.figure(figsize=(8, 5))
        df_a_sorted = df_a_filtered.sort_values('ASR', ascending=False)
        sns.barplot(
            data=df_a_sorted,
            x="ASR",
            y="Display_Model",
            color="#3498db"
        )
        plt.title("Track A: Static Narrative Attack ASR", fontsize=14, fontweight='bold')
        plt.xlabel("ASR (%)")
        plt.ylabel("")
        plt.xlim(0, max(df_a_sorted['ASR'].max() * 1.2, 10))
        
        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "track_a_asr.pdf"), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, "track_a_asr.png"), dpi=300)
        plt.close()

        # --- Plot 3: Track B Only ---
        plt.figure(figsize=(8, 5))
        df_b_sorted = df_b_filtered.sort_values('ASR', ascending=False)
        sns.barplot(
            data=df_b_sorted,
            x="ASR",
            y="Display_Model",
            color="#e74c3c"
        )
        plt.title("Track B: Dynamic Fuzzing Attack ASR", fontsize=14, fontweight='bold')
        plt.xlabel("ASR (%)")
        plt.ylabel("")
        plt.xlim(0, 105)
        
        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "track_b_asr.pdf"), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, "track_b_asr.png"), dpi=300)
        plt.close()
        
        print("Figures saved.")
        print("\nTrack A Stats:")
        print(df_a_filtered[['Display_Model', 'ASR']])
        print("\nTrack B Stats:")
        print(df_b_filtered[['Display_Model', 'ASR']])

if __name__ == "__main__":
    main()
