import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft JhengHei'] # Fallback for CJK if needed
plt.rcParams['axes.unicode_minus'] = False

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

# Paths
RESULTS_ROOT = r"c:\Users\jerry\Desktop\lab\0311\PromptFuzz-Thesis\Results\focus\redteam\baseline"
OUTPUT_DIR = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\paper\Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs\figures"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

model_stats = []

if os.path.exists(RESULTS_ROOT):
    for model_name in os.listdir(RESULTS_ROOT):
        model_path = os.path.join(RESULTS_ROOT, model_name)
        csv_path = os.path.join(model_path, "all_results.csv")
        
        if os.path.isdir(model_path) and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                # Check column
                if 'results' not in df.columns:
                    cols = [c for c in df.columns if c.lower() == 'results']
                    if cols:
                        df.rename(columns={cols[0]: 'results'}, inplace=True)
                    else:
                        continue

                # Calculate
                df['success_count'] = df['results'].apply(list_string_to_sum)
                df['total_attempts'] = df['results'].apply(list_string_to_len)
                
                total_queries = len(df)
                total_attempts = df['total_attempts'].sum()
                total_successes = df['success_count'].sum()
                
                prompts_with_break = len(df[df['success_count'] > 0])
                
                esr = (prompts_with_break / total_queries * 100) if total_queries > 0 else 0
                asr = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
                
                # Clean name using the new beautify function
                display_name = beautify_name(model_name)
                
                # Categorize
                category = "General"
                if "taide" in display_name.lower() or "breeze" in display_name.lower():
                    category = "Localized (TW)"
                elif "distill" in display_name.lower():
                    category = "Distilled/Reasoning"
                
                model_stats.append({
                    "Model": display_name,
                    "Total Prompts": total_queries,
                    "ESR": esr,
                    "ASR": asr,
                    "Category": category
                })
            except Exception as e:
                print(f"Error {model_name}: {e}")

    df_stats = pd.DataFrame(model_stats)
    df_stats = df_stats.sort_values("ASR", ascending=False)
    
    # Plot 1: Bar Chart of ASR
    plt.figure(figsize=(10, 6))
    
    # Create color palette based on category
    palette = {"Localized (TW)": "#d62728", "General": "#2ca02c", "Distilled/Reasoning": "#ff7f0e"}
    
    ax = sns.barplot(data=df_stats, x="ASR", y="Model", hue="Category", palette=palette, dodge=False)
    
    plt.title("Attack Success Rate (ASR) by Model", fontsize=14)
    plt.xlabel("True ASR (%)")
    plt.ylabel("")
    plt.xlim(0, 105)
    
    # Add value labels
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f%%', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "asr_bar_chart.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "asr_bar_chart.png")) # Save PNG for preview if needed
    plt.close()

    # Plot 2: Scatter Plot ASR vs ESR
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_stats, x="ASR", y="ESR", hue="Category", style="Category", s=150, palette=palette)
    
    # Add labels with better placement
    texts = []
    # Sort for deterministic processing
    df_labels = df_stats.sort_values(by=['ASR'], ascending=True)
    
    for i, row in df_labels.iterrows():
        # Simple alternating offset to reduce overlap
        # Using enumerate index based logic not df index
        pos_index = df_labels.index.get_loc(i)
        
        # Base offset
        y_offset = -1.5 if pos_index % 2 == 0 else 1.5
        x_offset = 0
        
        # Check if high congestion area (ESR close to 100)
        if row['ESR'] > 95:
            # Alternate more aggressively
            y_offset = -2.5 if pos_index % 3 == 0 else (2.5 if pos_index % 3 == 1 else 5)
            # Stagger X slightly for label
            x_offset = 1 if pos_index % 2 == 0 else -1

        plt.text(
            row['ASR'] + x_offset, 
            row['ESR'] + y_offset, 
            row['Model'], 
            fontsize=9, 
            alpha=0.9,
            color='black',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
        )

    plt.title("Attack Success Rate (ASR) vs. Prompt Coverage (ESR)", fontsize=14)
    plt.xlabel("True ASR (%) - Probability of Single Attack Success")
    plt.ylabel("Prompt ESR (%) - Percent of Prompts Finding Breach")
    plt.xlim(-5, 110)
    plt.ylim(-5, 110)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "esr_vs_asr_scatter.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "esr_vs_asr_scatter.png"))
    plt.close()
    
    print("Figures generated successfully in", OUTPUT_DIR)

else:
    print("Directory not found")
