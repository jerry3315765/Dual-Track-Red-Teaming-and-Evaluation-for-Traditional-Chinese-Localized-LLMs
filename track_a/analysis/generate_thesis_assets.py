import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jieba
from collections import Counter
from matplotlib.font_manager import FontProperties

# Setup
RESULTS_DIR = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\results\divi_combined"
FIGURES_DIR = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\paper\Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs\figures"
SEED_FILE = os.path.join(RESULTS_DIR, "clustered_traces_seed123.json")

# Ensure output dir
os.makedirs(FIGURES_DIR, exist_ok=True)

# Plotting Style
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# Expanded Stopwords
STOPWORDS = set([
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有', '看', '好', '自己', '這',
    'User', 'Model', 'User:', 'Model:', '\n', 'system', 'prompt', 'response', 'assistant',
    ',', '.', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', '，', '。', '！', '？', '：', '；', '“', '”', '、', '（', '）', '「', '」', '『', '』',
    '可能', '使用', '可以', '需要', '如果', '我們', '因為', '所以', '對於', '關於', '為了', '進行', '提供', '以及', '或者', '相關', '問題',
    'AI', '語言', '模型', '協助', '建議', '行為', '行動', '合法', '非法', '道德', '違法', '法律', '請', '確保', '必須', '任何', '無法', '抱歉', '對不起'
])

def extract_keywords(df, cluster_id, top_n=8):
    """
    Extract distinctive keywords using Relative Frequency (Ratio vs Background).
    """
    cluster_docs = df[df['cluster'] == cluster_id]
    other_docs = df[df['cluster'] != cluster_id]
    
    if len(cluster_docs) == 0: return []
    
    # 1. Background Counts (Entire Corpus)
    all_text = " ".join([str(r['response']) for _, r in df.iterrows()])
    all_tokens = [t for t in jieba.cut(all_text) if t.strip() not in STOPWORDS and len(t.strip()) > 1]
    bg_counts = Counter(all_tokens)
    total_bg = sum(bg_counts.values())

    # 2. Cluster Counts
    cluster_text = " ".join([str(r['response']) for _, r in cluster_docs.iterrows()])
    cluster_tokens = [t for t in jieba.cut(cluster_text) if t.strip() not in STOPWORDS and len(t.strip()) > 1]
    cluster_counts = Counter(cluster_tokens)
    total_cluster = sum(cluster_counts.values())
    
    # 3. Calculate Scores (Relative Frequency * Log Freq)
    # We want words that are frequent in THIS cluster relative to background, but also not super rare.
    scores = []
    for word, count in cluster_counts.most_common(500):
        if count < 5: continue # Ignore rare noises
        
        p_c = count / total_cluster
        p_bg = bg_counts[word] / total_bg
        
        lift = p_c / (p_bg + 1e-9)
        score = lift * np.log(count) # Boost by raw frequency to avoid rare words with high lift
        
        scores.append((word, score))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

def main():
    print(f"Loading {SEED_FILE}...")
    with open(SEED_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    print(f"Total Traces: {len(df)}")
    print("Cluster Distribution:")
    print(df['cluster'].value_counts())
    
    # 1. Cluster Distribution Plot
    plt.figure(figsize=(8, 5))
    cluster_counts = df['cluster'].value_counts().sort_index()
    colors = sns.color_palette("viridis", n_colors=len(cluster_counts))
    
    ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=colors)
    plt.title("Cluster Size Distribution (Seed 123)", fontsize=14)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.ylabel("Number of Traces", fontsize=12)
    
    # Add labels
    for i, v in enumerate(cluster_counts.values):
        ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "seed123_distribution.pdf"))
    print("Saved distribution plot.")
    
    # 2. Keyword Analysis per Cluster
    cluster_summaries = []
    
    # We focus on the populated clusters: 17, 18, 19, 20
    # Note: 17 and 18 are dominant. 19/20 are outliers.
    active_clusters = cluster_counts.index.tolist()
    
    for cid in active_clusters:
        kws = extract_keywords(df, cid)
        kw_str = ", ".join([f"{w}({c})" for w, c in kws])
        
        # Also get a representative sample (closest to length median?)
        sample_doc = df[df['cluster'] == cid].sample(1).iloc[0]
        short_prompt = sample_doc['prompt'][:50] + "..."
        
        cluster_summaries.append({
            "Cluster ID": cid,
            "Size": len(df[df['cluster'] == cid]),
            "Share": f"{len(df[df['cluster'] == cid])/len(df):.1%}",
            "Top Keywords": kw_str,
            "Sample Prompt": short_prompt
        })
        
    summary_df = pd.DataFrame(cluster_summaries)
    
    # 3. Save Summary Table to CSV and LaTeX
    csv_path = os.path.join(FIGURES_DIR, "seed123_cluster_summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved summary CSV to {csv_path}")
    
    latex_str = summary_df[['Cluster ID', 'Size', 'Share', 'Top Keywords']].to_latex(index=False, caption="Semantic Analysis of Discovered Clusters (Seed 123)", label="tab:cluster_content")
    print("\nXXX LATEX TABLE XXX")
    print(latex_str)
    print("XXXXXXXXXXXXXXXXXXX\n")

if __name__ == "__main__":
    main()