import json
import os
import jieba
import pandas as pd
from collections import Counter
import numpy as np

# Setup
RESULTS_DIR = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\results\divi_combined"
# We need to analyze all 3 seeds
SEEDS = [42, 123, 2025]

STOPWORDS = set([
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有', '看', '好', '自己', '這',
    'User', 'Model', 'User:', 'Model:', '\n', 'system', 'prompt', 'response', 'assistant',
    ',', '.', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', '，', '。', '！', '？', '：', '；', '“', '”', '、', '（', '）', '「', '」', '『', '』',
    '可能', '使用', '可以', '需要', '如果', '我們', '因為', '所以', '對於', '關於', '為了', '進行', '提供', '以及', '或者', '相關', '問題',
    'AI', '語言', '模型', '協助', '建議', '行為', '行動', '合法', '非法', '道德', '違法', '法律', '請', '確保', '必須', '任何', '無法', '抱歉', '對不起',
    '但', '與', '並', '或', '將', '其', '此', '如', '等'
])

def get_keywords(texts, top_n=8):
    if not texts: return []
    all_tokens = []
    for text in texts:
        # Simple tokenization
        tokens = [t for t in jieba.cut(str(text)) if t.strip() and t.strip() not in STOPWORDS and len(t.strip()) > 1]
        all_tokens.extend(tokens)
    
    # Just simple frequency for the single-cluster case (Seed 42/2025)
    # Since there is no "background" to compare against if K=1 (the whole dataset is the cluster),
    # relative frequency is meaningless unless we compare against a generic corpus.
    # But for Seed 123 (K=4), we can compare clusters against the whole dataset.
    
    counts = Counter(all_tokens)
    return [word for word, count in counts.most_common(top_n)]

# We need a different strategy for K=1 vs K>1
# For K=1 (Seed 42/2025), "Representative" means the most common topic words.
# For K>1 (Seed 123), "Representative" means distinctive words for the dominant clusters.

def analyze_seed(seed):
    file_path = os.path.join(RESULTS_DIR, f"clustered_traces_seed{seed}.json")
    if not os.path.exists(file_path):
        return f"Seed {seed}: File not found."
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    clusters = df['cluster'].value_counts()
    
    print(f"--- Seed {seed} ---")
    print(f"Clusters Found: {len(clusters)}")
    
    # Pick top 2 clusters
    top_clusters = clusters.head(2).index.tolist()
    
    results = []
    
    for cid in top_clusters:
        cluster_docs = df[df['cluster'] == cid]
        texts = cluster_docs['response'].tolist()
        keywords = get_keywords(texts, top_n=10)
        
        share = len(cluster_docs) / len(df)
        
        results.append({
            "Cluster ID": cid,
            "Share": f"{share:.1%}",
            "Keywords": ", ".join(keywords)
        })
        print(f"Cluster {cid} ({share:.1%}): {', '.join(keywords)}")
        
    return results

if __name__ == "__main__":
    final_table_data = {}
    for seed in SEEDS:
        final_table_data[seed] = analyze_seed(seed)
        
    print("\n\nLaTeX Content:")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Representative Clusters and Key Terms across Random Seeds}")
    print(r"\label{tab:seed_representatives}")
    print(r"\begin{tabularx}{\textwidth}{c c c X}")
    print(r"\toprule")
    print(r"\textbf{Seed} & \textbf{Cluster ID} & \textbf{Share} & \textbf{Top Keywords} \\")
    print(r"\midrule")
    
    for seed in SEEDS:
        rows = final_table_data[seed]
        # Multi-row for Seed
        first = True
        for row in rows:
            seed_str = f"Seed {seed}" if first else ""
            print(f"{seed_str} & {row['Cluster ID']} & {row['Share']} & {row['Keywords']} \\\\")
            if first:
                first = False
        print(r"\midrule")
        
    print(r"\bottomrule")
    print(r"\end{tabularx}")
    print(r"\end{table*}")
