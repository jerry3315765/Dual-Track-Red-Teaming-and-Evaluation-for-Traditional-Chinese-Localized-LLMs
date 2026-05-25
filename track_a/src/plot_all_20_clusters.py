import json
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import matplotlib as mpl

# IEEE Font Settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def main():
    print('Loading all data...')
    with open('C:/Users/jerry/Desktop/lab/0311/thesis-experiment/results/divi_combined/clustered_traces_seed123.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'Total traces to plot: {len(data)}')
    
    # Use ALL data points instead of sampling
    texts = [str(d.get('prompt', '')) for d in data]
    raw_clusters = [str(d.get('cluster')) for d in data]
    
    print('Loading model and computing embeddings (this will take about 1-2 mins for all 13k records)...')
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
    
    print('Applying PCA and t-SNE on FULL dataset...')
    pca = PCA(n_components=50)
    emb_pca = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb_pca)
    
    out_dir = 'C:/Users/jerry/Desktop/lab/0311/thesis-experiment/paper/Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs/figures'
    os.makedirs(out_dir, exist_ok=True)
    
    # -----------------------------
    # Plot Configuration
    # -----------------------------
    print('Generating the requested plot...')
    plt.figure(figsize=(10, 8))
    
    failure_clusters = ['17', '18', '19', '20']
    
    custom_palette = {
        'Cluster 17': '#ff7f0e', # Orange
        'Cluster 18': '#2ca02c', # Green
        'Cluster 19': '#d62728', # Red
        'Cluster 20': '#9467bd'  # Purple
    }
    
    # 1. Plot the "Other / Safe" clusters FIRST so they stay in the background
    gray_idx = [i for i, c in enumerate(raw_clusters) if str(c) not in failure_clusters]
    plt.scatter(
        emb_2d[gray_idx, 0], emb_2d[gray_idx, 1], 
        c='#e0e0e0', # light gray
        s=15, 
        alpha=0.4, 
        label='Other (Safe Clusters 1-16)', 
        edgecolors='none'
    )
    
    # 2. Plot the Failure clusters on TOP
    for cl_num in failure_clusters:
        idx = [i for i, c in enumerate(raw_clusters) if str(c) == cl_num]
        plt.scatter(
            emb_2d[idx, 0], emb_2d[idx, 1], 
            c=custom_palette[f'Cluster {cl_num}'], 
            s=25, 
            alpha=0.9, 
            label=f'Cluster {cl_num}', 
            edgecolors='w', 
            linewidths=0.3
        )
        
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, markerscale=2)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'divi_scatter_full_20_gray.pdf')
    plt.savefig(out_path)
    plt.close()
    
    print(f'Full data plot generated successfully and saved to: {out_path}')

if __name__ == '__main__':
    main()