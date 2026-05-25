import json
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random

def main():
    print('Loading data...')
    with open('C:/Users/jerry/Desktop/lab/0311/thesis-experiment/results/divi_combined/clustered_traces_seed123.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Total traces: {len(data)}')
    
    # Stratified sampling to ensure we capture all clusters evenly
    clustered_data = {}
    for d in data:
        c = d.get('cluster')
        if c not in clustered_data:
            clustered_data[c] = []
        clustered_data[c].append(d)
        
    sampled_data = []
    # Sample up to 500 per cluster for speed
    for c, items in clustered_data.items():
        sampled_data.extend(random.sample(items, min(500, len(items))))
        
    print(f'Sampled traces: {len(sampled_data)}')
    
    texts = [str(d.get('prompt', '')) + " " + str(d.get('response', '')) for d in sampled_data]
    clusters = [str(d.get('cluster')) for d in sampled_data]
    sources = [str(d.get('source', 'Unknown')) for d in sampled_data]

    print('Loading model...')
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    print('Encoding texts...')
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    
    print('Applying PCA then TSNE...')
    pca = PCA(n_components=min(50, len(sampled_data)))
    emb_pca = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb_pca)
    
    # Plotting
    print('Plotting...')
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=emb_2d[:, 0], y=emb_2d[:, 1],
        hue=clusters,
        style=sources,
        palette='viridis',
        s=60, alpha=0.8
    )
    plt.title('DIVI Latent Failure Clusters (t-SNE Projection)')
    plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    import os
    os.makedirs('C:/Users/jerry/Desktop/lab/0311/thesis-experiment/paper/Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs/figures', exist_ok=True)
    out_path = 'C:/Users/jerry/Desktop/lab/0311/thesis-experiment/paper/Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs/figures/divi_scatter.pdf'
    plt.savefig(out_path)
    print(f'Saved plot to {out_path}')

if __name__ == '__main__':
    main()
