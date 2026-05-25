import json
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
import os
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib as mpl

# Set font globally for IEEE compliance (usually Times New Roman or similar standard serif)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def main():
    print('Loading data...')
    with open('C:/Users/jerry/Desktop/lab/0311/thesis-experiment/results/divi_combined/clustered_traces_seed123.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Stratified sampling
    clustered_data = {}
    for d in data:
        c = d.get('cluster')
        if c not in clustered_data:
            clustered_data[c] = []
        clustered_data[c].append(d)
        
    sampled_data = []
    # Sample up to 350 per cluster to keep t-SNE and boundaries fast but accurate
    for c, items in clustered_data.items():
        sampled_data.extend(random.sample(items, min(350, len(items))))
        
    texts = [str(d.get('prompt', '')) for d in sampled_data]
    clusters = [str(d.get('cluster')) for d in sampled_data]
    is_harmful = [True if d.get('success') is True else False for d in sampled_data]

    print('Loading model and computing embeddings...')
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    
    print('Applying PCA and t-SNE...')
    pca = PCA(n_components=min(50, len(sampled_data)))
    emb_pca = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb_pca)
    
    out_dir = 'C:/Users/jerry/Desktop/lab/0311/thesis-experiment/paper/Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs/figures'
    os.makedirs(out_dir, exist_ok=True)
    
    # Variables for boundaries
    x_min, x_max = emb_2d[:, 0].min() - 3, emb_2d[:, 0].max() + 3
    y_min, y_max = emb_2d[:, 1].min() - 3, emb_2d[:, 1].max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
    
    # Plot 1: All Clusters
    print('Generating Plot 1 (All Clusters)...')
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=clusters, palette='tab20', s=45, alpha=0.9, edgecolor='white', linewidth=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'divi_scatter_all.pdf'))
    plt.close()
    
    # Plot 2: Harmful vs Harmless with SVM Boundary
    print('Generating Plot 2 (Harmful vs Harmless)...')
    plt.figure(figsize=(10, 8))
    clf_bin = SVC(kernel='rbf', C=1.0, gamma='scale')
    y_svm = [1 if h else 0 for h in is_harmful]
    clf_bin.fit(emb_2d, y_svm)
    Z_bin = clf_bin.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    cmap_bg = ListedColormap(['#f2f2f2', '#fce8e8']) # gentle gray and gentle red
    plt.contourf(xx, yy, Z_bin, cmap=cmap_bg, alpha=0.5)
    plt.contour(xx, yy, Z_bin, colors='black', linewidths=0.8, linestyles='dashed', alpha=0.7)
    
    harmful_labels = ['Attack Success (Harmful)' if h else 'Defense Success (Harmless)' for h in is_harmful]
    palette2 = {'Attack Success (Harmful)': '#d62728', 'Defense Success (Harmless)': '#7f7f7f'}
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=harmful_labels, palette=palette2, s=45, alpha=0.8, edgecolor='w', linewidth=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'divi_scatter_harmful_gray.pdf'))
    plt.close()

    # Plot 3: Highlight Failure Clusters (17, 18, 19, 20)
    print('Generating Plot 3 (Highlight Failure Clusters)...')
    plt.figure(figsize=(10, 8))
    failure_clusters = ['17', '18', '19', '20']
    cluster_labels = ['Cluster ' + c if c in failure_clusters else 'Other (Safe/Defense)' for c in clusters]
    
    clf_multi = SVC(kernel='rbf', C=1.5, gamma='scale')
    clf_multi.fit(emb_2d, cluster_labels)
    Z_m = clf_multi.predict(np.c_[xx.ravel(), yy.ravel()])
    unique_labels = list(np.unique(cluster_labels))
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    Z_m_int = np.array([label_to_int[lbl] for lbl in Z_m]).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z_m_int, cmap='Pastel2', alpha=0.2)
    plt.contour(xx, yy, Z_m_int, colors='gray', linewidths=0.5, linestyles='dotted', alpha=0.8)
    
    # Palette with vivid colors for specific clusters and dull gray for others
    custom_palette = {
        'Cluster 17': '#ff7f0e', # Orange
        'Cluster 18': '#2ca02c', # Green
        'Cluster 19': '#d62728', # Red
        'Cluster 20': '#9467bd', # Purple
        'Other (Safe/Defense)': '#b0b0b0' # Gray
    }
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=cluster_labels, palette=custom_palette, s=45, alpha=0.9, edgecolor='w', linewidth=0.3)
    
    plt.legend(title='', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'divi_scatter_highlight_clusters.pdf'))
    plt.close()
    
    print('All plots generated successfully.')

if __name__ == '__main__':
    main()