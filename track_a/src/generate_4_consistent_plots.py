import json
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import matplotlib as mpl

# IEEE Font Settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def main():
    data_path = 'C:/Users/jerry/Desktop/lab/0311/thesis-experiment/results/divi_combined/clustered_traces_seed123.json'
    print('Loading data...')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Total traces: {len(data)}')
    
    texts = [str(d.get('prompt', '')) for d in data]
    clusters = np.array([str(d.get('cluster')) for d in data])
    success = np.array([True if d.get('success') is True else False for d in data])
    
    cache_path = 'C:/Users/jerry/Desktop/lab/0311/thesis-experiment/results/divi_combined/cached_embeddings_13k.npy'
    if os.path.exists(cache_path):
        print('Loading cached embeddings. (Found in file system but we will just recalculate if fail)')
        try:
            embeddings = np.load(cache_path)
        except:
             model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
             embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
             np.save(cache_path, embeddings)
    else:
        print('Computing embeddings...')
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
        np.save(cache_path, embeddings)
        
    print('Applying PCA and TSNE...')
    pca = PCA(n_components=50)
    emb_pca = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb_pca)
    
    out_dir = 'C:/Users/jerry/Desktop/lab/0311/thesis-experiment/paper/Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs/figures'
    os.makedirs(out_dir, exist_ok=True)
    
    # -----------------------------
    # Plotting styles
    # -----------------------------
    # Removing white edgecolors from tiny dots makes them deeply colored instead of looking faded
    fig_kwargs = {'figsize': (9, 7)}
    scatter_bg = {'c': '#888888', 's': 8, 'alpha': 0.4, 'edgecolors': 'none', 'zorder': 1} 
    scatter_fg = {'s': 15, 'alpha': 0.95, 'edgecolors': 'none', 'zorder': 5}
    
    def plot_base():
        fig, ax = plt.subplots(**fig_kwargs)
        # remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return fig, ax

    # Plot 1: Attack Success vs Defense
    print('Generating Plot 1...')
    fig, ax = plot_base()
    ax.scatter(emb_2d[~success, 0], emb_2d[~success, 1], label='Defense Success (Safe)', **scatter_bg)
    ax.scatter(emb_2d[success, 0], emb_2d[success, 1], c='#D50000', label='Attack Success (Jailbreak)', **scatter_fg) # Bright Red
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_1_jailbreaks.pdf'))
    plt.close()

    # Plot 2: Danger Zones
    print('Generating Plot 2...')
    fig, ax = plot_base()
    failure_mask = np.isin(clusters, ['17', '18', '19', '20'])
    ax.scatter(emb_2d[~failure_mask, 0], emb_2d[~failure_mask, 1], label='Other Clusters', **scatter_bg)
    ax.scatter(emb_2d[failure_mask, 0], emb_2d[failure_mask, 1], c='#4B0082', label='Failure Clusters (17-20)', **scatter_fg) # Deep Indigo
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_2_danger_zones.pdf'))
    plt.close()

    # Plot 3: 4 Vulnerable Clusters Breakdown
    print('Generating Plot 3...')
    fig, ax = plot_base()
    ax.scatter(emb_2d[~failure_mask, 0], emb_2d[~failure_mask, 1], label='Other Clusters', **scatter_bg)
    palette = {'17': '#E66100', '18': '#007959', '19': '#9D2235', '20': '#004D80'}
    for c_id, color in palette.items():
        mask = clusters == c_id
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, label=f'Cluster {c_id}', **scatter_fg)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_3_breakdown.pdf'))
    plt.close()

    # Plot 4: Full 20 Clusters Landscape
    print('Generating Plot 4...')
    fig, ax = plot_base()
    cmap = plt.get_cmap('tab20')
    unique_clusters = sorted([int(c) for c in np.unique(clusters)])
    for i, c_id in enumerate(unique_clusters):
        mask = clusters == str(c_id)
        # All points have full color, no gray background
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[cmap(i)], label=f'C{c_id}', s=10, alpha=0.9, edgecolors='none')
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, markerscale=3, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_4_all_clusters.pdf'))
    plt.close()

    print('All plots generated successfully.')

if __name__ == '__main__':
    main()