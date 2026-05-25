import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import entropy

# Setup
RESULTS_DIR = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\results\divi_combined"
FIGURES_DIR = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\paper\Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs\figures"
SEEDS = [42, 123, 2025]

# Ensure figures dir exists
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

summary_stats = []
all_distribution_data = []

for seed in SEEDS:
    file_path = os.path.join(RESULTS_DIR, f"clustered_traces_seed{seed}.json")
    if not os.path.exists(file_path):
        print(f"Skipping {seed}, file not found.")
        continue
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Calculate Cluster Counts
    cluster_counts = df['cluster'].value_counts().sort_index()
    n_clusters = len(cluster_counts)
    total_samples = len(df)
    
    # Calculate Entropy (measure of distribution spread)
    # Higher entropy = more even distribution across clusters
    # Lower entropy = effectively fewer clusters (dominance)
    prob_dist = cluster_counts / total_samples
    ent = entropy(prob_dist)
    
    # Largest Cluster Dominance
    max_share = prob_dist.max()
    
    summary_stats.append({
        "Seed": seed,
        "Clusters Found (K)": n_clusters,
        "Entropy": f"{ent:.3f}",
        "Dominant Cluster Share": f"{max_share:.1%}",
        "Total Samples": total_samples
    })
    
    # Prepare data for plotting
    for k, count in cluster_counts.items():
        all_distribution_data.append({
            "Seed": str(seed),
            "Cluster ID": k,
            "Count": count,
            "Percentage": count / total_samples
        })

# 1. Generate Summary Table (LaTeX)
summary_df = pd.DataFrame(summary_stats)
print("\n=== LaTeX Table Generation ===")
print(summary_df.to_latex(index=False, caption="Stability Analysis of DIVI Clustering across Varying Initialization Seeds", label="tab:seed_stability"))

# 2. Generate Distribution Plot
plt.figure(figsize=(12, 6))
plot_df = pd.DataFrame(all_distribution_data)

# Sort by cluster ID
plot_df = plot_df.sort_values(by="Cluster ID")

sns.barplot(data=plot_df, x="Cluster ID", y="Count", hue="Seed", palette="viridis")
plt.title("Cluster Size Distribution across Random Seeds")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Traces")
plt.legend(title="Initialization Seed")
plt.tight_layout()

output_img = os.path.join(FIGURES_DIR, "divi_seed_distribution.pdf")
plt.savefig(output_img)
print(f"Saved distribution plot to {output_img}")

# 3. Generate Heatmap of Overlap (Conceptual) -> Requires aligned IDs, skipping for now
# Instead, plotting the Dominance
plt.figure(figsize=(6, 4))
sns.lineplot(data=plot_df, x="Cluster ID", y="Percentage", hue="Seed", marker="o")
plt.title("Cluster Density Profile")
plt.ylabel("Percentage of Dataset")
plt.savefig(os.path.join(FIGURES_DIR, "divi_cluster_profile.pdf"))
print(f"Saved profile plot.")
