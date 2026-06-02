
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from datetime import datetime

# Configuration
INPUT_CSV_PATH = r"C:\Users\jerry\Desktop\lab\code\merged_results_embedded.csv"
OUTPUT_DIR = r"C:\Users\jerry\Desktop\lab\code\divi_results"

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. Diagnosable Variational GMM
# ==============================================================================
class DiagnosableGMM(nn.Module):
    def __init__(self, input_dim, num_components, prior_phi_probs, init_means=None, temperature=1.0):
        super().__init__()
        self.D = input_dim
        self.K = num_components
        self.temperature = temperature

        self.register_buffer('prior_phi_probs', prior_phi_probs)
        self.register_buffer('prior_mu_0', torch.zeros(input_dim))
        # Tighten BG prior to force clusters to explain data
        self.register_buffer('prior_logvar_0', torch.tensor([-2.0] * input_dim))

        prior_logits = torch.log(prior_phi_probs / (1 - prior_phi_probs))
        self.phi_logits = nn.Parameter(prior_logits.clone())

        if init_means is not None:
            if init_means.shape[0] < num_components:
                pad = torch.randn(num_components - init_means.shape[0], input_dim)
                init_means = torch.cat([init_means, pad], dim=0)
            self.q_mu = nn.Parameter(init_means)
        else:
            self.q_mu = nn.Parameter(torch.randn(num_components, input_dim))

        self.q_logvar = nn.Parameter(torch.ones(num_components, input_dim) * -1.0)
        self.pi_logits = nn.Parameter(torch.ones(num_components))

    def gumbel_sigmoid_sample(self, logits):
        uniform = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(uniform + 1e-9) + 1e-9)
        return torch.sigmoid((logits + gumbel) / self.temperature)

    def forward(self, X):
        N, D = X.shape
        phi_sample = self.gumbel_sigmoid_sample(self.phi_logits).unsqueeze(0)

        x_exp = X.unsqueeze(1)
        mu_exp = self.q_mu.unsqueeze(0)
        # Prevent variance collapse
        logvar_clamped = torch.clamp(self.q_logvar, min=-5.0, max=5.0)
        logvar_exp = logvar_clamped.unsqueeze(0)

        log_prob_cluster = -0.5 * (np.log(2 * np.pi) + logvar_exp + (x_exp - mu_exp)**2 / torch.exp(logvar_exp))
        log_prob_bg = -0.5 * (np.log(2 * np.pi) + self.prior_logvar_0 + (x_exp - self.prior_mu_0)**2 / torch.exp(self.prior_logvar_0))

        weighted_log_prob = phi_sample.unsqueeze(1) * log_prob_cluster + \
                            (1 - phi_sample.unsqueeze(1)) * log_prob_bg

        log_p_x_given_z = weighted_log_prob.sum(dim=2)

        pi = torch.softmax(self.pi_logits, dim=0)
        log_joint = log_p_x_given_z + torch.log(pi + 1e-9).unsqueeze(0)
        log_likelihood = torch.logsumexp(log_joint, dim=1).sum()

        q_phi = torch.sigmoid(self.phi_logits)
        q_phi = torch.clamp(q_phi, 1e-6, 1-1e-6)
        p_phi = torch.clamp(self.prior_phi_probs, 1e-6, 1-1e-6)

        kl_phi = (q_phi * (torch.log(q_phi) - torch.log(p_phi)) + \
                 (1 - q_phi) * (torch.log(1 - q_phi) - torch.log(1 - p_phi))).sum() * N

        loss = -log_likelihood + kl_phi
        return loss, q_phi, log_p_x_given_z

    def get_cluster_diagnostics(self, X):
        with torch.no_grad():
            _, _, log_p_x_given_z = self.forward(X)
            z_hard = torch.argmax(log_p_x_given_z, dim=1)
            cluster_scores = []

            for k in range(self.K):
                mask = (z_hard == k)
                if mask.sum() == 0:
                    cluster_scores.append(-np.inf) # Ignore empty clusters
                else:
                    log_probs = log_p_x_given_z[mask, k]
                    score = -log_probs.mean().item()
                    cluster_scores.append(score)
            return np.array(cluster_scores)

# ==============================================================================
# 2. DIVI Clustering Wrapper
# ==============================================================================
class DIVIClustering:
    def __init__(self, split_threshold=22.0, split_interval=60, max_epochs=300, lr=0.01, verbose=True, cluster_split_penalty=None):
        self.split_threshold = split_threshold
        self.split_interval = split_interval
        self.max_epochs = max_epochs
        self.lr = lr
        self.verbose = verbose
        # If penalty is provided, we can use it to adjust the splitting criterion (though simple implementation is threshold based)
        self.cluster_split_penalty = cluster_split_penalty 

        self.model = None
        self.history = {'loss': [], 'k': [], 'phi': []}

    def _step_a_calculate_prior(self, X):
        if self.verbose: print("Running Step A: Calculating Heuristic Priors...")
        N, D = X.shape
        kmeans = KMeans(n_clusters=min(3, N), random_state=42).fit(X)
        labels = kmeans.labels_

        scores = []
        for j in range(D):
            feat = X[:, j]
            groups = [feat[labels == k] for k in range(3) if len(feat[labels == k]) > 0]
            if len(groups) > 1:
                try:
                    stat, _ = stats.kruskal(*groups)
                    score = np.log1p(stat)
                except: score = 0
            else: score = 0
            scores.append(score)

        scores = np.array(scores)
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        logits = (norm - 0.5) * 6
        rho = torch.sigmoid(torch.tensor(logits, dtype=torch.float32))
        rho = torch.clamp(rho, 0.01, 0.99)
        return rho

    def _expand_model(self, old_model, split_idx):
        D, K = old_model.D, old_model.K
        new_model = DiagnosableGMM(D, K + 1, old_model.prior_phi_probs, temperature=old_model.temperature)

        with torch.no_grad():
            new_model.phi_logits.copy_(old_model.phi_logits)

            old_mu = old_model.q_mu.data
            target_mu = old_mu[split_idx]
            mu_a = target_mu + torch.randn(D) * 0.2
            mu_b = target_mu - torch.randn(D) * 0.2

            keep_idx = [i for i in range(K) if i != split_idx]
            if keep_idx:
                new_mus = torch.cat([old_mu[keep_idx], mu_a.unsqueeze(0), mu_b.unsqueeze(0)], dim=0)
            else:
                new_mus = torch.cat([mu_a.unsqueeze(0), mu_b.unsqueeze(0)], dim=0)
            new_model.q_mu.copy_(new_mus)

            old_logvar = old_model.q_logvar.data
            target_logvar = old_logvar[split_idx]
            if keep_idx:
                new_logvars = torch.cat([old_logvar[keep_idx], target_logvar.unsqueeze(0), target_logvar.unsqueeze(0)], dim=0)
            else:
                new_logvars = torch.cat([target_logvar.unsqueeze(0), target_logvar.unsqueeze(0)], dim=0)
            new_model.q_logvar.copy_(new_logvars)

        return new_model

    def fit(self, X_np):
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        N, D = X_tensor.shape

        if self.split_threshold is None:
            self.split_threshold = 0.5 * D * (1 + np.log(2 * np.pi) + np.log(1.0))
            if self.verbose:
                print(f"Auto-configured Split Threshold: {self.split_threshold:.2f}")

        rho = self._step_a_calculate_prior(X_np)
        global_mean = torch.mean(X_tensor, dim=0, keepdim=True)
        self.model = DiagnosableGMM(D, 1, rho, init_means=global_mean)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.verbose: print(f"Starting Training (Initial K=1)...")

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            loss, q_phi, _ = self.model(X_tensor)
            loss.backward()
            optimizer.step()

            self.model.temperature = max(0.1, self.model.temperature * 0.98)

            self.history['loss'].append(loss.item())
            self.history['k'].append(self.model.K)
            self.history['phi'].append(q_phi.detach().numpy())

            if epoch % self.split_interval == 0:
                scores = self.model.get_cluster_diagnostics(X_tensor)
                worst_idx = np.argmax(scores)
                worst_score = scores[worst_idx]
                
                # Dynamic Threshold Check using Penalty
                current_threshold = self.split_threshold
                if self.cluster_split_penalty is not None:
                     current_threshold *= self.cluster_split_penalty 

                if self.verbose:
                    print(f"Epoch {epoch}: K={self.model.K}, Max NLL={worst_score:.2f} (Threshold: {current_threshold:.2f})")

                if worst_score > current_threshold:
                    if self.verbose: print(f"   >>> Splitting Cluster {worst_idx}...")
                    self.model = self._expand_model(self.model, worst_idx)
                    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.verbose: print("Training Completed.")
        return self

# ==============================================================================
# 3. Experiment Runner
# ==============================================================================
def run_divi_experiment(X_input, document_ids, seed_value, D):
    print("\n" + "="*70)
    print(f"Running DIVI with Seed = {seed_value}")
    print("="*70)
    
    set_seed(seed_value)
    
    divi_exp = DIVIClustering(
        split_threshold=None,
        split_interval=60,
        max_epochs=300,
        lr=0.01,
        verbose=True
    )
    
    divi_exp.split_threshold = 0.5 * D * (1 + np.log(2 * np.pi) + np.log(0.9))
    print(f"Split Threshold: {divi_exp.split_threshold:.2f}")
    
    divi_exp.fit(X_input)
    
    _, _, log_p = divi_exp.model(torch.tensor(X_input, dtype=torch.float32))
    y_pred = torch.argmax(log_p, dim=1).detach().numpy()
    cluster_probs = torch.softmax(log_p, dim=1).detach().numpy()
    
    print(f"\n✓ Training completed: K={divi_exp.model.K}")
    
    final_phi = divi_exp.history['phi'][-1]
    
    output_data = {
        "metadata": {
            "seed": seed_value,
            "total_samples": len(y_pred),
            "total_features": D,
            "num_clusters": int(divi_exp.model.K)
        },
        "clusters": {}
    }
    
    for cluster_id in range(divi_exp.model.K):
        mask = (y_pred == cluster_id)
        cluster_samples = X_input[mask]
        
        if len(cluster_samples) == 0:
            continue
        
        feature_means = cluster_samples.mean(axis=0)
        feature_stds = cluster_samples.std(axis=0)
        phi_weights = final_phi
        importance_scores = phi_weights * np.abs(feature_means)
        top_indices = np.argsort(importance_scores)[::-1][:20]
        
        cluster_info = {
            "size": int(mask.sum()),
            "percentage": float(mask.sum() / len(y_pred) * 100),
            "avg_membership_prob": float(cluster_probs[mask, cluster_id].mean()),
            "top_features": []
        }
        
        for rank, idx in enumerate(top_indices, 1):
            cluster_info["top_features"].append({
                "rank": rank,
                "feature_id": int(idx + 1),
                "feature_name": f"Feature_{idx + 1}",
                "importance_score": float(importance_scores[idx]),
                "phi_weight": float(phi_weights[idx]),
                "cluster_mean": float(feature_means[idx]),
                "cluster_std": float(feature_stds[idx])
            })
        
        output_data["clusters"][f"cluster_{cluster_id}"] = cluster_info
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    json_filename = f"gibbs_cluster_features_seed{seed_value}.json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON saved: {json_path}")
    
    csv_rows = []
    for cluster_id in range(divi_exp.model.K):
        mask = (y_pred == cluster_id)
        if mask.sum() == 0:
            continue
        
        cluster_samples = X_input[mask]
        feature_means = cluster_samples.mean(axis=0)
        importance_scores = phi_weights * np.abs(feature_means)
        top_indices = np.argsort(importance_scores)[::-1][:20]
        
        for rank, idx in enumerate(top_indices, 1):
            csv_rows.append({
                'Seed': seed_value,
                'Cluster': cluster_id,
                'Cluster_Size': int(mask.sum()),
                'Rank': rank,
                'Feature_ID': idx + 1,
                'Feature_Name': f'Feature_{idx + 1}',
                'Importance_Score': importance_scores[idx],
                'Phi_Weight': phi_weights[idx],
                'Cluster_Mean': feature_means[idx],
                'Cluster_Std': cluster_samples.std(axis=0)[idx]
            })
    
    csv_df = pd.DataFrame(csv_rows)
    csv_filename = f"gibbs_cluster_features_seed{seed_value}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ CSV saved: {csv_path}")
    
    result_df = pd.DataFrame({
        'ID': document_ids,
        'Cluster': y_pred
    })
    result_filename = f"gibbs_cluster_assignments_seed{seed_value}.csv"
    result_path = os.path.join(OUTPUT_DIR, result_filename)
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ Cluster assignments saved: {result_path}")
    print("="*70)

def main():
    print("Loading Embedding Data...")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        print("Please run generate_embeddings.py first.")
        return

    # Read with utf-8-sig to handle Chinese characters and potential BOM
    df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8-sig')
    
    # Handle missing ID column if it doesn't exist
    if 'ID' not in df.columns:
        print("Warning: 'ID' column not found. Generating IDs based on index.")
        df['ID'] = df.index

    document_ids = df['ID'].values
    feature_columns = [col for col in df.columns if col.startswith('Feature_')]
    
    if len(feature_columns) == 0:
        print("Error: No feature columns (starting with 'Feature_') found in the input CSV.")
        return

    X_emb = df[feature_columns].values
    
    print("Preprocessing embeddings...")
    X_emb = np.nan_to_num(X_emb, nan=0.0)
    X_norm = normalize(X_emb, norm='l2')
    scaler = StandardScaler()
    X_input = scaler.fit_transform(X_norm)
    
    D = X_input.shape[1]
    
    # Run experiments for multiple seeds
    seeds = [42, 123, 999, 2025, 1]
    for seed in seeds:
        run_divi_experiment(X_input, document_ids, seed, D)

if __name__ == "__main__":
    main()
