#!/usr/bin/env python3
"""
K-Means based spatial clustering evaluation - simplified with scanpy plotting
"""

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from module import SpatialMAELightning
from config import Config
from mae_tokenizer import get_default_mae_tokenizer



def main():
    # Setup
    checkpoint_path = './checkpoints/last.ckpt'
    data_dir = '/leonardo_work/EUHPC_B25_011/ST/DLPFC'
    
    config = Config.from_yaml('configs/antioverfit.yaml')
    tokenizer = get_default_mae_tokenizer()
    config.model.vocab_size = len(tokenizer)
    config.model.padding_idx = tokenizer.pad_token_id
    
    model = SpatialMAELightning.load_from_checkpoint(
        checkpoint_path, config=config, strict=False, map_location='cpu'
    )
    model.eval()
    
    # Load slide
    h5ad_file = sorted(Path(data_dir).glob('*.h5ad'))[0]
    adata = sc.read_h5ad(h5ad_file)
    
    # Store original spatial coordinates for plotting
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    else:
        spatial_coords = adata.obs[['array_row', 'array_col']].values
    adata.obsm['spatial'] = spatial_coords.astype(np.float32)
    
    # Normalize coordinates for model
    coords = spatial_coords.astype(np.float32)
    coords[:, 0] = coords[:, 0] - coords[:, 0].min()
    coords[:, 1] = coords[:, 1] - coords[:, 1].min()
    coords = coords / coords.max() * 100
    
    # Process expression - compute HVG before normalization
    sc.pp.highly_variable_genes(adata, n_top_genes=300, subset=False, flavor='seurat_v3')
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    
    # Filter genes by vocabulary
    hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
    vocab_genes = [g for g in tokenizer.get_stoi().keys() if not g.startswith("<")]
    filtered_genes = [g for g in hvg_genes if g in vocab_genes][:config.model.n_genes]
    
    gene_ids = [tokenizer.gene_vocab[g] for g in filtered_genes]
    gene_exp = adata[:, filtered_genes].X.toarray().astype(np.float32)
    
    # Direct full-slide inference without partitioning
    n_spots = len(coords)
    n_genes = config.model.n_genes
    
    # Prepare tensors for the entire slide
    coords_t = torch.from_numpy(coords).float().unsqueeze(0)  # [1, n_spots, 2]
    gene_exp_t = torch.from_numpy(gene_exp).float()  # [n_spots, n_genes]
    gene_ids_t = torch.tensor(gene_ids).long()
    
    # Pad genes to match model config
    pad_len = n_genes - len(gene_ids_t)
    gene_ids_t = torch.cat([gene_ids_t, torch.zeros(pad_len, dtype=torch.long)])
    gene_exp_t = torch.cat([gene_exp_t, torch.zeros(n_spots, pad_len)], dim=1)
    
    # Create batch
    batch_gene_ids = gene_ids_t.unsqueeze(0).expand(1, n_spots, -1)  # [1, n_spots, n_genes]
    batch_gene_exp = gene_exp_t.unsqueeze(0)  # [1, n_spots, n_genes]
    
    # Get features for the entire slide
    with torch.no_grad():
        features = model.model.encode(batch_gene_ids, batch_gene_exp, coords_t)
        features = features.squeeze(0).numpy()
    
    # Store features in adata
    adata.obsm['X_mae'] = features
    
    # Clustering with scanpy
    sc.pp.neighbors(adata, use_rep='X_mae', n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0, flavor='igraph', n_iterations=2, directed=False)
    
    # Evaluation metrics
    predictions = adata.obs['leiden'].astype(int).values
    silhouette = silhouette_score(features, predictions)
    
    # Compute metrics with ground truth (DLPFC has 'ground_truth' column)
    labels = adata.obs['ground_truth'].values
    valid_mask = ~pd.isna(labels)
    labels_valid = pd.factorize(labels[valid_mask])[0]
    pred_valid = predictions[valid_mask]
    ari = adjusted_rand_score(labels_valid, pred_valid)
    nmi = normalized_mutual_info_score(labels_valid, pred_valid)
    print(f"Metrics - Silhouette: {silhouette:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}")
    
    # Plotting
    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Spatial plots
    sc.pl.embedding(adata, basis='spatial', color='leiden', 
                    ax=axes[0,0], show=False, title='MAE Clustering', frameon=False)
    sc.pl.embedding(adata, basis='spatial', color='ground_truth',
                    ax=axes[0,1], show=False, title='Ground Truth', frameon=False)
    
    # UMAP plots
    sc.pl.umap(adata, color='leiden', ax=axes[1,0], show=False, title='UMAP - Clusters')
    sc.pl.umap(adata, color='ground_truth', ax=axes[1,1], show=False, title='UMAP - Layers')
    
    plt.tight_layout()
    plt.savefig('mae_clustering.png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()