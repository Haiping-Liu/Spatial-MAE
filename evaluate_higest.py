import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from module.higest_module import HiGeSTLightning
from configs.config import Config
from data.tokenizer import get_default_mae_tokenizer


def load_or_build_global_hvgs(config: Config, tokenizer) -> list:
    save_dir = Path(config.logging.save_dir)
    hvgs_file = save_dir / 'global_hvgs.txt'

    if hvgs_file.exists():
        ordered = [line.strip() for line in hvgs_file.read_text().splitlines() if line.strip()]
        # If length matches, return directly; if longer, truncate; if shorter, rebuild to avoid shape mismatch
        if len(ordered) == config.model.n_genes:
            return ordered
        if len(ordered) > config.model.n_genes:
            return ordered[: config.model.n_genes]
        print(f"Found HVGs with length {len(ordered)} != expected {config.model.n_genes}. Rebuilding.")

    # Fallback: rebuild global HVGs the same way as training
    print("Global HVGs not found. Rebuilding from training data directory...")
    data_dir = Path(config.dataset.data_dir)
    h5ad_files = sorted(data_dir.glob('*.h5ad'))
    assert len(h5ad_files) > 0, "No h5ad files found to rebuild HVGs"

    vocab_genes = [g for g in tokenizer.get_stoi().keys() if not g.startswith('<')]

    adatas = []
    for i, p in enumerate(h5ad_files):
        ad = sc.read_h5ad(p)
        # Filter by vocabulary genes
        gene_mask = ad.var.index.isin(vocab_genes)
        ad = ad[:, gene_mask].copy()
        # Same preprocessing as training
        if config.dataset.normalize_total:
            sc.pp.normalize_total(ad, target_sum=config.dataset.normalize_total)
        if config.dataset.log1p:
            sc.pp.log1p(ad)
        ad.obs['batch'] = f'ds_{i}'
        adatas.append(ad)

    combined = sc.concat(adatas, join='outer', fill_value=0)
    sc.pp.highly_variable_genes(
        combined,
        n_top_genes=config.model.n_genes,
        batch_key='batch',
        flavor='seurat_v3'
    )
    ordered = combined.var.index[combined.var.highly_variable].tolist()

    # Persist for future evaluations
    save_dir.mkdir(parents=True, exist_ok=True)
    hvgs_file.write_text('\n'.join(ordered))
    return ordered

def evaluate_expression_reconstruction(model, gene_ids, gene_exp, coords, mask_ratio=0.15):
    """Evaluate expression reconstruction performance"""
    B, N, G = gene_exp.shape
    device = next(model.parameters()).device
    
    # Move data to device
    gene_ids = gene_ids.to(device)
    gene_exp = gene_exp.to(device)
    coords = coords.to(device)
    
    # Generate mask (similar to dataset)
    gene_mask = torch.zeros_like(gene_exp, dtype=torch.bool)
    masked_values = gene_exp.clone()
    
    # Apply masking
    valid_mask = (gene_ids != 0) & (gene_exp > 0)
    mask_prob = torch.rand_like(gene_exp)
    gene_mask = (mask_prob < mask_ratio) & valid_mask
    
    # Simple zero masking for evaluation
    masked_values[gene_mask] = 0
    
    # Forward pass with masked values using updated BERT interface
    # Call the underlying model directly to have control over use_predictor
    with torch.no_grad():
        predictions = model(gene_ids, gene_exp, masked_values, coords, gene_mask, use_predictor=True)
    
    # Get predictions for masked genes
    if 'expr_pred' in predictions and gene_mask.any():
        expr_pred = predictions['expr_pred'][gene_mask].cpu().numpy()
        expr_true = gene_exp[gene_mask].cpu().numpy()
        
        # Compute metrics
        mse = mean_squared_error(expr_true.flatten(), expr_pred.flatten())
        mae = mean_absolute_error(expr_true.flatten(), expr_pred.flatten())
        
        # Pearson correlation per spot
        correlations = []
        # Reshape to compute correlation per spot
        n_spots = B * N
        for spot_idx in range(n_spots):
            spot_mask = gene_mask.view(n_spots, -1)[spot_idx]
            if spot_mask.sum() > 1:  # Need at least 2 values for correlation
                pred_spot = predictions['expr_pred'].view(n_spots, -1)[spot_idx][spot_mask].cpu().numpy()
                true_spot = gene_exp.view(n_spots, -1)[spot_idx][spot_mask].cpu().numpy()
                if np.std(true_spot) > 0 and np.std(pred_spot) > 0:
                    corr, _ = pearsonr(true_spot, pred_spot)
                    correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        return {
            'expr_mse': mse,
            'expr_mae': mae,
            'expr_pearson': avg_correlation,
            'n_masked_genes': int(gene_mask.sum()),
            'n_spots_evaluated': len(correlations)
        }
    
    return None

def get_spot_embeddings(model, gene_ids, gene_exp, coords):
    """Get spot embeddings without any masking for clustering"""
    device = next(model.parameters()).device
    
    # Move data to device
    gene_ids = gene_ids.to(device)
    gene_exp = gene_exp.to(device)
    coords = coords.to(device)
    
    # Forward pass without masking (use original values as both input and target)
    # For embeddings, we don't need the predictor
    with torch.no_grad():
        predictions = model(gene_ids, gene_exp, gene_exp, coords, None, use_predictor=False)
        # cls_tokens now has shape [B, N, D]
        spot_embeddings = predictions['cls_tokens']
    
    return spot_embeddings.cpu().numpy()


def main():
    # Configuration
    checkpoint_path = './checkpoints/epoch=17-val_loss=0.7240.ckpt'  # Update path for BERT checkpoint
    data_dir = '/leonardo_work/EUHPC_B25_011/ST/DLPFC'
    
    # Load config for BERT
    config = Config.from_yaml('configs/bert_config.yaml')
    tokenizer = get_default_mae_tokenizer()
    config.model.vocab_size = len(tokenizer)
    config.model.padding_idx = tokenizer.pad_token_id
    
    # Load model
    print("Loading BERT model...")
    model = HiGeSTLightning.load_from_checkpoint(
        checkpoint_path, config=config, strict=False, map_location='cpu'
    )
    model.eval()
    
    # Using BERT architecture
    print("Using Spatial BERT architecture")
    
    # Load test slide
    h5ad_file = sorted(Path(data_dir).glob('*.h5ad'))[0]
    print(f"Loading data from {h5ad_file}")
    adata = sc.read_h5ad(h5ad_file)
    
    # Store original spatial coordinates
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
    
    # Preprocess expression exactly as in training: normalize_total -> log1p
    if config.dataset.normalize_total:
        sc.pp.normalize_total(adata, target_sum=config.dataset.normalize_total)
    if config.dataset.log1p:
        sc.pp.log1p(adata)

    # Load global HVG order saved during training; fallback to rebuild if missing
    ordered_genes = load_or_build_global_hvgs(config, tokenizer)

    # Construct expression matrix with the exact training gene order
    present = {g: i for i, g in enumerate(adata.var_names.tolist())}
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    X_full = np.zeros((X.shape[0], len(ordered_genes)), dtype=np.float32)
    for j, g in enumerate(ordered_genes):
        i = present.get(g)
        if i is not None:
            X_full[:, j] = X[:, i]

    gene_exp = X_full.astype(np.float32)
    gene_ids = tokenizer.encode_genes(ordered_genes)
    
    # Prepare tensors
    n_spots = len(coords)
    n_genes = len(ordered_genes)
    
    coords_t = torch.from_numpy(coords).float().unsqueeze(0)
    gene_exp_t = torch.from_numpy(gene_exp).float()
    gene_ids_t = torch.tensor(gene_ids).long()
    
    # Inputs are already aligned to the exact training gene order and size
    batch_gene_ids = gene_ids_t.unsqueeze(0).expand(1, n_spots, -1)
    batch_gene_exp = gene_exp_t.unsqueeze(0)
       
    # 1. Expression Reconstruction Evaluation
    print("\n1. Expression Reconstruction:")
    # Use the underlying BERT model directly for evaluation
    expr_metrics = evaluate_expression_reconstruction(
        model.model, batch_gene_ids, batch_gene_exp, coords_t, mask_ratio=0.15
    )
    if expr_metrics:
        print(f"   MSE: {expr_metrics['expr_mse']:.4f}")
        print(f"   MAE: {expr_metrics['expr_mae']:.4f}")
        print(f"   Avg Pearson Correlation: {expr_metrics['expr_pearson']:.4f}")
        print(f"   Masked genes: {expr_metrics['n_masked_genes']}")
        print(f"   Spots evaluated: {expr_metrics['n_spots_evaluated']}")
        
    # 2. Get embeddings for clustering (without masking)
    print("\n2. Spatial Clustering Evaluation:")
    # Use the underlying BERT model directly for embeddings
    features = get_spot_embeddings(model.model, batch_gene_ids, batch_gene_exp, coords_t)
    features = features.squeeze(0)  # Remove batch dimension
    
    # Store features in adata
    adata.obsm['X_bert'] = features
    
    # Clustering with scanpy
    sc.pp.neighbors(adata, use_rep='X_bert', n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.3, flavor='igraph', n_iterations=2, directed=False)
    
    # Clustering metrics
    predictions = adata.obs['leiden'].astype(int).values
    silhouette = silhouette_score(features, predictions)
    
    # Metrics with ground truth
    labels = adata.obs['ground_truth'].values
    valid_mask = ~pd.isna(labels)
    labels_valid = pd.factorize(labels[valid_mask])[0]
    pred_valid = predictions[valid_mask]
    ari = adjusted_rand_score(labels_valid, pred_valid)
    nmi = normalized_mutual_info_score(labels_valid, pred_valid)
    
    print(f"   Silhouette Score: {silhouette:.4f}")
    print(f"   ARI: {ari:.4f}")
    print(f"   NMI: {nmi:.4f}")
    
    # 4. Visualization
    print("\n4. Generating visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Spatial plots
    sc.pl.embedding(adata, basis='spatial', color='leiden', 
                    ax=axes[0,0], show=False, title='BERT Clustering', frameon=False)
    sc.pl.embedding(adata, basis='spatial', color='ground_truth',
                    ax=axes[0,1], show=False, title='Ground Truth', frameon=False)
    
    # UMAP plots
    sc.pl.umap(adata, color='leiden', ax=axes[1,0], show=False, title='UMAP - Clusters')
    sc.pl.umap(adata, color='ground_truth', ax=axes[1,1], show=False, title='UMAP - Ground Truth')
    
    # Feature distribution
    axes[2,0].hist(features.flatten(), bins=50, alpha=0.7, color='blue')
    axes[2,0].set_xlabel('Feature values')
    axes[2,0].set_ylabel('Frequency')
    axes[2,0].set_title('BERT Feature Distribution')
    
    # Cluster sizes
    cluster_sizes = pd.Series(predictions).value_counts().sort_index()
    axes[2,1].bar(cluster_sizes.index, cluster_sizes.values, color='green', alpha=0.7)
    axes[2,1].set_xlabel('Cluster ID')
    axes[2,1].set_ylabel('Number of spots')
    axes[2,1].set_title('Cluster Size Distribution')
    
    plt.tight_layout()
    plt.savefig('bert_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to bert_evaluation.png")
    
    # Save results
    results = {
        'expression_metrics': expr_metrics if expr_metrics else {},
        'clustering_metrics': {
            'silhouette': float(silhouette),
            'ari': float(ari),
            'nmi': float(nmi)
        }
    }
    
    import json
    with open('bert_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
    print(f"   Saved metrics to bert_evaluation_results.json")


if __name__ == "__main__":
    main()
