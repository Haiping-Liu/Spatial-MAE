"""
MAE Dataset for Spatial Transcriptomics using scGPT tokenizer
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional, Dict
import scanpy as sc
from tqdm import tqdm
from pathlib import Path

from sampling import PatchSampler
from sklearn.cluster import KMeans
from mae_tokenizer import MAETokenizer, get_default_mae_tokenizer


class DatasetPath:
    """Container for dataset path information"""
    def __init__(self, name: str, source: str, h5ad_path: str, **kwargs):
        self.name = name
        self.source = source
        self.h5ad_path = h5ad_path
        
        # Store any additional attributes
        for k, v in kwargs.items():
            setattr(self, k, v)


class STData:
    """Container for spatial transcriptomics data"""
    def __init__(self, coords, gene_exp, gene_ids, gene_names):
        self.coords = coords  # [n_spots, 2]
        self.gene_exp = gene_exp  # [n_spots, n_genes]
        self.gene_ids = gene_ids  # [n_genes]
        self.gene_names = gene_names  # List of gene names
        
        # Normalize coordinates to [0, 100] range
        self.coords[:, 0] = self.coords[:, 0] - self.coords[:, 0].min()
        self.coords[:, 1] = self.coords[:, 1] - self.coords[:, 1].min()
        max_coord = self.coords.max()
        if max_coord > 0:
            self.coords = self.coords / max_coord * 100
    
    def __len__(self):
        return len(self.coords)
    
    def get_patch(self, indices):
        """Get a patch of spots by indices"""
        return self.coords[indices], self.gene_exp[indices]


class MAESTDataset(Dataset):
    """
    MAE Dataset for spatial transcriptomics with scGPT tokenizer
    
    Key improvements:
    - Uses pre-built scGPT vocabulary for stable gene IDs
    - Consistent gene filtering across all samples
    - Efficient HVG computation with batch correction
    """
    
    def __init__(
        self,
        dataset_list: List[DatasetPath],
        tokenizer: Optional[MAETokenizer] = None,
        n_spots: int = 128,
        n_hvg: int = 2000,
        use_hvg: bool = True,
        sampling_method: str = 'nearest',
        normalize_total: float = 1e4,
        log1p: bool = True,
        max_gene_len: int = 2000,
        patches_per_slide: int = 100,
        anchor_k: int = 16,
        anchor_method: str = 'kmeans++',
    ):
        """
        Initialize MAE dataset
        
        Args:
            dataset_list: List of dataset paths
            tokenizer: MAE tokenizer (if None, use default)
            n_spots: Number of spots per patch
            n_hvg: Number of highly variable genes
            use_hvg: Whether to use HVG selection
            sampling_method: Method for spatial patch sampling
            normalize_total: Total count normalization target
            log1p: Whether to apply log1p transformation
            max_gene_len: Maximum gene sequence length
            patches_per_slide: Number of patches to sample from each slide per epoch
        """
        super().__init__()
        
        self.dataset_list = dataset_list
        self.n_spots = n_spots
        self.n_hvg = n_hvg
        self.use_hvg = use_hvg
        self.normalize_total = normalize_total
        self.log1p = log1p
        self.max_gene_len = max_gene_len
        self.patches_per_slide = patches_per_slide
        self.anchor_k = anchor_k
        self.anchor_method = anchor_method
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = get_default_mae_tokenizer()
        else:
            self.tokenizer = tokenizer
        
        # Get vocabulary genes for filtering
        self.vocab_genes = list(self.tokenizer.get_stoi().keys())
        # Remove special tokens from gene list
        self.vocab_genes = [g for g in self.vocab_genes if not g.startswith("<")]
        print(f"Using vocabulary with {len(self.vocab_genes)} genes")
        
        # Get special token IDs
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_value = 0.0
        
        # Initialize patch sampler
        self.patch_sampler = PatchSampler(
            sampling_method=sampling_method,
            fixed_n_spots=n_spots
        )
        
        # Compute global HVGs if needed
        self.global_hvgs = None
        if use_hvg:
            self.global_hvgs = self._compute_global_hvgs()
        
        # Load all datasets
        self.st_datasets = []
        for dataset in tqdm(self.dataset_list):
            stdata = self.load_dataset(dataset)
            if stdata is not None:
                # compute anchors per slide
                anchors = self._compute_anchors(stdata.coords, self.anchor_k, self.anchor_method)
                stdata.anchors = torch.from_numpy(anchors).float()
                self.st_datasets.append(stdata)        
        print(f"Successfully loaded {len(self.st_datasets)}/{len(dataset_list)} datasets")

    def _compute_anchors(self, coords: torch.Tensor, k: int, method: str = 'kmeans++', seed: int = 42):
        """Compute per-slide global anchors from coordinates.
        Returns numpy array of shape [k, 2] in the same scale as input coords.
        """
        coords_np = coords.detach().cpu().numpy()
        n = len(coords_np)
        k = min(k, max(1, n))
        if method == 'kmeans++':
            # Use KMeans with k-means++ init; fall back to n_init=10 for broad sklearn compatibility
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
            km.fit(coords_np)
            return km.cluster_centers_.astype(coords_np.dtype)
        else:
            # Fallback: random sample as anchors
            idx = np.random.default_rng(seed).choice(n, size=k, replace=(k>n))
            return coords_np[idx]
    
    def _compute_global_hvgs(self) -> List[str]:
        """
        Compute HVGs globally across all datasets with batch correction
        
        Returns:
            List of HVG gene names
        """
        all_adatas = []
        
        for dataset in tqdm(self.dataset_list, desc="Processing for HVG"):
            adata = sc.read_h5ad(dataset.h5ad_path)
            
            # Filter by vocabulary genes
            gene_mask = adata.var.index.isin(self.vocab_genes)
            adata = adata[:, gene_mask].copy()
            
            if self.normalize_total:
                sc.pp.normalize_total(adata, target_sum=self.normalize_total)
            if self.log1p:
                sc.pp.log1p(adata)
            
            # Add batch information
            adata.obs['batch'] = dataset.name
            all_adatas.append(adata)
        
        combined = sc.concat(all_adatas, join='outer', fill_value=0)
        sc.pp.highly_variable_genes(
            combined,
            n_top_genes=self.n_hvg,
            batch_key='batch',
            flavor='seurat_v3'
        )
        hvg_genes = combined.var.index[combined.var.highly_variable].tolist()
        
        del combined, all_adatas
        
        return hvg_genes
    
    def load_dataset(self, dataset):
        """Load and preprocess a single dataset"""
        adata = sc.read_h5ad(dataset.h5ad_path)
        coords = torch.from_numpy(adata.obsm["spatial"]).float()
        
        # Filter by vocab
        gene_mask = adata.var.index.isin(self.vocab_genes)
        adata = adata[:, gene_mask].copy()
        
        # Preprocessing
        if self.normalize_total:
            sc.pp.normalize_total(adata, target_sum=self.normalize_total)
        if self.log1p:
            sc.pp.log1p(adata)
        
        # Apply HVG selection
        if self.global_hvgs is not None:
            hvg_mask = adata.var.index.isin(self.global_hvgs)
            adata = adata[:, hvg_mask].copy()
        
        # Get gene info
        gene_names = adata.var.index.tolist()
        gene_ids = self.tokenizer.encode_genes(gene_names)
        
        # Get expression matrix
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        gene_exp = torch.from_numpy(X).float()
        
        return STData(coords, gene_exp, gene_ids, gene_names)
    
    def __len__(self):
        # Total number of patches = slides × patches_per_slide
        return len(self.st_datasets) * self.patches_per_slide
    
    def __getitem__(self, idx):
        """
        Get a spatial patch with fixed dimensions
        
        Returns:
            Dictionary with:
                - gene_exp: Expression matrix [n_spots, max_gene_len]
                - coords: Spatial coordinates [n_spots, 2]
                - gene_ids: Gene token IDs [max_gene_len]
        """
        # Calculate which slide and patch within that slide
        slide_idx = idx // self.patches_per_slide
        patch_idx = idx % self.patches_per_slide
        
        st_data = self.st_datasets[slide_idx]
        
        # Sample a spatial patch
        sampled_indices = self.patch_sampler(st_data.coords.numpy())
        coords, gene_exp = st_data.get_patch(sampled_indices)
        
        # Pad or truncate to fixed length
        gene_ids, gene_exp = self.tokenizer.pad_sequence(
            st_data.gene_ids,
            gene_exp,
            target_length=self.max_gene_len
        )
        
        return {
            'gene_exp': gene_exp,  # [n_spots, max_gene_len]
            'coords': coords,  # [n_spots, 2]
            'gene_ids': gene_ids,  # [max_gene_len]
            'anchors': st_data.anchors,  # [K, 2]
        }


def mae_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    gene_exp = torch.stack([s['gene_exp'] for s in batch])  # [B, n_spots, max_gene_len]
    coords = torch.stack([s['coords'] for s in batch])  # [B, n_spots, 2]
    gene_ids = torch.stack([s['gene_ids'] for s in batch])  # [B, max_gene_len]
    anchors = torch.stack([s['anchors'] for s in batch])  # [B, K, 2]
    
    # Expand gene_ids to match spatial dimensions: [B, max_gene_len] -> [B, n_spots, max_gene_len]
    batch_size, n_spots, n_genes = gene_exp.shape
    gene_ids = gene_ids.unsqueeze(1).expand(batch_size, n_spots, n_genes)
    
    return {
        'gene_values': gene_exp,  # Rename to match Lightning module interface
        'coords': coords,
        'gene_ids': gene_ids,
        'anchors': anchors,
    }


if __name__ == "__main__":
    # Test with DLPFC data
    print("Testing MAE Dataset with scGPT tokenizer")
    print("=" * 60)
    
    # Create sample dataset list
    data_path = "/leonardo_work/EUHPC_B25_011/ST/DLPFC"
    dataset_list = []
    
    for i in range(1, 3):  # Test with first 2 samples
        h5ad_path = f"{data_path}/sample_{i}.h5ad"
        dataset_list.append(
            DatasetPath(
                name=f"DLPFC_sample_{i}",
                source="DLPFC",
                h5ad_path=h5ad_path
            )
        )
    
    # Initialize tokenizer
    tokenizer = get_default_mae_tokenizer()
    
    # Create dataset
    dataset = MAESTDataset(
        dataset_list=dataset_list,
        tokenizer=tokenizer,
        n_spots=128,
        n_hvg=1000,
        use_hvg=True,
        max_gene_len=1000
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample shapes:")
        print(f"  gene_exp: {sample['gene_exp'].shape}")
        print(f"  coords: {sample['coords'].shape}")
        print(f"  gene_ids: {sample['gene_ids'].shape}")
        
        # Test batch collation
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=mae_collate_fn,
            num_workers=0
        )
        
        for batch in dataloader:
            print(f"\nBatch shapes:")
            print(f"  gene_exp: {batch['gene_exp'].shape}")
            print(f"  coords: {batch['coords'].shape}")
            print(f"  gene_ids: {batch['gene_ids'].shape}")
            break
