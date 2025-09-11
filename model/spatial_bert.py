import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from model.layers import (
    ValueEncoder,
    GeneEncoder,
    PositionalEncoding,
    GeneAttentionLayer, 
    CellAttentionLayer,
    FiLMLayer,
    MVCDecoder,
)


class SpatialBERTLayer(nn.Module):
    """
    Single layer of Spatial BERT: Gene attention (visible only) → Cell attention (all)
    Similar to SpatialEncoderLayer but with masking
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.gene_layer = GeneAttentionLayer(d_model, n_heads, dropout)
        self.cell_layer = CellAttentionLayer(d_model, n_heads, dropout)
        self.film_layer = FiLMLayer(d_model)
    
    def forward(
        self,
        gene_tokens: torch.Tensor,  # [B*N, G, D]
        cls_tokens: torch.Tensor,   # [B*N, 1, D]
        gene_padding_mask: torch.Tensor,  # [B*N, G] - True for padding
        spot_embeddings: torch.Tensor,  # [B, N, D] - full tensor
        pos_encoding: torch.Tensor,  # [B, N, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one layer: Gene attention → Cell attention
        
        Returns:
            gene_tokens: Updated gene tokens for next layer
            cls_tokens: Updated CLS tokens for next layer  
            spot_embeddings: Updated spot embeddings
        """
        # Gene attention
        gene_tokens, cls_features = self.gene_layer(gene_tokens, cls_tokens, gene_padding_mask)
        
        # Reshape cls_features to match spot_embeddings
        B, N, D = spot_embeddings.shape
        cls_features_reshaped = cls_features.view(B, N, D)
        
        # Update spot embeddings with new CLS features
        spot_embeddings = cls_features_reshaped

        # Cell attention for all spots
        spot_embeddings = self.cell_layer(spot_embeddings, pos_encoding)

        # Flatten back for FiLM modulation
        updated_cls = spot_embeddings.view(B * N, D)
        gene_tokens = self.film_layer(gene_tokens, updated_cls)
        cls_tokens = updated_cls.unsqueeze(1)
        
        return gene_tokens, cls_tokens, spot_embeddings


class SpatialBERT(nn.Module):
    """
    Spatial BERT with gene-level masking (80/10/10 strategy)
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_genes: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        gene_mask_ratio: float = 0.15,
        noise_scale: float = 0.1,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.d_model = d_model
        self.gene_mask_ratio = gene_mask_ratio
        self.noise_scale = noise_scale
        
        # Gene encoding components
        self.gene_encoder = GeneEncoder(vocab_size, d_model, padding_idx=0)
        self.value_encoder = ValueEncoder(d_model, dropout)
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Stack of layers
        self.layers = nn.ModuleList([
            SpatialBERTLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_output = nn.LayerNorm(d_model)
        
        # Gene reconstruction decoder (cell-conditioned masked value prediction)
        self.expr_decoder = MVCDecoder(
            d_model=d_model,
            arch_style="inner product",  # or "inner product, detach"
            query_activation=nn.Sigmoid,
            hidden_activation=nn.PReLU,
            explicit_zero_prob=False,
            use_batch_labels=False,
        )
        
    def generate_gene_mask(self, gene_values: torch.Tensor, gene_ids: torch.Tensor):
        """Generate gene-level mask with 80/10/10 strategy"""
        B, N, G = gene_values.shape
        device = gene_values.device
        
        # Valid genes: non-padding and positive expression
        valid_mask = (gene_ids != 0) & (gene_values > 0)
        
        # Random mask for valid genes
        mask_prob = torch.rand_like(gene_values)
        gene_mask = (mask_prob < self.gene_mask_ratio) & valid_mask
        
        # Split masked genes into 80/10/10
        mask_indices = gene_mask.nonzero(as_tuple=True)
        n_masked = len(mask_indices[0])
        
        # Shuffle and split
        perm = torch.randperm(n_masked, device=device)
        n_zero = int(0.8 * n_masked)
        n_keep = int(0.1 * n_masked)
        
        # Create mask types: 0=unmask, 1=zero, 2=keep, 3=noise
        mask_type = torch.zeros_like(gene_values, dtype=torch.int8)
        mask_type[mask_indices] = 3  # Default to noise
        
        # Apply splits
        zero_idx = perm[:n_zero]
        keep_idx = perm[n_zero:n_zero + n_keep]
        
        mask_type[
            mask_indices[0][zero_idx],
            mask_indices[1][zero_idx],
            mask_indices[2][zero_idx]
        ] = 1
        
        mask_type[
            mask_indices[0][keep_idx],
            mask_indices[1][keep_idx],
            mask_indices[2][keep_idx]
        ] = 2
        
        return gene_mask, mask_type
    
    def apply_mask(self, gene_values: torch.Tensor, mask_type: torch.Tensor):
        """Apply masking operations based on mask type"""
        masked_values = gene_values.clone()
        
        # 80%: zero out
        masked_values[mask_type == 1] = 0
        
        # 10%: keep original (no change)
        
        # 10%: add noise
        noise_mask = (mask_type == 3)
        if noise_mask.any():
            noise = torch.randn_like(masked_values[noise_mask]) * self.noise_scale
            masked_values[noise_mask] += noise
        
        return masked_values
    
    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        coords: torch.Tensor,
        apply_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gene-level masking
        
        Args:
            gene_ids: [B, N, G]
            gene_values: [B, N, G]
            coords: [B, N, 2]
            apply_mask: Whether to apply masking (False for teacher)
        """
        B, N, G = gene_values.shape
        device = gene_values.device
        
        # Generate and apply gene-level mask during training
        if self.training and apply_mask:
            gene_mask, mask_type = self.generate_gene_mask(gene_values, gene_ids)
            masked_values = self.apply_mask(gene_values, mask_type)
        else:
            masked_values = gene_values
            gene_mask = None
        
        # Flatten batch and spatial dimensions for processing
        gene_ids_flat = gene_ids.view(B * N, G)
        masked_values_flat = masked_values.view(B * N, G)
        
        # Create padding mask
        gene_padding_mask = (gene_ids_flat == 0)
        
        # Encode all genes (with masked values)
        gene_embeds = self.gene_encoder(gene_ids_flat)
        value_embeds = self.value_encoder(masked_values_flat)
        gene_tokens = gene_embeds + value_embeds
        
        # CLS tokens for all spots
        cls_tokens = self.cls_token.expand(B * N, 1, self.d_model)
        
        # Get positional encoding
        pos_encoding = self.pos_encoder(coords)
        
        # Initialize spot embeddings from CLS tokens
        spot_embeddings = cls_tokens.squeeze(1).view(B, N, self.d_model)
        
        # Process through stacked layers (all spots participate)
        for layer in self.layers:
            gene_tokens, cls_tokens, spot_embeddings = layer(
                gene_tokens, cls_tokens, gene_padding_mask, spot_embeddings, pos_encoding
            )
        
        spot_embeddings = self.ln_output(spot_embeddings)
        
        # Reconstruction using MVCDecoder conditioned on CLS tokens
        cls_features = cls_tokens.squeeze(1)              # [B*N, D]
        out = self.expr_decoder(cls_features, gene_embeds)  # pred: [B*N, G]
        expr_pred = out["pred"].view(B, N, G)
        
        result = {
            'spot_embeddings': spot_embeddings,
            'cls_tokens': cls_features,  # [B, N, D] for topology loss
            'expr_pred': expr_pred,
            'gene_mask': gene_mask,
            'gene_values': gene_values,  # Original values for loss
        }
        
        return result


def build_adjacency_matrix(coords: torch.Tensor, k: int = 6) -> torch.Tensor:
    """
    Build KNN adjacency matrix for spatial topology
    
    Args:
        coords: [B, N, 2] - spatial coordinates
        k: number of nearest neighbors
    
    Returns:
        adj_matrix: [B, N, N] - binary adjacency matrix
    """
    B, N, _ = coords.shape
    device = coords.device
    
    # Compute pairwise distances [B, N, N]
    dist = torch.cdist(coords, coords, p=2)
    
    # Get k nearest neighbors (excluding self)
    # Set diagonal to large value to exclude self
    dist_with_inf = dist + torch.eye(N, device=device).unsqueeze(0) * 1e10
    
    # Find k nearest neighbors for each spot
    _, knn_indices = torch.topk(dist_with_inf, k=k, dim=-1, largest=False)
    
    # Build adjacency matrix
    adj_matrix = torch.zeros(B, N, N, device=device)
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
    row_indices = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
    
    adj_matrix[batch_indices, row_indices, knn_indices] = 1.0
    
    # Make symmetric
    adj_matrix = torch.maximum(adj_matrix, adj_matrix.transpose(1, 2))
    
    return adj_matrix


def compute_topology_loss(
    student_cls: torch.Tensor,
    teacher_cls: torch.Tensor,
    adj_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute topology-aware contrastive loss (only on KNN edges)
    
    Args:
        student_cls: [B, N, D] - student CLS tokens (masked input)
        teacher_cls: [B, N, D] - teacher CLS tokens (full input, no grad)
        adj_matrix: [B, N, N] - binary adjacency matrix
    
    Returns:
        topology_loss: scalar loss
    """
    # L2 normalize for cosine similarity
    student_norm = F.normalize(student_cls, dim=-1)
    teacher_norm = F.normalize(teacher_cls, dim=-1)
    
    # Compute similarity matrix [B, N, N]
    similarity = torch.bmm(student_norm, teacher_norm.transpose(1, 2))
    
    # Only compute loss on edges (neighbors)
    edge_mask = adj_matrix.bool()
    edge_similarities = similarity[edge_mask]
    
    # Target: neighbors should have high similarity (cos=1)
    target = torch.ones_like(edge_similarities)
    
    # MSE loss only on edges
    topology_loss = F.mse_loss(edge_similarities, target)
    
    return topology_loss


def compute_bert_loss(
    predictions: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute reconstruction loss only for masked genes"""
    
    expr_pred = predictions['expr_pred']
    gene_values = predictions['gene_values']
    gene_mask = predictions['gene_mask']
    
    losses = {}
    
    if gene_mask is not None and gene_mask.any():
        # Only compute loss on masked positions
        pred_masked = expr_pred[gene_mask]
        target_masked = gene_values[gene_mask]
        losses['loss'] = F.mse_loss(pred_masked, target_masked)
    else:
        # No mask (evaluation mode)
        losses['loss'] = torch.tensor(0.0, device=expr_pred.device)
    
    return losses
