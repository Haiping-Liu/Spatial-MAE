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


class HiGeSTLayer(nn.Module):
    """
    Single layer of HiGeST: Gene attention (visible only) → Cell attention (all)
    Hierarchical Gene expression in Spatial Transcriptomics
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
        pos_encoding: torch.Tensor,  # [B, N, D]
        batch_size: int,
        n_spots: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one layer: Gene attention → Cell attention
        
        Returns:
            gene_tokens: Updated gene tokens for next layer
            cls_tokens: Updated CLS tokens for next layer  
        """
        # Gene attention
        gene_tokens, cls_features = self.gene_layer(gene_tokens, cls_tokens, gene_padding_mask)
        
        # Reshape for cell attention
        cls_3d = cls_features.view(batch_size, n_spots, -1)  # [B, N, D]
        
        # Cell attention for all spots
        cls_3d = self.cell_layer(cls_3d, pos_encoding)

        # Flatten back for gene processing
        cls_features = cls_3d.view(batch_size * n_spots, -1)  # [B*N, D]
        gene_tokens = self.film_layer(gene_tokens, cls_features)
        cls_tokens = cls_features.unsqueeze(1)  # [B*N, 1, D]
        
        return gene_tokens, cls_tokens


class HiGeST(nn.Module):
    """
    HiGeST: Hierarchical Gene expression in Spatial Transcriptomics
    with gene-level masking (80/10/10 strategy)
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_genes: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.d_model = d_model
        
        # Gene encoding components
        self.gene_encoder = GeneEncoder(vocab_size, d_model, padding_idx=0)
        self.value_encoder = ValueEncoder(d_model, dropout)
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Stack of layers
        self.layers = nn.ModuleList([
            HiGeSTLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_output = nn.LayerNorm(d_model)
        
        # BYOL-style projector (for both student and teacher)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, 256)
        )
        
        # BYOL-style predictor (only for student)
        self.predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
        
        # Gene reconstruction decoder (cell-conditioned masked value prediction)
        self.expr_decoder = MVCDecoder(
            d_model=d_model,
            arch_style="inner product",  # or "inner product, detach"
            query_activation=nn.Sigmoid,
            hidden_activation=nn.PReLU,
            explicit_zero_prob=False,
            use_batch_labels=False,
        )
    
    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        masked_values: torch.Tensor,
        coords: torch.Tensor,
        gene_mask: torch.Tensor = None,
        use_predictor: bool = True,  # Set to False for teacher model
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gene-level masking
        
        Args:
            gene_ids: [B, N, G]
            gene_values: [B, N, G] - Original values for loss
            masked_values: [B, N, G] - Values after 80/10/10 masking
            coords: [B, N, 2]
            gene_mask: [B, N, G] - Boolean mask for genes to reconstruct
        """
        B, N, G = gene_values.shape
        
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
        
        # Process through stacked layers
        for layer in self.layers:
            gene_tokens, cls_tokens = layer(
                gene_tokens, cls_tokens, gene_padding_mask, pos_encoding, B, N
            )
        
        # Apply layer norm to cls tokens
        cls_features = cls_tokens.squeeze(1)  # [B*N, D]
        cls_features_3d = cls_features.view(B, N, self.d_model)  # [B, N, D]
        cls_features_3d = self.ln_output(cls_features_3d)
        
        # Reconstruction using MVCDecoder conditioned on CLS tokens
        cls_features_flat = cls_features_3d.view(B * N, self.d_model)  # [B*N, D]
        out = self.expr_decoder(cls_features_flat, gene_embeds)  # pred: [B*N, G]
        expr_pred = out["pred"].view(B, N, G)
        
        # Apply projector to cls tokens for topology learning
        B, N, D = cls_features_3d.shape
        cls_flat = cls_features_3d.view(B * N, D)  # [B*N, D]
        cls_projected = self.projector(cls_flat)  # [B*N, 256]
        cls_projected_3d = cls_projected.view(B, N, 256)  # [B, N, 256]
        
        # Apply predictor (only for student, will be skipped for teacher)
        if use_predictor:
            cls_predicted = self.predictor(cls_projected)  # [B*N, 256]
            cls_predicted_3d = cls_predicted.view(B, N, 256)  # [B, N, 256]
        else:
            cls_predicted_3d = None  # Teacher doesn't use predictor
        
        result = {
            'cls_tokens': cls_features_3d,  # [B, N, D] original features
            'cls_projected': cls_projected_3d,  # [B, N, 256] after projector
            'expr_pred': expr_pred,
            'gene_mask': gene_mask,
            'gene_values': gene_values,  # Original values for loss
        }
        
        if cls_predicted_3d is not None:
            result['cls_predicted'] = cls_predicted_3d  # [B, N, 256] after predictor (student only)
        
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
