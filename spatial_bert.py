"""
Spatial BERT with replacement-based masking
Only visible spots go through gene attention, masked spots use mask tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np

from spatial_mae import (
    ValueEncoder,
    GeneEncoder,
    PositionalEncoding,
    GeneAttentionLayer, 
    CellAttentionLayer,
    FiLMLayer,
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
        gene_tokens: torch.Tensor,  # [n_visible, G, D]
        cls_tokens: torch.Tensor,   # [n_visible, 1, D]
        gene_padding_mask: torch.Tensor,  # [n_visible, G] - True for padding
        spot_embeddings: torch.Tensor,  # [B, N, D] - full tensor with mask tokens
        pos_encoding: torch.Tensor,  # [B, N, D]
        visible_mask: torch.Tensor,  # [B, N] - which spots are visible
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one layer: Gene attention for visible → Cell attention for all
        
        Returns:
            gene_tokens: Updated gene tokens for next layer
            cls_tokens: Updated CLS tokens for next layer  
            spot_embeddings: Updated spot embeddings
        """
        gene_tokens, cls_features = self.gene_layer(gene_tokens, cls_tokens, gene_padding_mask)
        
        # Update spot embeddings for visible spots
        batch_idx, spot_idx = visible_mask.nonzero(as_tuple=True)
        spot_embeddings[batch_idx, spot_idx] = cls_features

        # Cell attention for ALL spots (visible + masked)
        spot_embeddings = self.cell_layer(spot_embeddings, pos_encoding)

        # Use updated CLS for FiLM modulation
        updated_cls = spot_embeddings[batch_idx, spot_idx]
        gene_tokens = self.film_layer(gene_tokens, updated_cls)
        cls_tokens = updated_cls.unsqueeze(1)
        
        return gene_tokens, cls_tokens, spot_embeddings


class SpatialBERT(nn.Module):
    """
    Complete Spatial BERT model with replacement masking and task heads
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_genes: int,
        d_model: int,
        n_layers: int,  # Number of stacked layers
        n_heads: int,
        dropout: float,
        expr_mask_ratio: float,
        pos_mask_ratio: float,
        coord_noise_std: float,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.d_model = d_model
        self.expr_mask_ratio = expr_mask_ratio
        self.pos_mask_ratio = pos_mask_ratio
        self.coord_noise_std = coord_noise_std
        
        # Gene encoding components
        self.gene_encoder = GeneEncoder(vocab_size, d_model, padding_idx=0)
        self.value_encoder = ValueEncoder(d_model, dropout)
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_spot_token = nn.Parameter(torch.randn(1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Stack of layers
        self.layers = nn.ModuleList([
            SpatialBERTLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_output = nn.LayerNorm(d_model)
        
        # Expression reconstruction head
        self.expr_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_genes)
        )
        
        # Coordinate prediction head
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        coords: torch.Tensor,
        expr_mask: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual masking strategy
        
        Args:
            gene_ids: [B, N, G]
            gene_values: [B, N, G]
            coords: [B, N, 2]
            expr_mask: [B, N] cells for expression reconstruction
            pos_mask: [B, N] cells for coordinate prediction
        """
        B, N, G = gene_values.shape
        device = gene_values.device
        
        # Generate masks if training
        if expr_mask is None and self.training:
            expr_mask = torch.rand(B, N, device=device) < self.expr_mask_ratio
        if pos_mask is None and self.training:
            pos_mask = torch.rand(B, N, device=device) < self.pos_mask_ratio
            pos_mask = pos_mask & (~expr_mask)  # Ensure no overlap
            
        if expr_mask is None:
            expr_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        if pos_mask is None:
            pos_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # Add noise to coordinates for position-masked cells
        noisy_coords = coords.clone()
        if pos_mask.any():
            noise = torch.randn_like(coords[pos_mask]) * self.coord_noise_std
            noisy_coords[pos_mask] = coords[pos_mask] + noise
        
        # Initialize spot embeddings
        spot_embeddings = torch.zeros(B, N, self.d_model, device=device)
        
        # Get visible cells (not expression-masked)
        visible_mask = ~expr_mask
        batch_idx, spot_idx = visible_mask.nonzero(as_tuple=True)
        
        vis_gene_ids = gene_ids[batch_idx, spot_idx]
        vis_gene_vals = gene_values[batch_idx, spot_idx]
        
        # Create padding mask
        gene_padding_mask = (vis_gene_ids == 0)
        
        # Encode visible genes
        gene_embeds = self.gene_encoder(vis_gene_ids)
        value_embeds = self.value_encoder(vis_gene_vals)
        gene_tokens = gene_embeds + value_embeds
        
        # CLS tokens
        n_vis = gene_tokens.shape[0]
        cls_tokens = self.cls_token.expand(n_vis, 1, self.d_model)

        # Fill expression-masked spots with mask token
        spot_embeddings[expr_mask] = self.mask_spot_token
        
        # Get positional encoding (using noisy coordinates)
        pos_encoding = self.pos_encoder(noisy_coords)
        
        # Process through stacked layers
        for layer in self.layers:
            gene_tokens, cls_tokens, spot_embeddings = layer(
                gene_tokens, cls_tokens, gene_padding_mask, spot_embeddings, pos_encoding, visible_mask
            )
        
        spot_embeddings = self.ln_output(spot_embeddings)        
        result = {'spot_embeddings': spot_embeddings}
        
        # Dual predictions
        if expr_mask.any():
            expr_masked_embeds = spot_embeddings[expr_mask]
            expr_pred = self.expr_head(expr_masked_embeds)
            result['expr_pred'] = expr_pred
            result['expr_mask'] = expr_mask
            
        if pos_mask.any():
            pos_masked_embeds = spot_embeddings[pos_mask]
            coord_pred = self.coord_head(pos_masked_embeds)
            result['coord_pred'] = coord_pred
            result['pos_mask'] = pos_mask
            result['original_coords'] = coords[pos_mask]
        
        return result


def compute_bert_loss(
    predictions: Dict[str, torch.Tensor],
    gene_values: torch.Tensor,
    expr_weight: float = 1.0,
    coord_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    losses = {}
    
    # Expression reconstruction loss
    expr_pred = predictions.get('expr_pred')
    expr_mask = predictions.get('expr_mask')
    expr_target = gene_values[expr_mask]
    expr_loss = F.mse_loss(expr_pred, expr_target)
    losses['expr_loss'] = expr_loss

    # Coordinate prediction loss  
    coord_pred = predictions.get('coord_pred')
    original_coords = predictions.get('original_coords')
    coord_loss = F.mse_loss(coord_pred, original_coords)
    losses['coord_loss'] = coord_loss
    
    # Total loss
    total_loss = expr_weight * losses['expr_loss'] + coord_weight * losses['coord_loss']
    losses['total_loss'] = total_loss
    
    return losses
