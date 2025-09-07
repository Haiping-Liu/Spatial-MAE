"""
Spatial MAE - Clean implementation with proper masking strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class PositionalEncoding(nn.Module):
    """2D sinusoidal positional encoding for spatial coordinates"""
    def __init__(self, d_model: int, freqs_scale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.freqs_scale = freqs_scale
        
        # Ensure d_model is even so we can split between x and y
        assert d_model % 2 == 0, "d_model must be even for 2D positional encoding"
        
        # Create div_term for sinusoidal encoding (for d_model/4 dimensions each for x,y)
        dim_t = torch.arange(0, d_model // 4, dtype=torch.float32)
        self.register_buffer('div_term', freqs_scale ** (dim_t / (d_model // 4)))
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [bs, n_spots, 2] or [bs*n_spots, 2] - (x, y) coordinates
        Returns:
            pos_encoding: [bs, n_spots, d_model] or [bs*n_spots, d_model]
        """
        # Split x and y coordinates
        x_pos = coords[..., 0:1]  # [..., 1]
        y_pos = coords[..., 1:2]  # [..., 1]
        
        # Compute 1D PE for x coordinate (d_model/2 dimensions)
        x_div = x_pos / self.div_term  # [..., d_model//4]
        x_pe = torch.zeros(*coords.shape[:-1], self.d_model // 2, 
                          device=coords.device, dtype=coords.dtype)
        x_pe[..., 0::2] = torch.sin(x_div)
        x_pe[..., 1::2] = torch.cos(x_div)
        
        # Compute 1D PE for y coordinate (d_model/2 dimensions)
        y_div = y_pos / self.div_term  # [..., d_model//4]
        y_pe = torch.zeros(*coords.shape[:-1], self.d_model // 2,
                          device=coords.device, dtype=coords.dtype)
        y_pe[..., 0::2] = torch.sin(y_div)
        y_pe[..., 1::2] = torch.cos(y_div)
        
        # Concatenate x and y positional encodings
        pos_encoding = torch.cat([x_pe, y_pe], dim=-1)  # [..., d_model]
        
        return pos_encoding


class GeneEncoder(nn.Module):
    """Encode gene IDs"""
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ValueEncoder(nn.Module):
    """Encode gene expression values"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class GeneAttentionLayer(nn.Module):
    """Gene-level attention with CLS token"""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        gene_tokens: torch.Tensor,
        cls_token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gene_tokens: [bs*n_spots, n_genes, d_model]
            cls_token: [bs*n_spots, 1, d_model]
        Returns:
            gene_tokens: [bs*n_spots, n_genes, d_model]
            cls_features: [bs*n_spots, d_model]
        """
        # Concatenate CLS with gene tokens
        tokens = torch.cat([cls_token, gene_tokens], dim=1)  # [bs*n_spots, n_genes+1, d_model]
        
        # Self-attention
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + self.dropout(ffn_out))
        
        # Split CLS and gene tokens
        cls_features = tokens[:, 0, :]  # [bs*n_spots, d_model]
        gene_tokens = tokens[:, 1:, :]  # [bs*n_spots, n_genes, d_model]
        
        return gene_tokens, cls_features


class CellAttentionLayer(nn.Module):
    """Cell-level attention among CLS tokens"""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        cls_features: torch.Tensor,
        pos_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cls_features: [bs, n_visible, d_model] - only visible cells
            pos_encoding: [bs, n_visible, d_model] - only visible cells
        Returns:
            cls_features: [bs, n_visible, d_model]
        """
        # Add positional encoding
        cls_with_pos = cls_features + pos_encoding
        
        # Self-attention (no mask needed - all cells are visible)
        attn_out, _ = self.attention(
            cls_with_pos, cls_with_pos, cls_with_pos
        )
        cls_features = self.norm1(cls_features + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(cls_features)
        cls_features = self.norm2(cls_features + self.dropout(ffn_out))
        
        return cls_features


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, d_model: int):
        super().__init__()
        
        self.film_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2)
        )
    
    def forward(
        self,
        gene_features: torch.Tensor,
        cls_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            gene_features: [bs*n_spots, n_genes, d_model]
            cls_features: [bs*n_spots, d_model]
        Returns:
            modulated: [bs*n_spots, n_genes, d_model]
        """
        # Generate FiLM parameters
        film_params = self.film_generator(cls_features)
        scale, shift = film_params.chunk(2, dim=-1)
        
        # Apply to each gene
        scale = scale.unsqueeze(1)  # [bs*n_spots, 1, d_model]
        shift = shift.unsqueeze(1)
        
        modulated = gene_features * (1 + scale) + shift
        
        return modulated


class SpatialEncoderLayer(nn.Module):
    """Single encoder layer with Gene-Cell alternating attention and FiLM"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float
    ):
        super().__init__()
        self.d_model = d_model
        
        # Three sub-layers
        self.gene_attention = GeneAttentionLayer(d_model, n_heads, dropout)
        self.cell_attention = CellAttentionLayer(d_model, n_heads, dropout)
        self.film_layer = FiLMLayer(d_model)
    
    def forward(
        self,
        gene_tokens: torch.Tensor,
        cls_tokens: torch.Tensor,
        pos_encoding: torch.Tensor,
        bs: int,
        n_visible: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gene_tokens: [bs*n_visible, n_genes, d_model]
            cls_tokens: [bs*n_visible, 1, d_model]
            pos_encoding: [bs, n_visible, d_model]
            bs: batch size
            n_visible: number of visible spots
        Returns:
            gene_tokens: [bs*n_visible, n_genes, d_model]
            cls_tokens: [bs*n_visible, 1, d_model]
        """
        # Gene attention (CLS participates)
        gene_tokens, cls_features = self.gene_attention(gene_tokens, cls_tokens)
        
        # Reshape for cell attention
        cls_features = cls_features.reshape(bs, n_visible, self.d_model)
        
        # Cell attention (CLS tokens interact with positional encoding)
        cls_features = self.cell_attention(cls_features, pos_encoding)
        
        # Flatten back
        cls_features_flat = cls_features.reshape(bs * n_visible, self.d_model)
        
        # FiLM modulation on gene tokens
        gene_tokens = self.film_layer(gene_tokens, cls_features_flat)
        
        # Update CLS tokens for next iteration
        cls_tokens = cls_features_flat.unsqueeze(1)
        
        return gene_tokens, cls_tokens



class SpatialMAEEncoder(nn.Module):
    """Encoder with alternating Gene-Cell attention - only processes visible cells"""
    def __init__(
        self,
        vocab_size: int,  # Vocabulary size for gene embeddings
        n_genes: int,     # Number of genes per sample (sequence length)
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        mask_ratio: float = 0.75,
        padding_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_genes = n_genes
        self.d_model = d_model
        self.n_layers = n_layers
        self.mask_ratio = mask_ratio
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.gene_encoder = GeneEncoder(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx)
        self.value_encoder = ValueEncoder(d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            SpatialEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)  # Each layer does Gene->Cell->FiLM
        ])
    
    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_exp: torch.Tensor,
        coords: torch.Tensor,
        mask_info: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            gene_ids: [bs, n_spots, n_genes] - gene IDs
            gene_exp: [bs, n_spots, n_genes] - gene expression values
            coords: [bs, n_spots, 2] - spatial coordinates
            mask_info: Optional pre-computed mask info from generate_random_mask
        
        Returns:
            encoded_features: [bs, n_visible, d_model] - encoded visible features
            mask_info: Dictionary with mask, masked_indices, visible_indices
        """
        bs, n_spots, n_genes = gene_exp.shape
        
        visible_indices = mask_info['visible_indices']
        
        # Gather visible cells using advanced indexing
        batch_indices = torch.arange(bs, device=gene_exp.device).unsqueeze(1).expand(-1, visible_indices.size(1))
        
        visible_gene_ids = gene_ids[batch_indices, visible_indices]  # [bs, n_visible, n_genes]
        visible_gene_exp = gene_exp[batch_indices, visible_indices]  # [bs, n_visible, n_genes]
        visible_coords = coords[batch_indices, visible_indices]  # [bs, n_visible, 2]
        
        n_visible = visible_indices.size(1)
        
        # Get positional encoding for visible spots
        pos_encoding = self.pos_encoder(visible_coords)  # [bs, n_visible, d_model]
        
        # Get gene and value embeddings
        gene_embs = self.gene_encoder(visible_gene_ids)  # [bs, n_visible, n_genes, d_model]
        value_embs = self.value_encoder(visible_gene_exp)  # [bs, n_visible, n_genes, d_model]
        
        # Combine embeddings
        gene_tokens = gene_embs + value_embs  # [bs, n_visible, n_genes, d_model]
        
        # Reshape for processing
        gene_tokens = gene_tokens.reshape(bs * n_visible, n_genes, self.d_model)
        
        # Initialize CLS token for each visible cell
        cls_tokens = self.cls_token.expand(bs * n_visible, 1, self.d_model)
        
        # Process through stacked encoder layers
        for layer in self.layers:
            gene_tokens, cls_tokens = layer(
                gene_tokens, cls_tokens, pos_encoding, bs, n_visible
            )
        
        # Extract final CLS features and reshape
        cls_features = cls_tokens.squeeze(1).reshape(bs, n_visible, self.d_model)
        
        return cls_features, mask_info


class SpatialMAEDecoder(nn.Module):
    """Lightweight decoder that reconstructs masked positions"""
    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        d_decoder: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        anchor_k: int = 0,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.d_decoder = d_decoder
        self.anchor_k = anchor_k
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_decoder)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_decoder,
                nhead=n_heads,
                dim_feedforward=d_decoder * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_decoder, n_genes)
        
        # Optional anchor distance head: predict distances to K anchors per spot
        if anchor_k and anchor_k > 0:
            self.anchor_dist_head = nn.Sequential(
                nn.Linear(d_decoder, d_decoder),
                nn.GELU(),
                nn.Linear(d_decoder, anchor_k)
            )
        else:
            self.anchor_dist_head = None
    
    def forward(
        self,
        encoded_features: torch.Tensor,
        coords: torch.Tensor,
        mask_info: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            encoded_features: [bs, n_visible, d_model] - encoder output for visible positions
            coords: [bs, n_spots, 2] - all spatial coordinates
            mask_info: Dictionary with mask, masked_indices, visible_indices
        
        Returns:
            predictions: [bs, n_spots, n_genes] - predictions for all positions
        """
        bs, n_visible, d_model = encoded_features.shape
        n_spots = coords.size(1)
        
        visible_indices = mask_info['visible_indices']
        masked_indices = mask_info['masked_indices']
        n_masked = masked_indices.size(1)
        
        # Create full sequence with visible features and mask tokens
        full_sequence = torch.zeros(bs, n_spots, d_model, device=encoded_features.device)
        
        # Place visible features at correct positions using advanced indexing
        batch_idx = torch.arange(bs, device=encoded_features.device).unsqueeze(1)
        full_sequence[batch_idx, visible_indices] = encoded_features
        
        # Fill masked positions with mask tokens
        mask_tokens = self.mask_token.expand(bs, n_masked, d_model)
        full_sequence[batch_idx, masked_indices] = mask_tokens
        
        # Add positional encoding to all positions
        pos_encoding = self.pos_encoder(coords)  # [bs, n_spots, d_model]
        full_sequence = full_sequence + pos_encoding
        
        # Project to decoder dimension
        decoder_features = self.input_proj(full_sequence)  # [bs, n_spots, d_decoder]
        
        # Process through transformer layers
        for layer in self.layers:
            decoder_features = layer(decoder_features)
        
        # Predict gene expression for all positions
        predictions = self.output_proj(decoder_features)  # [bs, n_spots, n_genes]
        
        # Optional: predict distances to global anchors per spot
        anchor_dist_pred = None
        if self.anchor_dist_head is not None:
            # Non-negative distances are encouraged via relu at loss time; here leave raw
            anchor_dist_pred = self.anchor_dist_head(decoder_features)  # [bs, n_spots, K]
        
        return predictions, anchor_dist_pred


class SpatialMAE(nn.Module):
    """Complete Spatial MAE model with proper masking"""
    def __init__(
        self,
        vocab_size: int,  # Vocabulary size for gene embeddings
        n_genes: int,     # Number of genes per sample (sequence length)
        d_model: int = 256,
        d_decoder: int = 128,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        mask_ratio: float = 0.75,
        padding_idx: int = 0,
        anchor_k: int = 0,
        use_anchor_loss: bool = False,
        anchor_loss_weight: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_genes = n_genes
        self.mask_ratio = mask_ratio
        self.anchor_k = anchor_k
        self.use_anchor_loss = use_anchor_loss and (anchor_k is not None and anchor_k > 0)
        self.anchor_loss_weight = anchor_loss_weight
        
        self.encoder = SpatialMAEEncoder(
            vocab_size=vocab_size,
            n_genes=n_genes,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
            mask_ratio=mask_ratio,
            padding_idx=padding_idx
        )
        
        self.decoder = SpatialMAEDecoder(
            n_genes=n_genes,
            d_model=d_model,
            d_decoder=d_decoder,
            n_layers=n_decoder_layers,
            n_heads=n_heads // 2,
            dropout=dropout,
            anchor_k=anchor_k,
        )
    
    def generate_random_mask(self, batch_size: int, n_spots: int, mask_ratio: float, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Generate random mask without using for loops
        
        Args:
            batch_size: batch size
            n_spots: number of spots
            mask_ratio: ratio of spots to mask
            device: device to create tensors on
        
        Returns:
            Dictionary with:
                - mask: [bs, n_spots] boolean mask (True for masked)
                - masked_indices: [bs, n_masked] indices of masked spots
                - visible_indices: [bs, n_visible] indices of visible spots
        """
        n_masked = int(n_spots * mask_ratio)
        n_visible = n_spots - n_masked
        
        # Generate random permutations for all batches at once
        noise = torch.rand(batch_size, n_spots, device=device)
        shuffled_indices = torch.argsort(noise, dim=1)
        
        # Split into masked and visible indices
        masked_indices = shuffled_indices[:, :n_masked]
        visible_indices = shuffled_indices[:, n_masked:]
        
        # Create boolean mask
        mask = torch.zeros(batch_size, n_spots, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_indices, True)
        
        return {
            'mask': mask,
            'masked_indices': masked_indices,
            'visible_indices': visible_indices
        }
    
    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_exp: torch.Tensor,
        coords: torch.Tensor,
        mask_info: Optional[Dict[str, torch.Tensor]] = None,
        anchors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            gene_ids: [bs, n_spots, n_genes] - gene IDs
            gene_exp: [bs, n_spots, n_genes] - gene expression values  
            coords: [bs, n_spots, 2] - spatial coordinates
            mask_info: Optional pre-computed mask info
        
        Returns:
            predictions: [bs, n_spots, n_genes] - predictions for all positions
            loss: scalar - reconstruction loss on masked positions
            mask_info: Dictionary with mask information
        """
        bs, n_spots, n_genes = gene_exp.shape
        
        # Generate mask if not provided
        if mask_info is None:
            mask_info = self.generate_random_mask(bs, n_spots, self.mask_ratio, gene_exp.device)
        
        encoded_features, mask_info = self.encoder(gene_ids, gene_exp, coords, mask_info)
        predictions, anchor_dist_pred = self.decoder(encoded_features, coords, mask_info)
        
        mask = mask_info['mask']
        if mask.any():
            pred_masked = predictions[mask]
            target_masked = gene_exp[mask]
            loss = F.mse_loss(pred_masked, target_masked)
        else:
            # When mask_ratio is 0, compute reconstruction loss on all positions
            # This makes the model work as a standard autoencoder
            loss = F.mse_loss(predictions, gene_exp)
        
        # Auxiliary: anchor distance loss on masked positions only
        if self.use_anchor_loss and anchors is not None and anchor_dist_pred is not None:
            # Normalize coordinates to [0,1] based on dataset scaling (~[0,100])
            bs = coords.shape[0]
            masked_indices = mask_info['masked_indices']  # [bs, n_mask]
            batch_idx = torch.arange(bs, device=coords.device).unsqueeze(1)
            # True distances: [bs, n_mask, K]
            masked_coords = coords[batch_idx, masked_indices]  # [bs, n_mask, 2]
            true_dist = torch.cdist(masked_coords, anchors, p=2)
            # Predicted distances at masked positions
            pred_dist = anchor_dist_pred[batch_idx, masked_indices]  # [bs, n_mask, K]
            anchor_loss = F.smooth_l1_loss(pred_dist, true_dist)
            loss = loss + self.anchor_loss_weight * anchor_loss
            return predictions, loss, mask_info, self.anchor_loss_weight * anchor_loss
        
        return predictions, loss, mask_info
    
    def encode(
        self,
        gene_ids: torch.Tensor,
        gene_exp: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Get encodings for all positions without masking (for downstream tasks)
        
        Args:
            gene_ids: [bs, n_spots, n_genes]
            gene_exp: [bs, n_spots, n_genes]
            coords: [bs, n_spots, 2]
        
        Returns:
            features: [bs, n_spots, d_model]
        """
        # Create a no-mask scenario (mask_ratio = 0)
        bs, n_spots, _ = gene_exp.shape
        mask_info = {
            'mask': torch.zeros(bs, n_spots, dtype=torch.bool, device=gene_exp.device),
            'masked_indices': torch.empty(bs, 0, dtype=torch.long, device=gene_exp.device),
            'visible_indices': torch.arange(n_spots, device=gene_exp.device).unsqueeze(0).expand(bs, -1)
        }
        
        # Encode all positions
        encoded_features, _ = self.encoder(gene_ids, gene_exp, coords, mask_info)
        
        return encoded_features
