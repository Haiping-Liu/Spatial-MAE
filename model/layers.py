import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union

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
        cls_token: torch.Tensor,
        gene_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gene_tokens: [bs*n_spots, n_genes, d_model]
            cls_token: [bs*n_spots, 1, d_model]
            gene_padding_mask: [bs*n_spots, n_genes] - True for padding positions
        Returns:
            gene_tokens: [bs*n_spots, n_genes, d_model]
            cls_features: [bs*n_spots, d_model]
        """
        # Concatenate CLS with gene tokens
        tokens = torch.cat([cls_token, gene_tokens], dim=1)  # [bs*n_spots, n_genes+1, d_model]
        
        # Create attention mask if gene padding mask is provided
        key_padding_mask = None
        if gene_padding_mask is not None:
            # Create mask for [CLS + genes]: CLS is never masked, genes can be masked
            key_padding_mask = torch.zeros(
                tokens.shape[0], tokens.shape[1], 
                dtype=torch.bool, device=tokens.device
            )
            key_padding_mask[:, 1:] = gene_padding_mask  # Skip CLS (position 0)
        
        # Self-attention
        attn_out, _ = self.attention(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
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


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """
    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: torch.Tensor, gene_embs: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)
            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)
            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)
