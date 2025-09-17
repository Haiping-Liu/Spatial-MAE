import torch
import torch.nn.functional as F
from typing import Dict, Optional


def topology_loss(
    student_cls: torch.Tensor,
    teacher_cls: torch.Tensor,
    adj_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute topology-aware loss over all pairs using BCE with logits,
    applying class-imbalance weighting and excluding self-pairs.
    
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

    # Compute similarity matrix [-1, 1], then scale as logits
    similarity = torch.bmm(student_norm, teacher_norm.transpose(1, 2))  # [B, N, N]
    logit_scale = 5.0
    logits = logit_scale * similarity

    
    # Exclude self-pairs (diagonal)
    B, N, _ = logits.shape
    device = logits.device
    diag = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    valid_mask = ~diag
    
    # Compute pos_weight to address class imbalance: pos_weight = #neg / #pos
    pos_mask = adj_matrix.bool() & valid_mask
    num_pos = pos_mask.sum().clamp_min(1)
    num_valid = valid_mask.sum()
    num_neg = (num_valid - num_pos).clamp_min(0)
    pos_weight_value = (num_neg.float() / num_pos.float()).clamp(1.0, 100.0)
    
    # BCE with logits over valid pairs only
    loss_elem = F.binary_cross_entropy_with_logits(
        logits, adj_matrix.float(), pos_weight=pos_weight_value, reduction='none'
    )
    topology_loss = loss_elem[valid_mask].mean()
    
    return topology_loss


def mask_loss(
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


def infonce_topology_loss(
    student_cls: torch.Tensor,
    teacher_cls: torch.Tensor,
    gene_similarity: torch.Tensor,
    spatial_knn: torch.Tensor,
    temperature: float = 0.07,
    n_bg_neg: int = 10,
) -> torch.Tensor:
    """
    InfoNCE loss with smart sampling based on gene similarity.

    Args:
        student_cls: [B, N, D] student representations
        teacher_cls: [B, N, D] teacher representations
        gene_similarity: [B, N, N] precomputed gene similarity matrix
        spatial_knn: [B, N, K] spatial k-nearest neighbor indices
        temperature: temperature for similarity scaling
        n_bg_neg: number of background negative samples

    Returns:
        InfoNCE loss value
    """
    B, N, D = student_cls.shape
    K = spatial_knn.shape[-1]
    device = student_cls.device

    # L2 normalize representations
    student_norm = F.normalize(student_cls, dim=-1)
    teacher_norm = F.normalize(teacher_cls, dim=-1)

    # ========== Select positive samples ==========
    # Get gene similarity between each anchor and its K neighbors
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)
    anchor_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)
    neighbor_gene_sim = gene_similarity[batch_idx, anchor_idx, spatial_knn]  # [B, N, K]

    # Handle invalid neighbors (-1)
    valid_mask = (spatial_knn >= 0)  # [B, N, K]
    neighbor_gene_sim = neighbor_gene_sim.masked_fill(~valid_mask, float('-inf'))

    # Select most similar neighbor as positive
    pos_idx_in_knn = neighbor_gene_sim.argmax(dim=-1)  # [B, N]
    pos_idx = torch.gather(spatial_knn, 2, pos_idx_in_knn.unsqueeze(-1)).squeeze(-1)  # [B, N]

    # ========== Select hard negative samples ==========
    # Select least similar neighbor (excluding positive)
    pos_mask_in_knn = F.one_hot(pos_idx_in_knn, K).bool()  # [B, N, K]
    hard_neg_gene_sim = neighbor_gene_sim.masked_fill(pos_mask_in_knn, float('inf'))

    hard_neg_idx_in_knn = hard_neg_gene_sim.argmin(dim=-1)  # [B, N]
    hard_neg_idx = torch.gather(spatial_knn, 2, hard_neg_idx_in_knn.unsqueeze(-1)).squeeze(-1)  # [B, N]

    # ========== Select background negative samples ==========
    # Create neighbor mask
    neighbor_mask = torch.zeros(B, N, N, dtype=torch.bool, device=device)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)
    anchor_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)
    valid_knn = torch.where(valid_mask, spatial_knn, torch.zeros_like(spatial_knn))
    neighbor_mask[batch_idx, anchor_idx, valid_knn] = valid_mask

    # Exclude self
    self_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    neighbor_mask = neighbor_mask | self_mask

    # Background samples = non-neighbors
    bg_gene_sim = gene_similarity.masked_fill(neighbor_mask, float('inf'))

    # Select least similar non-neighbors
    n_bg = min(n_bg_neg, N - K - 1)
    if n_bg > 0:
        _, bg_neg_indices = torch.topk(bg_gene_sim, k=n_bg, dim=-1, largest=False)  # [B, N, n_bg]
    else:
        bg_neg_indices = torch.randint(0, N, (B, N, 1), device=device)

    # ========== Compute similarities ==========
    # Compute all pairwise similarities
    all_sim = torch.bmm(student_norm, teacher_norm.transpose(1, 2)) / temperature  # [B, N, N]

    # Extract required similarities
    batch_range = torch.arange(B, device=device).view(B, 1).expand(B, N)
    anchor_range = torch.arange(N, device=device).view(1, N).expand(B, N)

    # Positive similarity
    pos_sim = all_sim[batch_range, anchor_range, pos_idx]  # [B, N]

    # Hard negative similarity
    hard_neg_sim = all_sim[batch_range, anchor_range, hard_neg_idx]  # [B, N]

    # Background negative similarities
    batch_idx_bg = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, n_bg)
    anchor_idx_bg = torch.arange(N, device=device).view(1, N, 1).expand(B, N, n_bg)
    bg_neg_sims = all_sim[batch_idx_bg, anchor_idx_bg, bg_neg_indices]  # [B, N, n_bg]

    # ========== InfoNCE Loss ==========
    # Concatenate all similarities
    all_sims = torch.cat([
        pos_sim.unsqueeze(-1),      # [B, N, 1]
        hard_neg_sim.unsqueeze(-1),  # [B, N, 1]
        bg_neg_sims                   # [B, N, n_bg]
    ], dim=-1)  # [B, N, 2+n_bg]

    # Compute log-sum-exp for numerical stability
    log_sum_exp = torch.logsumexp(all_sims, dim=-1)  # [B, N]

    # InfoNCE loss
    loss = -pos_sim + log_sum_exp  # [B, N]

    # Handle invalid anchors (no valid neighbors)
    valid_anchors = valid_mask.any(dim=-1)  # [B, N]
    loss = loss.masked_fill(~valid_anchors, 0.0)

    # Average over valid anchors
    n_valid = valid_anchors.sum()
    if n_valid > 0:
        return loss.sum() / n_valid
    else:
        return torch.tensor(0.0, device=device)
