import torch
import torch.nn.functional as F
from typing import Dict


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
