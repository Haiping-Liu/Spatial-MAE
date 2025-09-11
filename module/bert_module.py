import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
import copy

from model.spatial_bert import SpatialBERT, compute_bert_loss, build_adjacency_matrix, compute_topology_loss
from configs.config import Config


class SpatialBERTLightning(pl.LightningModule):    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize student model
        self.model = SpatialBERT(
            vocab_size=config.model.vocab_size,
            n_genes=config.model.n_genes,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            dropout=config.model.dropout,
            gene_mask_ratio=getattr(config.model, 'gene_mask_ratio', 0.15),
            noise_scale=getattr(config.model, 'noise_scale', 0.1),
        )
        
        # Initialize teacher model (EMA)
        self.teacher = copy.deepcopy(self.model)
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Topology learning parameters
        self.k_neighbors = getattr(config.model, 'k_neighbors', 6)
        self.ema_decay = getattr(config.model, 'ema_decay', 0.99)
        
        # Learnable sigmoid gate for topology loss
        self.topo_gate_logit = torch.nn.Parameter(torch.tensor(-4.0)) # Initialize with a small gate value
    
    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        coords: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.model(gene_ids, gene_values, coords)
    
    def update_teacher(self):
        """Update teacher model with EMA"""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.model.parameters()):
                teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with gene-level masking and topology learning"""
        gene_ids = batch['gene_ids']
        gene_values = batch['gene_values']
        coords = batch['coords']
        
        # Student forward (with masking)
        student_out = self.model(gene_ids, gene_values, coords, apply_mask=True)
        
        # Teacher forward (no masking)
        with torch.no_grad():
            teacher_out = self.teacher(gene_ids, gene_values, coords, apply_mask=False)
        
        # Compute losses
        recon_losses = compute_bert_loss(student_out)
        recon_loss = recon_losses['loss']
        
        # Build adjacency matrix dynamically
        adj_matrix = build_adjacency_matrix(coords, k=self.k_neighbors)
        
        # Compute topology loss
        topo_loss = compute_topology_loss(
            student_out['cls_tokens'],
            teacher_out['cls_tokens'].detach(),
            adj_matrix
        )
        
        # Gated total loss with a learnable sigmoid gate
        topo_gate = torch.sigmoid(self.topo_gate_logit)
        total_loss = recon_loss + topo_gate * topo_loss
        
        # Update EMA decay (increase over time)
        current_decay = min(self.ema_decay + 0.00005 * self.current_epoch, 0.9995)
        self.ema_decay = current_decay
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('topo_gate', topo_gate, on_step=True, on_epoch=False)
        self.log('ema_decay', self.ema_decay, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with topology loss"""
        gene_ids = batch['gene_ids']
        gene_values = batch['gene_values']
        coords = batch['coords']
        
        # Student forward (with masking even in val for consistency)
        student_out = self.model(gene_ids, gene_values, coords, apply_mask=True)
        
        # Teacher forward (no masking)
        with torch.no_grad():
            teacher_out = self.teacher(gene_ids, gene_values, coords, apply_mask=False)
        
        # Compute losses
        recon_losses = compute_bert_loss(student_out)
        recon_loss = recon_losses['loss']
        
        # Build adjacency matrix
        adj_matrix = build_adjacency_matrix(coords, k=self.k_neighbors)
        
        # Compute topology loss
        topo_loss = compute_topology_loss(
            student_out['cls_tokens'],
            teacher_out['cls_tokens'],
            adj_matrix
        )
        
        # Total loss (use full weight in validation)
        topo_gate = torch.sigmoid(self.topo_gate_logit)
        total_loss = recon_loss + topo_gate * topo_loss
        
        # Logging
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher model after optimizer step"""
        self.update_teacher()
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_eps,
            weight_decay=self.config.training.weight_decay
        )
    
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.t_max,
            eta_min=self.config.training.min_lr
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
