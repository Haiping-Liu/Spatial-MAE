import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
import copy

from model.higest import HiGeST, compute_bert_loss, build_adjacency_matrix, compute_topology_loss
from configs.config import Config


class HiGeSTLightning(pl.LightningModule):    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Initialize student model
        self.model = HiGeST(
            vocab_size=config.model.vocab_size,
            n_genes=config.model.n_genes,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            dropout=config.model.dropout,
        )
        
        # Initialize teacher model (EMA)
        self.teacher = copy.deepcopy(self.model)
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Topology learning parameters (fixed weights)
        self.k_neighbors = getattr(config.model, 'k_neighbors', 6)
        self.lambda_topo = getattr(config.model, 'lambda_topo', 0.3)
        self.ema_decay = getattr(config.model, 'ema_decay', 0.995)
    
    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        coords: torch.Tensor,
        masked_values: torch.Tensor = None,
        gene_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if masked_values is None:
            masked_values = gene_values
        return self.model(gene_ids, gene_values, masked_values, coords, gene_mask)
    
    def update_teacher(self):
        """Update teacher model with EMA"""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.model.parameters()):
                teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with gene-level masking and topology learning"""
        gene_ids = batch['gene_ids']
        gene_values = batch['gene_values']
        masked_values = batch['masked_values']
        coords = batch['coords']
        gene_mask = batch['gene_mask']
        
        # Student forward (with masked values)
        student_out = self.model(gene_ids, gene_values, masked_values, coords, gene_mask)
        
        # Teacher forward (with original values)
        # Note: Teacher uses projector but not predictor
        with torch.no_grad():
            self.teacher.eval()  # Ensure teacher is in eval mode for BatchNorm
            teacher_out = self.teacher(gene_ids, gene_values, gene_values, coords, use_predictor=False)
        
        # Compute losses
        recon_losses = compute_bert_loss(student_out)
        recon_loss = recon_losses['loss']
        
        # Build adjacency matrix dynamically
        adj_matrix = build_adjacency_matrix(coords, k=self.k_neighbors)
        
        # Compute topology loss using BYOL-style prediction
        # Student predictor output tries to predict teacher projector output
        topo_loss = compute_topology_loss(
            student_out['cls_predicted'],  # Student: backbone → projector → predictor
            teacher_out['cls_projected'].detach(),  # Teacher: backbone → projector (stop grad)
            adj_matrix
        )
        
        # Total loss with fixed topology weight
        total_loss = recon_loss + self.lambda_topo * topo_loss
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('ema_decay', self.ema_decay, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with topology loss"""
        gene_ids = batch['gene_ids']
        gene_values = batch['gene_values']
        masked_values = batch['masked_values']
        coords = batch['coords']
        gene_mask = batch['gene_mask']
        
        # Student forward (with masked values)
        student_out = self.model(gene_ids, gene_values, masked_values, coords, gene_mask)
        
        # Teacher forward (with original values)
        # Note: Teacher uses projector but not predictor
        with torch.no_grad():
            self.teacher.eval()  # Ensure teacher is in eval mode for BatchNorm
            teacher_out = self.teacher(gene_ids, gene_values, gene_values, coords, use_predictor=False)
        
        # Compute losses
        recon_losses = compute_bert_loss(student_out)
        recon_loss = recon_losses['loss']
        
        # Build adjacency matrix
        adj_matrix = build_adjacency_matrix(coords, k=self.k_neighbors)
        
        # Compute topology loss using BYOL-style prediction (same as training)
        topo_loss = compute_topology_loss(
            student_out['cls_predicted'],  # Student: backbone → projector → predictor
            teacher_out['cls_projected'].detach(),  # Teacher: backbone → projector (stop grad)
            adj_matrix
        )
        
        # Optionally compute topology loss with original cls_tokens for comparison
        topo_loss_cls = compute_topology_loss(
            student_out['cls_tokens'],
            teacher_out['cls_tokens'].detach(),
            adj_matrix
        )
        
        # Total loss (use same fixed weight in validation)
        total_loss = recon_loss + self.lambda_topo * topo_loss
        
        # Logging
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_topo_loss_cls', topo_loss_cls, on_step=False, on_epoch=True)  # Optional metric for comparison
        
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
