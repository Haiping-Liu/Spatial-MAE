import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math
from typing import Dict
import copy

from model.higest import HiGeST, build_adjacency_matrix
from model.loss import mask_loss, topology_loss
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
            # Use batch statistics for BatchNorm while keeping Dropout disabled
            self.teacher.eval()
            for m in self.teacher.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.train()
                elif isinstance(m, torch.nn.Dropout):
                    m.eval()
            teacher_out = self.teacher(gene_ids, gene_values, gene_values, coords, use_predictor=False)
        
        # Compute losses
        recon_losses = mask_loss(student_out)
        recon_loss = recon_losses['loss']
        
        # Build adjacency matrix dynamically
        adj_matrix = build_adjacency_matrix(coords, k=self.k_neighbors)
        
        # Compute topology loss using BYOL-style prediction
        topo_loss = topology_loss(
            student_out['cls_predicted'],  # Student: backbone → projector → predictor
            teacher_out['cls_projected'].detach(),  # Teacher: backbone → projector (stop grad)
            adj_matrix
        )
        
        # Total loss with fixed topology weight
        total_loss = recon_loss + self.lambda_topo * topo_loss
        
        # Logging
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
            # Use batch statistics for BatchNorm while keeping Dropout disabled
            self.teacher.eval()
            for m in self.teacher.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.train()
                elif isinstance(m, torch.nn.Dropout):
                    m.eval()
            teacher_out = self.teacher(gene_ids, gene_values, gene_values, coords, use_predictor=False)
        
        # Compute losses
        recon_losses = mask_loss(student_out)
        recon_loss = recon_losses['loss']
        
        # Build adjacency matrix
        adj_matrix = build_adjacency_matrix(coords, k=self.k_neighbors)
        
        # Compute topology loss using BYOL-style prediction (same as training)
        topo_loss = topology_loss(
            student_out['cls_predicted'],  # Student: backbone → projector → predictor
            teacher_out['cls_projected'].detach(),  # Teacher: backbone → projector (stop grad)
            adj_matrix
        )

        total_loss = recon_loss + self.lambda_topo * topo_loss
        
        # Logging
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher model after optimizer step"""
        self.update_teacher()
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers with warmup + cosine annealing"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_eps,
            weight_decay=self.config.training.weight_decay
        )

        # Get total training steps from trainer
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps // self.config.training.max_epochs

        # Get warmup epochs and convert to steps (default 10 epochs warmup)
        warmup_epochs = getattr(self.config.training, 'warmup_epochs', 10)
        warmup_steps = warmup_epochs * steps_per_epoch
        min_lr_ratio = self.config.training.min_lr / self.config.training.learning_rate

        print(f"LR Schedule: warmup {warmup_epochs} epochs ({warmup_steps} steps), total {self.config.training.max_epochs} epochs ({total_steps} steps)")

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing after warmup
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
