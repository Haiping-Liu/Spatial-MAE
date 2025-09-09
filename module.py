import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple
from spatial_mae import SpatialMAE
from spatial_bert import SpatialBERT, compute_bert_loss
from config import Config


class SpatialMAELightning(pl.LightningModule):
    """PyTorch Lightning wrapper for Spatial MAE"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model
        self.model = SpatialMAE(
            vocab_size=config.model.vocab_size,
            n_genes=config.model.n_genes,
            d_model=config.model.d_model,
            d_decoder=config.model.d_decoder,
            n_encoder_layers=config.model.n_encoder_layers,
            n_decoder_layers=config.model.n_decoder_layers,
            n_heads=config.model.n_heads,
            dropout=config.model.dropout,
            mask_ratio=config.model.mask_ratio,
            padding_idx=config.model.padding_idx,
        )
    
    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return self.model(gene_ids, gene_values, coords)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        _, loss, _ = self.model(
            batch['gene_ids'], 
            batch['gene_values'], 
            batch['coords']
        )
        # Log with sync_dist=True for proper epoch-level aggregation
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        _, loss, _ = self.model(
            batch['gene_ids'], 
            batch['gene_values'], 
            batch['coords']
        )
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
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


class SpatialBERTLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for Spatial BERT"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model
        self.model = SpatialBERT(
            vocab_size=config.model.vocab_size,
            n_genes=config.model.n_genes,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            dropout=config.model.dropout,
            expr_mask_ratio=getattr(config.model, 'expr_mask_ratio', 0.1),
            pos_mask_ratio=getattr(config.model, 'pos_mask_ratio', 0.1),
            coord_noise_std=getattr(config.model, 'coord_noise_std', 10.0),
        )
        
        # Loss weights
        self.expr_weight = getattr(config.training, 'expr_weight', 1.0)
        self.coord_weight = getattr(config.training, 'coord_weight', 0.1)
    
    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        coords: torch.Tensor,
        expr_mask: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.model(gene_ids, gene_values, coords, expr_mask, pos_mask)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with dual-task loss"""
        predictions = self.model(
            batch['gene_ids'], 
            batch['gene_values'], 
            batch['coords']
        )
        
        # Compute dual-task loss
        losses = compute_bert_loss(
            predictions, 
            batch['gene_values'],
            expr_weight=self.expr_weight,
            coord_weight=self.coord_weight
        )
        
        # Log all losses
        self.log('train_loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_expr_loss', losses['expr_loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_coord_loss', losses['coord_loss'], on_step=True, on_epoch=True, sync_dist=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        predictions = self.model(
            batch['gene_ids'], 
            batch['gene_values'], 
            batch['coords']
        )
        
        # Compute dual-task loss
        losses = compute_bert_loss(
            predictions, 
            batch['gene_values'],
            expr_weight=self.expr_weight,
            coord_weight=self.coord_weight
        )
        
        # Log validation losses
        self.log('val_loss', losses['total_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_expr_loss', losses['expr_loss'], on_step=False, on_epoch=True)
        self.log('val_coord_loss', losses['coord_loss'], on_step=False, on_epoch=True)
        
        return losses['total_loss']
    
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
