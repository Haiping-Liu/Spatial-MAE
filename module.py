import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple
from spatial_mae import SpatialMAE
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
