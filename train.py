"""
Main training script for Spatial MAE
"""
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from config import Config
from module import SpatialMAELightning
from mae_dataset import MAESTDataset, DatasetPath, mae_collate_fn
from mae_tokenizer import get_default_mae_tokenizer


class MAETrainer:    
    def __init__(self, config_path: str = None):
        self.config = Config.from_yaml(config_path) if config_path else Config()
        self.setup_seed()
        self.setup_datasets()
        self.setup_model()
        self.setup_trainer()
    
    def setup_seed(self):
        """Set random seed"""
        pl.seed_everything(self.config.seed, workers=True)
    
    def setup_datasets(self):
        tokenizer = get_default_mae_tokenizer()
        self.config.model.vocab_size = len(tokenizer)
        self.config.model.padding_idx = tokenizer.pad_token_id
        
        # Build dataset paths: scan a directory if provided; otherwise use explicit list
        dataset_paths = []
        if getattr(self.config.dataset, 'data_dir', None):
            root = Path(self.config.dataset.data_dir)
            h5ad_files = sorted(root.glob('*.h5ad'))
            for i, p in enumerate(h5ad_files, start=1):
                dataset_paths.append(DatasetPath(name=f"dataset_{i}", source="DLPFC", h5ad_path=str(p)))
        else:
            dataset_paths = [DatasetPath(**ds) for ds in self.config.dataset.dataset_list]
        
        # Split: all but last for train, last for val (if only one, both use the same)
        train_paths = dataset_paths[:-1] if len(dataset_paths) > 1 else dataset_paths
        val_paths = dataset_paths[-1:] if len(dataset_paths) > 1 else dataset_paths
        
        self.train_dataset = MAESTDataset(
            dataset_list=train_paths,
            tokenizer=tokenizer,
            n_spots=self.config.dataset.n_spots,
            n_hvg=self.config.dataset.n_hvg,
            use_hvg=self.config.dataset.use_hvg,
            sampling_method=self.config.dataset.sampling_method,
            normalize_total=self.config.dataset.normalize_total,
            log1p=self.config.dataset.log1p,
            max_gene_len=self.config.dataset.max_gene_len,
            patches_per_slide=self.config.dataset.patches_per_slide,
            anchor_k=self.config.dataset.anchor_k,
            anchor_method=self.config.dataset.anchor_method,
        )
        
        self.val_dataset = MAESTDataset(
            dataset_list=val_paths,
            tokenizer=tokenizer,
            n_spots=self.config.dataset.n_spots,
            n_hvg=self.config.dataset.n_hvg,
            use_hvg=self.config.dataset.use_hvg,
            sampling_method=self.config.dataset.sampling_method,
            normalize_total=self.config.dataset.normalize_total,
            log1p=self.config.dataset.log1p,
            max_gene_len=self.config.dataset.max_gene_len,
            patches_per_slide=self.config.dataset.patches_per_slide,
            anchor_k=self.config.dataset.anchor_k,
            anchor_method=self.config.dataset.anchor_method,
        )
        
        self.config.model.n_genes = self.config.dataset.max_gene_len
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.num_workers,
            drop_last=True,
            collate_fn=mae_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            collate_fn=mae_collate_fn
        )
    
    def setup_model(self):
        self.model = SpatialMAELightning(self.config)
    
    def setup_callbacks(self):
        callbacks = [
            ModelCheckpoint(
                dirpath=Path(self.config.logging.save_dir),
                filename='{epoch:02d}-{' + self.config.logging.monitor_metric + ':.4f}',
                monitor=self.config.logging.monitor_metric,
                mode=self.config.logging.monitor_mode,
                save_top_k=self.config.logging.save_top_k,
                save_last=True
            ),
            EarlyStopping(
                monitor=self.config.logging.monitor_metric,
                mode=self.config.logging.monitor_mode,
                patience=self.config.training.patience,
                min_delta=self.config.training.min_delta
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        return callbacks
    
    def setup_logger(self):     
        logger = WandbLogger(
            project=self.config.logging.project_name,
            save_dir=self.config.logging.log_dir,
            name=self.config.logging.run_name,
            offline=True
        )
        return logger
    
    def setup_trainer(self):
        # Setup callbacks and logger
        callbacks = self.setup_callbacks()
        logger = self.setup_logger()
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            accelerator=self.config.compute.accelerator,
            devices=self.config.compute.devices,
            precision=self.config.compute.precision,
            gradient_clip_val=self.config.training.gradient_clip_val,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=self.config.logging.log_every_n_steps,
            val_check_interval=self.config.logging.val_check_interval,
            deterministic=True
        )
    
    def fit(self):
        """Train the model"""
        self.trainer.fit(
            self.model, 
            self.train_loader, 
            self.val_loader
        )
    
    def run(self):
        print("Starting training...")
        self.fit()
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Spatial MAE")
    parser.add_argument('--config', type=str, default='configs/antioverfit.yaml', help='Path to config file')
    args = parser.parse_args()
    
    trainer = MAETrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
