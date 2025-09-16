"""
Main training script for HiGeST
"""
import argparse
from pathlib import Path
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from configs.config import Config
from module.higest_module import HiGeSTLightning
from data.dataset import MAESTDataset, DatasetPath, mae_collate_fn
from data.tokenizer import get_default_mae_tokenizer


class MAETrainer:    
    def __init__(self, config_path: str = None):
        self.config = Config.from_yaml(config_path) if config_path else Config()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.config.model.arch}_{timestamp}"
        self.checkpoint_dir = Path(self.config.logging.save_dir) / self.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        config_save_path = self.checkpoint_dir / "config.yaml"
        self.config.to_yaml(str(config_save_path))
        print(f"Config saved to: {config_save_path}")
        
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
        )

        self.config.model.n_genes = self.config.dataset.max_gene_len

        # Persist the global HVG order for evaluation reproducibility
        try:
            save_dir = Path(self.config.logging.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            if getattr(self.train_dataset, 'global_hvgs', None):
                hvgs = list(self.train_dataset.global_hvgs)
                # Save gene names
                (save_dir / 'global_hvgs.txt').write_text('\n'.join(hvgs))
                # Save token ids aligned to the same order
                hvgs_ids = self.train_dataset.tokenizer.encode_genes(hvgs)
                (save_dir / 'global_hvgs_ids.txt').write_text('\n'.join(map(str, hvgs_ids)))
        except Exception as e:
            # Non-fatal: continue training even if saving fails
            print(f"Warning: failed to persist global HVGs: {e}")
        
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
        self.model = HiGeSTLightning(self.config)
    
    def setup_callbacks(self):
        callbacks = [
            ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                filename='epoch{epoch:02d}-{' + self.config.logging.monitor_metric + ':.4f}',
                monitor=self.config.logging.monitor_metric,
                mode=self.config.logging.monitor_mode,
                save_top_k=self.config.logging.save_top_k,
                save_last=True
            )
        ]
        return callbacks
    
    def setup_logger(self):
        logger = TensorBoardLogger(
            save_dir=self.config.logging.log_dir,
            name=self.run_name,
            version="",  # Don't create version_0 subfolder
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
        print(f"Starting training...")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        print(f"Logs will be saved to: {self.config.logging.log_dir}/{self.run_name}")
        self.fit()
        print(f"Training completed!")
        print(f"Best checkpoint saved in: {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train HiGeST model")
    parser.add_argument('--config', type=str, default='configs/higest_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    trainer = MAETrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
