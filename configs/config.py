"""
Configuration management using dataclasses
"""
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from pathlib import Path
import yaml
from omegaconf import OmegaConf, DictConfig


@dataclass
class ModelConfig:
    """Model architecture configuration for Spatial BERT"""
    arch: str = 'bert'       # Model architecture type
    vocab_size: int = 60000  # Vocabulary size for gene embeddings
    n_genes: int = 500       # Number of genes per sample
    d_model: int = 96        # Model dimension
    n_layers: int = 3        # Number of transformer layers
    n_heads: int = 3         # Number of attention heads
    dropout: float = 0.1     # Dropout rate
    
    # Gene-level masking parameters
    gene_mask_ratio: float = 0.15  # Mask 15% of valid genes per spot
    noise_scale: float = 0.1       # Noise scale for masked genes (10% strategy)
    
    # Topology learning parameters
    k_neighbors: int = 6            # Number of nearest neighbors for adjacency matrix
    ema_decay: float = 0.99         # Initial EMA decay (will increase to 0.9995)
    
    # Token parameters
    max_value: int = 512
    padding_idx: int = 0

@dataclass
class DatasetConfig:
    """Dataset configuration"""
    data_dir: Optional[str] = None
    dataset_list: List[Dict[str, str]] = field(default_factory=list)
    n_spots: int = 128
    n_hvg: int = 1000
    use_hvg: bool = True
    sampling_method: str = 'nearest'
    normalize_total: float = 1e4
    log1p: bool = True
    max_gene_len: int = 1000
    patches_per_slide: int = 100  # Number of patches to sample from each slide per epoch
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class TokenizerConfig:
    """Tokenizer configuration"""
    vocab_file: Optional[str] = None
    default_vocab_type: str = 'census'  # 'census' or 'standard'
    special_tokens: List[str] = field(default_factory=lambda: ["<pad>", "<unk>", "<mask>", "<cls>", "<eos>"])
    default_token: str = "<pad>"


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Optimizer
    optimizer: str = 'adamw'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    
    # Scheduler
    scheduler: str = 'cosine'
    min_lr: float = 1e-6
    t_max: int = 10000
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Loss (removed, using learnable gate in model)
    


@dataclass
class LoggingConfig:
    """Logging configuration"""
    project_name: str = 'spatial-mae'
    run_name: Optional[str] = None
    logger: str = 'wandb'  # 'wandb' or 'tensorboard'
    log_dir: str = './logs'
    save_dir: str = './checkpoints'
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    save_top_k: int = 1
    monitor_metric: str = 'val_loss'
    monitor_mode: str = 'min'


@dataclass
class ComputeConfig:
    """Compute configuration"""
    accelerator: str = 'gpu'
    devices: Any = 1  # Can be int or list
    precision: str = '16-mixed'  # '32', '16-mixed', 'bf16-mixed'
    strategy: str = 'auto'  # 'auto', 'ddp', 'dp'
    gradient_checkpointing: bool = False
    compile_model: bool = False
    detect_anomaly: bool = False
    
    # Memory optimization
    enable_cpu_offload: bool = False
    empty_cache_freq: int = 100


@dataclass
class Config:
    """Main configuration aggregating all configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    
    # Runtime
    seed: int = 42
    debug: bool = False
    resume_from_checkpoint: Optional[str] = None
    test_only: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        
        # Use OmegaConf for nested dataclass instantiation
        schema = OmegaConf.structured(cls)
        config = OmegaConf.merge(schema, OmegaConf.create(yaml_dict))
        
        return OmegaConf.to_object(config)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = OmegaConf.create(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(config_dict), f, default_flow_style=False)
    
    def update_from_dict(self, update_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        config_omega = OmegaConf.struct(self)
        updated = OmegaConf.merge(config_omega, OmegaConf.create(update_dict))
        return OmegaConf.to_object(updated)
    
    def print_config(self):
        """Pretty print configuration"""
        print(OmegaConf.to_yaml(self))


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
