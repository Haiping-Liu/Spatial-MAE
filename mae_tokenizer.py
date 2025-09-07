"""
MAE-specific tokenizer for spatial transcriptomics
Simplified version of scGPT tokenizer focused on MAE requirements
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from collections import OrderedDict, Counter


class MAEGeneVocab:
    """
    Gene vocabulary for MAE tasks
    Simplified from scGPT's GeneVocab, focusing on core functionality
    """
    
    def __init__(
        self,
        gene_list_or_dict: Union[List[str], Dict[str, int]] = None,
        special_tokens: List[str] = None,
        default_token: str = "<pad>",
    ):
        """
        Initialize gene vocabulary
        
        Args:
            gene_list_or_dict: List of genes or dict of gene->id mappings
            special_tokens: Special tokens to add at the beginning
            default_token: Default token for OOV genes
        """
        # Default special tokens for MAE
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<mask>", "<cls>", "<eos>"]
        
        self.special_tokens = special_tokens
        self.vocab = OrderedDict()
        self.itos = {}  # index to string
        
        # Add special tokens first
        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx
            self.itos[idx] = token
        
        # Add genes
        if isinstance(gene_list_or_dict, dict):
            # Load from existing vocabulary
            for gene, idx in sorted(gene_list_or_dict.items(), key=lambda x: x[1]):
                if gene not in self.vocab:  # Don't overwrite special tokens
                    self.vocab[gene] = idx
                    self.itos[idx] = gene
        elif isinstance(gene_list_or_dict, list):
            # Build from gene list
            start_idx = len(special_tokens)
            for idx, gene in enumerate(sorted(set(gene_list_or_dict))):
                if gene not in self.vocab:
                    self.vocab[gene] = start_idx + idx
                    self.itos[start_idx + idx] = gene
        
        # Set default token
        self.default_token = default_token
        self.default_index = self.vocab.get(default_token, 0)
    
    @classmethod
    def from_file(cls, file_path: Union[Path, str], special_tokens: List[str] = None):
        """Load vocabulary from JSON file"""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        with file_path.open("r") as f:
            gene_dict = json.load(f)
        
        return cls(gene_dict, special_tokens=special_tokens)
    
    def save_json(self, file_path: Union[Path, str]):
        """Save vocabulary to JSON file"""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        with file_path.open("w") as f:
            json.dump(dict(self.vocab), f, indent=2)
    
    def __getitem__(self, key: Union[str, int]) -> Union[int, str]:
        """Get token ID for gene or gene for token ID"""
        if isinstance(key, str):
            return self.vocab.get(key, self.default_index)
        elif isinstance(key, int):
            return self.itos.get(key, self.default_token)
        else:
            raise KeyError(f"Invalid key type: {type(key)}")
    
    def __contains__(self, key: str) -> bool:
        """Check if gene is in vocabulary"""
        return key in self.vocab
    
    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    def get_stoi(self) -> Dict[str, int]:
        """Get string-to-index dictionary"""
        return dict(self.vocab)
    
    def get_itos(self) -> Dict[int, str]:
        """Get index-to-string dictionary"""
        return dict(self.itos)
    
    @property
    def pad_token_id(self) -> int:
        return self.vocab.get("<pad>", 0)
    
    @property
    def unk_token_id(self) -> int:
        return self.vocab.get("<unk>", 1)
    
    @property
    def mask_token_id(self) -> int:
        return self.vocab.get("<mask>", 2)
    
    @property
    def cls_token_id(self) -> int:
        return self.vocab.get("<cls>", 3)
    
    @property
    def eos_token_id(self) -> int:
        return self.vocab.get("<eos>", 4)


class MAETokenizer:
    """
    Tokenizer for MAE on spatial transcriptomics data
    
    Key features:
    - Uses pre-built scGPT vocabulary
    - No binning (MAE works with continuous values)
    - Simplified API focused on MAE requirements
    - Efficient gene filtering and HVG selection
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        special_tokens: List[str] = None,
        max_seq_len: int = 2000,
        pad_value: float = 0.0,
        mask_value: float = -1.0,
    ):
        """
        Initialize MAE tokenizer
        
        Args:
            vocab_file: Path to vocabulary JSON file
            special_tokens: List of special tokens
            max_seq_len: Maximum sequence length
            pad_value: Value for padding
            mask_value: Value for masked positions
        """
        # Load vocabulary
        if vocab_file is None:
            vocab_file = Path(__file__).parent / "scgpt_tokenizer" / "default_gene_vocab.json"
        
        self.gene_vocab = MAEGeneVocab.from_file(vocab_file, special_tokens=special_tokens)
        self.vocab_size = len(self.gene_vocab)
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.mask_value = mask_value
        
        # Quick access to special token IDs
        self.pad_token_id = self.gene_vocab.pad_token_id
        self.unk_token_id = self.gene_vocab.unk_token_id
        self.mask_token_id = self.gene_vocab.mask_token_id
        self.cls_token_id = self.gene_vocab.cls_token_id
        self.eos_token_id = self.gene_vocab.eos_token_id
        
        print(f"Loaded vocabulary with {self.vocab_size} tokens")
        print(f"Special tokens: pad={self.pad_token_id}, unk={self.unk_token_id}, "
              f"mask={self.mask_token_id}, cls={self.cls_token_id}, eos={self.eos_token_id}")
    
    def filter_genes_by_vocab(self, gene_names: List[str]) -> Tuple[List[str], List[int]]:
        """
        Filter genes to only those in vocabulary
        
        Args:
            gene_names: List of gene names
            
        Returns:
            filtered_names: List of genes in vocabulary
            gene_ids: Corresponding token IDs
        """
        filtered_names = []
        gene_ids = []
        
        for gene in gene_names:
            if gene in self.gene_vocab:
                filtered_names.append(gene)
                gene_ids.append(self.gene_vocab[gene])
        
        return filtered_names, gene_ids
    
    def encode_genes(self, gene_names: List[str]) -> torch.Tensor:
        """
        Encode gene names to token IDs
        
        Args:
            gene_names: List of gene names
            
        Returns:
            Tensor of token IDs
        """
        gene_ids = [self.gene_vocab[gene] for gene in gene_names]
        return torch.tensor(gene_ids, dtype=torch.long)
    
    def decode_genes(self, gene_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs to gene names
        
        Args:
            gene_ids: Tensor of token IDs
            
        Returns:
            List of gene names
        """
        if gene_ids.dim() == 0:
            return [self.gene_vocab[int(gene_ids)]]
        return [self.gene_vocab[int(idx)] for idx in gene_ids]
    
    def pad_sequence(
        self,
        gene_ids: torch.Tensor,
        gene_exp: torch.Tensor,
        target_length: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad gene IDs and expression values to target length
        
        Args:
            gene_ids: Tensor of gene IDs [n_genes]
            gene_exp: Tensor of expression values [n_spots, n_genes]
            target_length: Target sequence length (default: max_seq_len)
            
        Returns:
            padded_ids: Padded gene IDs [target_length]
            padded_exp: Padded expression [n_spots, target_length]
        """
        if target_length is None:
            target_length = self.max_seq_len
        
        current_len = gene_ids.shape[0]
        
        if current_len >= target_length:
            # Truncate if too long
            return gene_ids[:target_length], gene_exp[:, :target_length]
        
        # Pad if too short
        pad_size = target_length - current_len
        
        padded_ids = torch.cat([
            gene_ids,
            torch.full((pad_size,), self.pad_token_id, dtype=gene_ids.dtype)
        ])
        
        padded_exp = torch.nn.functional.pad(
            gene_exp,
            (0, pad_size),
            value=self.pad_value
        )
        
        return padded_ids, padded_exp
    
    def create_mae_batch(
        self,
        gene_exp: torch.Tensor,
        gene_ids: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Dict[str, torch.Tensor]:
        """
        Create a batch for MAE training
        
        Args:
            gene_exp: Expression matrix [batch_size, n_spots, n_genes]
            gene_ids: Gene IDs [n_genes]
            mask_ratio: Ratio of genes to mask
            
        Returns:
            Dictionary with masked inputs and targets
        """
        batch_size, n_spots, n_genes = gene_exp.shape
        
        # Create random mask
        n_mask = int(n_genes * mask_ratio)
        
        batch_data = {
            'gene_exp': gene_exp,
            'gene_ids': gene_ids,
            'mask_ratio': mask_ratio,
        }
        
        # For each sample in batch, create different random masks
        masks = []
        for i in range(batch_size):
            mask_idx = torch.randperm(n_genes)[:n_mask]
            mask = torch.zeros(n_genes, dtype=torch.bool)
            mask[mask_idx] = True
            masks.append(mask)
        
        batch_data['masks'] = torch.stack(masks)
        
        return batch_data
    
    def __len__(self) -> int:
        return len(self.gene_vocab)
    
    def __getitem__(self, key: Union[str, int]) -> Union[int, str]:
        return self.gene_vocab[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.gene_vocab
    
    def get_stoi(self) -> Dict[str, int]:
        """Get string-to-index dictionary for compatibility"""
        return self.gene_vocab.get_stoi()
    
    def get_itos(self) -> Dict[int, str]:
        """Get index-to-string dictionary"""
        return self.gene_vocab.get_itos()


def get_default_mae_tokenizer() -> MAETokenizer:
    """Get default MAE tokenizer with scGPT vocabulary"""
    return MAETokenizer()


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = get_default_mae_tokenizer()
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Test gene encoding
    test_genes = ["GAPDH", "ACTB", "TP53", "UNKNOWN_GENE"]
    for gene in test_genes:
        if gene in tokenizer:
            gene_id = tokenizer[gene]
            decoded = tokenizer[gene_id]
            print(f"{gene} -> {gene_id} -> {decoded}")
        else:
            print(f"{gene} not in vocabulary, will use UNK token")