from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Hyperparameters container"""
    vocab_size: int = 9961
    dim_embedding: int = 1536
    dim_inner_layer: int = 3072
    n_head: int = 6
    n_layers: int = 8
    batch_size: int = 16
    block_size: int = 256
    epochs: int = 30
    dropout: float = 0.2
    bias: bool = True
    device: str = "cuda"
    lr: float = 6e-4
    weight_decay: float = 1e-1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 42