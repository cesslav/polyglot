from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """With a dataclass we don't need to write an init function, just specify class attributes and their types"""

    vocab_size: int = 9961
    dim_embedding: int = 1536
    dim_inner_layer: int =  3072
    n_head: int = 6
    n_layers: int = 8
    batch_size: int = 16
    block_size: int = 256
    epochs: int = 30
    dropout: float = 0.2
    bias: bool = True
    device: str = "cuda"
