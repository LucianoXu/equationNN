from model import ModelArgs
from dataclasses import dataclass
from typing import Optional

@dataclass
class LargeArgs(ModelArgs):
    dim: int = 512
    n_layers: int = 32
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = 18
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_seq_len: int = 96