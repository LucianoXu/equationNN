from model import ModelArgs
from dataclasses import dataclass
from typing import Optional

from tokenizer import token2id

@dataclass
class SmallArgs(ModelArgs):
    dim: int = 256
    n_layers: int = 10
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = len(token2id)
    d_ff: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 5000.0

    context_length: int = 160