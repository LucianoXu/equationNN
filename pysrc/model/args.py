from .model import ModelArgs
from dataclasses import dataclass
from typing import Optional, Type

@dataclass
class TinyArgs(ModelArgs):
    '''
    ~2.5M parameters
    '''
    dim: int = 128
    n_layers: int = 10
    n_heads: int = 16
    n_kv_heads: int = 8
    d_ff: int = 512
    norm_eps: float = 1e-5
    rope_theta: float = 5000.0


@dataclass
class SmallArgs(ModelArgs):
    '''
    ~9M parameters
    '''
    dim: int = 256
    n_layers: int = 10
    n_heads: int = 16
    n_kv_heads: int = 8
    d_ff: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 5000.0

@dataclass
class MediumArgs(ModelArgs):
    '''
    ~35M parameters
    '''
    dim: int = 128 * 3
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 16
    d_ff: int = 128 * 3 * 4
    norm_eps: float = 1e-5
    rope_theta: float = 5000.0

@dataclass
class MediumLArgs(ModelArgs):
    '''
    ~20M parameters
    '''
    dim: int = 192
    n_layers: int = 36
    n_heads: int = 16
    n_kv_heads: int = 8
    d_ff: int = 192 * 4
    norm_eps: float = 1e-5
    rope_theta: float = 5000.0

@dataclass
class MediumXLArgs(ModelArgs):
    '''
    ~20M parameters
    '''
    dim: int = 144
    n_layers: int = 64
    n_heads: int = 16
    n_kv_heads: int = 8
    d_ff: int = 144 * 4
    norm_eps: float = 1e-5
    rope_theta: float = 5000.0

@dataclass
class LargeArgs(ModelArgs):
    '''
    ~126M parameters
    '''
    dim: int = 128 * 4
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 16
    d_ff: int = 128 * 4 * 4
    norm_eps: float = 1e-5
    rope_theta: float = 5000.0

modelargs_dict : dict[str, Type[ModelArgs]] = {
    'tiny': TinyArgs,

    'small': SmallArgs,
    
    'medium': MediumArgs,
    'mediumL' : MediumLArgs,
    'mediumXL' : MediumXLArgs,
    
    'large': LargeArgs,
}

