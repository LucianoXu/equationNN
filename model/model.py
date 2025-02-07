import math
from dataclasses import dataclass
from typing import Optional, Tuple

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group, destroy_process_group

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 32
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 0
    d_ff: int = 2048
    norm_eps: float = 1e-5
    rope_theta: float = 5000.0

    context_length: int = 160

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device: str = 'cpu'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device = 'cpu'):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device = device)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freq_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freq_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # slice and reshape freqs_cis
    ndim = xq_.ndim
    assert 0 <= 1 < ndim, "xq, xk must have at least 2 dimensions"
    assert freqs_cis.shape == (xq_.shape[1], xq_.shape[-1]), "freqs_cis shape mismatch"
    freqs_cis_shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*freqs_cis_shape)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, device: str = 'cpu'):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            device=device
        )

    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        device = 'cpu'
    ):
        super().__init__()

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False, device = device
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False, device = device
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False, device = device
        )
    
    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, device: str = 'cpu'):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        self.attention = Attention(args, device)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.d_ff,
            device = device
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, device = device)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, device = device)

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

def mask_calc(batch_size: int, seq_len: int, device: torch.device,
            front_pad_lengths: Optional[torch.Tensor|list[int]] = None, 
            input_lengths: Optional[torch.Tensor|list[int]] = None) -> torch.Tensor:
        
        # Create causal mask
        # Shape: (seqlen, seqlen)
        # example: 
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        causal_mask = ~ torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # Create pad attention mask
        # Shape: (batch_size, seqlen, seqlen)
        # example (one sequence):
        # 0 0 0 0
        # 0 1 1 1
        # 0 1 1 1
        # 0 1 1 1
        if front_pad_lengths is None:
            front_pad_lengths = torch.zeros(batch_size, device=device)
        else:
            front_pad_lengths = torch.tensor(front_pad_lengths, device=device)
        pad_mask = torch.arange(seq_len, device=device) >= front_pad_lengths.unsqueeze(1)
        pad_attn_mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)


        # Create input attention mask
        # Shape: (batch_size, seqlen, seqlen)
        # example (one sequence): 
        # 1 1 1 0
        # 1 1 1 0
        # 1 1 1 0
        # 0 0 0 0
        if input_lengths is None:
            input_lengths = torch.zeros(batch_size, device=device)
        else:
            input_lengths = torch.tensor(input_lengths, device=device)
        input_mask = torch.arange(seq_len, device=device).unsqueeze(0) < (input_lengths + front_pad_lengths).unsqueeze(1)
        input_attn_mask = input_mask.unsqueeze(1) & input_mask.unsqueeze(2)  # Shape: (batch_size, seqlen, seqlen)

        # Combine masks:
        # (casual_mask | input_attn_mask) & pad_attn_mask
        # example:
        # 0 0 0 0
        # 0 1 1 0
        # 0 1 1 0
        # 0 1 1 1
        combined_mask = causal_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        combined_mask = (combined_mask | input_attn_mask) & pad_attn_mask

        # Expand mask to match attention heads
        mask = torch.where(combined_mask, torch.zeros(batch_size, seq_len, seq_len, device=device), -1e9)

        # Handle MPS NaN issue
        if device.type == 'mps':
            mask = torch.nan_to_num(mask, nan=0.0)

        return mask.unsqueeze(1)

class Llama3(nn.Module):
    def __init__(self,
                 model_args: ModelArgs,
                 device: str = 'cpu'):
        
        super().__init__()

        # check that dim is a multiple of n_heads
        assert model_args.dim % model_args.n_heads == 0, "dim must be a multiple of n_heads"

        # check that n_heads is a multiple of n_kv_heads
        assert model_args.n_heads % model_args.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"

        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        # Move model components to the initial device
        self.device = torch.device(device)  # Store the device for the model
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim, device=device)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(model_args.n_layers):
            self.layers.append(TransformerBlock(layer_id, model_args, device=device))

        self.norm = RMSNorm(model_args.dim, eps=model_args.norm_eps, device=device)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False, device=device)

        freqs_cis = precompute_freqs_cis(
            dim=model_args.dim // model_args.n_heads,
            end=model_args.context_length,
            theta=model_args.rope_theta,
            device=device
        )

        # Register precomputed values as buffer
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def reset_parameters(self):
        self.tok_embeddings.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.norm.reset_parameters()
        self.output.reset_parameters()

    def forward(self, tokens: torch.Tensor, input_seq_len: Optional[torch.Tensor|list[int]] = None, front_pad_len: Optional[torch.Tensor|list[int]] = None):
        '''
        input_seq_len: (batch_size,) shape. The length of the input sequence for each example in the batch. (Does not include front_pad_len)
        The input sequence tokens are able to attend to each other within the input sequence. The attention for rest of the tokens follow the causal mask.
        '''
        # check sequence length
        assert tokens.shape[1] <= self.model_args.context_length, "sequence length exceeds context length"
        batch_size, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)  # Embeddings layer
        freqs_cis = self.freqs_cis[:seqlen]

        mask = mask_calc(batch_size, seqlen, self.device, front_pad_lengths=front_pad_len, input_lengths=input_seq_len)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h).float()
        return output