
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tokenizer import TOKENS, tok_decode, tok_encode,  token2id


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 32
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = len(TOKENS)
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_seq_len: int = 96



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freq_llama = torch.outer(t, freqs)
    cos = torch.cos(freq_llama)
    sin = torch.sin(freq_llama)
    freq_llama_cis = torch.stack((cos, sin), dim=-1)
    return freq_llama_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 1 < ndim, "Input tensor must have at least 2 dimensions"
    assert freqs_cis.shape == (x.shape[1], x.shape[-2], 2), "Invalid shape input shape: " + str(x.shape) + " cannot broadcast with freqs_cis shape: " + str(freqs_cis.shape)
    shape = [d if i == 1 or i == ndim - 2 else 1 for i, d in enumerate(x.shape[:-1])]
    shape += [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_real = xq_[..., 0] * freqs_cis[..., 0] - xq_[..., 1] * freqs_cis[..., 1]
    xq_imag = xq_[..., 0] * freqs_cis[..., 1] + xq_[..., 1] * freqs_cis[..., 0]
    xq_out = torch.cat((xq_real, xq_imag), dim=-1).flatten(3)

    xk_real = xk_[..., 0] * freqs_cis[..., 0] - xk_[..., 1] * freqs_cis[..., 1]
    xk_imag = xk_[..., 0] * freqs_cis[..., 1] + xk_[..., 1] * freqs_cis[..., 0]
    xk_out = torch.cat((xk_real, xk_imag), dim=-1).flatten(3)
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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )
        
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
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, device: str):
        super().__init__()
        self.device = device

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        freqs_cis = precompute_freqs_cis(
            dim=params.dim // params.n_heads,
            end=params.max_seq_len * 2,
            theta=params.rope_theta,
        )

        # register precomputed as buffer
        self.register_buffer("freqs_cis", freqs_cis)
        
        self.to(device=device)

    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[: seqlen]

        
        mask = torch.full((seqlen, seqlen), float("-inf"), device=self.device)
        mask = torch.triu(mask, diagonal=1)
        if mask.device.type == torch.device('mps').type:
            mask = torch.nan_to_num(mask, nan=0.0)
        

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
            
        h = self.norm(h)
        output = self.output(h).float()
        return output



if __name__ == "__main__":
    model = Transformer(ModelArgs(), "mps")
    
    # output the number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))