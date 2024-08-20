
from datagen import PosInst
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def precompute_theta_pos_frequencies(head_dim: int, length: int, device: str|torch.device = 'cpu', theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()

    # Shape: (Head_Dim / 2)
    theta_m = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)

    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(length, device=device)

    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta_m).float()

    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings_H(
        x: torch.Tensor, 
        freqs_vec: torch.Tensor, 
        indices: list[tuple[int, int]],
        device: str|torch.device = 'cpu'):
    '''
    apply the rotary embedding in 'H' (height) direction
    using the frequency vector provided (x does not adopt sequential order)
    indices: the example index in batch and the node index. indicating the positions to apply the rotary embedding

    Note: the angle of permutation is controlled by the freqs_vec
    '''
    if len(indices) == 0:
        return x
    
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Head_Dim/2) --> (1, 1, 1, Head_Dim/2)
    freqs_complex = freqs_vec.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Apply the rotary embedding to the specified indices
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex.clone()
    example_idx, node_idx = zip(*indices)
    x_rotated[example_idx, node_idx, :, :] = x_complex[example_idx, node_idx, :, :] * freqs_complex

    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


def apply_rotary_embeddings_B(
        x: torch.Tensor, 
        freqs_vec: torch.Tensor, 
        indices: list[tuple[int, int]],
        device: str|torch.device = 'cpu'):
    '''
    apply the rotary embedding in 'B' (branch) direction
    using the frequency vector provided (x does not adopt sequential order)
    indices: the example and node index pairs, indicating the positions to apply the rotary embedding

    Note: the angle of permutation is controlled by the freqs_vec
    '''
    if len(indices) == 0:
        return x
    
    # The contiguous operation rearranges the memory. It can be optimized, by explicitly calculating the rotation without using complex numbers.
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).contiguous())
    
    freqs_complex = freqs_vec.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    x_rotated = x_complex.clone()
    example_idx, node_idx = zip(*indices)
    x_rotated[example_idx, node_idx, :, :] = x_complex[example_idx, node_idx, :, :] * freqs_complex

    x_out = torch.view_as_real(x_rotated)

    x_out = x_out.transpose(-1, -2).reshape(*x.shape)

    return x_out.type_as(x).to(device)

import itertools

def apply_rotary_embeddings_instructions(
        x: torch.Tensor,
        freqs: torch.Tensor,
        pos_instructs: list[PosInst],
        device: str|torch.device = 'cpu'):
    '''
    apply the rotary embedding specified by pos_instructs, using apply rotary embeddings in 'H' and 'B' directions

    freqs: the tenser containing rotation frequencies for all different angles
    '''
    Hinstructs_batch = [instruct[0] for instruct in pos_instructs]
    Binstructs_batch = [instruct[1] for instruct in pos_instructs]
    for i in range(len(Hinstructs_batch[0])):
        # process all the H-instructions
        Hinstruct = [[(batch_i, x) for x in Hinstructs_batch[batch_i][i]] for batch_i in range(len(Hinstructs_batch))]
        Hinstruct = list(itertools.chain(*Hinstruct))
        
        x = apply_rotary_embeddings_H(x, freqs[1], Hinstruct, device=device)
        
        # process all the B-instructions
        for b in range(len(Binstructs_batch[0][i])):
            Binstruct = [[(batch_i, x) for x in Binstructs_batch[batch_i][i][b]] for batch_i in range(len(Binstructs_batch))]
            Binstruct = list(itertools.chain(*Binstruct))
            x = apply_rotary_embeddings_B(x, freqs[b+1], Binstruct, device=device)

    return x


@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 8

    vocab_size: int = -1 # Later set in the build method
    output_size: int = -1 # Later set in the build method

    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # this max depth is used to precompute the frequency vectors
    max_depth: int = 20

    device: str|torch.device = 'cpu'


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)
    


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads
        self.n_heads = args.n_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        pos_instructs: list[PosInst],
        freqs_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape  # (B, Seq_Len, Dim)

        # (B, Seq_Len, Dim) -> (B, Seq_Len, H * Head_Dim)
        xq = self.wq(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H * Head_Dim)
        xk = self.wk(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H * Head_Dim)
        xv = self.wv(x)

        # (B, Seq_Len, H * Head_Dim) -> (B, Seq_Len, H, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # (B, Seq_Len, H * Head_Dim) -> (B, Seq_Len, H, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # (B, Seq_Len, H * Head_Dim) -> (B, Seq_Len, H, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # (B, Seq_Len, H, Head_Dim) --> (B, Seq_Len, H, Head_Dim)
        xq = apply_rotary_embeddings_instructions(xq, freqs_complex, pos_instructs, device=x.device)
        # (B, Seq_Len, H, Head_Dim) --> (B, Seq_Len, H, Head_Dim)
        xk = apply_rotary_embeddings_instructions(xk, freqs_complex, pos_instructs, device=x.device)

        # (B, Seq_Len, H, Head_Dim) -> (B, H, Seq_Len, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len, H, Head_Dim) -> (B, H, Seq_Len, Head_Dim)
        xk = xk.transpose(1, 2)
        # (B, Seq_Len, H, Head_Dim) -> (B, H, Seq_Len, Head_Dim)
        xv = xv.transpose(1, 2)

        # (B, H, Seq_Len, Head_Dim) @ (B, H, Head_Dim, Seq_Len) -> (B, H, Seq_Len, Seq_Len)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # apply the mask
        scores = scores.float().masked_fill_(masks == 0, -1e9)
        # (B, H, Seq_Len, Seq_Len) -> (B, H, Seq_Len, Seq_Len)
        scores = F.softmax(scores, dim=-1).type_as(xq)


        # (B, H, Seq_Len, Seq_Len) @ (B, H, Seq_Len, Head_Dim) -> (B, H, Seq_Len, Head_Dim)
        output = torch.matmul(scores, xv)
        # (B, H, Seq_Len, Head_Dim) -> (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
    

class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, 
                x: torch.Tensor, 
                masks: torch.Tensor,
                pos_instructs: list[PosInst],
                freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), masks, pos_instructs, freqs_complex
        )
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"
        assert args.output_size != -1, "Output size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.output_size = args.output_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(args.dim, self.output_size, bias=False)

        # self.freqs_complex is not the parameter of the model
        self.freqs_complex = precompute_theta_pos_frequencies(
            head_dim = self.args.dim // self.args.n_heads, 
            length = self.args.max_depth * 2, 
            device = self.args.device)
        
        self.freqs_complex.requires_grad = False
        
    def forward(self, 
                tokens: torch.Tensor, 
                masks: torch.Tensor,
                pos_instructs: list[PosInst]):

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)
        
        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, masks, pos_instructs, self.freqs_complex)

        h = self.norm(h)

        # the final result shape suits the cross entropy loss
        # (B, Seq_Len, Dim) -> (B, Seq_Len, Output_Size) -> (B, Output_Size, Seq_Len)
        output = self.output(h).float().transpose(1, 2)

        return output