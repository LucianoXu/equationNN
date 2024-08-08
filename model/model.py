
import torch

def precompute_theta_pos_frequencies(head_dim: int, length: int, device: str = 'cpu', theta: float = 10000.0):
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
        device: str = 'cpu'):
    '''
    apply the rotary embedding in 'H' (height) direction
    using the frequency vector provided (x does not adopt sequential order)
    indices: the example and node index pairs, indicating the positions to apply the rotary embedding
    '''
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
        device: str = 'cpu'):
    '''
    apply the rotary embedding in 'B' (branch) direction
    using the frequency vector provided (x does not adopt sequential order)
    indices: the example and node index pairs, indicating the positions to apply the rotary embedding
    '''
    # The contiguous operation rearranges the memory. It can be optimized, by explicitly calculating the rotation without using complex numbers.
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).contiguous())
    
    freqs_complex = freqs_vec.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    x_rotated = x_complex.clone()
    example_idx, node_idx = zip(*indices)
    x_rotated[example_idx, node_idx, :, :] = x_complex[example_idx, node_idx, :, :] * freqs_complex

    x_out = torch.view_as_real(x_rotated)

    x_out = x_out.transpose(-1, -2).reshape(*x.shape)

    return x_out.type_as(x).to(device)