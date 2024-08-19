import torch

from model import *
from datagen import *

def test_RoPE_H():
    freqs = precompute_theta_pos_frequencies(6, 3)
    x = torch.randn(1, 3, 2, 6)
    indices = [(0, 0), (0, 2)]
    r2 = apply_rotary_embeddings_H(x, freqs[2], indices)
    r1r1 = apply_rotary_embeddings_H(
        apply_rotary_embeddings_H(x, freqs[1], indices), 
        freqs[1], indices)
    assert torch.allclose(r2, r1r1)

def test_RoPE_B():
    freqs = precompute_theta_pos_frequencies(6, 3)
    x = torch.randn(1, 3, 2, 6)
    indices = [(0, 0), (0, 2)]
    r2 = apply_rotary_embeddings_B(x, freqs[2], indices)
    r1r1 = apply_rotary_embeddings_B(
        apply_rotary_embeddings_B(x, freqs[1], indices), 
        freqs[1], indices)
    assert torch.allclose(r2, r1r1)

def test_RoPE_example():

    freqs = precompute_theta_pos_frequencies(6, 3)
    x = torch.randn(2, 3, 2, 6)
    posinst = [
        ([[1,2],[2]], [[[1, 2], []], [[2], []]]),
        ([[1,2],[1]], [[[1, 2], []], [[2], []]]),
    ]
    apply_rotary_embeddings_instructions(x, freqs, posinst)


def test_model_forward():

    args = ModelArgs()
    args.vocab_size = len(term_tokenizer)
    args.output_size = len(opt_tokenizer)
    model = Transformer(args)

    term = parse("a+(a+b)")

    term_data, pos_instruct = get_model_input_from_term(term, 3, 3, term_tokenizer)
    model.forward(torch.tensor([term_data]), [pos_instruct])



