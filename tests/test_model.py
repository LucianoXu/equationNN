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

    x = synthesize_example_thread(5, 20, 10, 1000, max_length=100)

    ds = InverseDataset(x, 100)
    example = ds[0]
    print(example)
    input = example['input']
    input_mask = example['input_mask']
    pos_inst = example['pos_inst'] 
    label = example['label']

    args = ModelArgs()
    args.vocab_size = len(term_tokenizer)
    args.output_size = len(opt_tokenizer) + 1
    model = Transformer(args)

    logits = model.forward(input.unsqueeze(0), input_mask.unsqueeze(0), [pos_inst])
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, label.unsqueeze(0))
    print(loss)
