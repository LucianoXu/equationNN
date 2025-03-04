from ..model import *

def test_mask_calc():
    batch_size = 3
    seq_len = 5
    device = torch.device('cpu')

    front_pad_lengths = [0, 1, 2]
    input_lengths = [2, 0, 2]

    actual_mask = mask_calc(batch_size, seq_len, device, front_pad_lengths, input_lengths)

    target_mask = torch.tensor([
        [[0., 0., -1e9, -1e9, -1e9],
         [0., 0., -1e9, -1e9, -1e9],
         [0., 0., 0., -1e9, -1e9],
         [0., 0., 0., 0., -1e9],
         [0., 0., 0., 0., 0.]],

         [[-1e9, -1e9, -1e9, -1e9, -1e9],
          [-1e9, 0., -1e9, -1e9, -1e9],
          [-1e9, 0., 0., -1e9, -1e9],
          [-1e9, 0., 0., 0., -1e9],
          [-1e9, 0., 0., 0., 0.]],
         
         [[-1e9, -1e9, -1e9, -1e9, -1e9],
          [-1e9, -1e9, -1e9, -1e9, -1e9],
          [-1e9, -1e9, 0., 0., -1e9],
          [-1e9, -1e9, 0., 0., -1e9],
          [-1e9, -1e9, 0., 0., 0.]]
    ]).unsqueeze(1)

    assert torch.allclose(actual_mask, target_mask)

