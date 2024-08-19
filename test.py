
from treenn import *
from randomgen import *
from datagen import *
from model import *

if __name__ == '__main__':
    # Test randomgen

    x = synthesize_example_thread(5, 18, 10, 10000)
    print(x)

    ##############################
    
    # freqs = precompute_theta_pos_frequencies(6, 3)
    # print("freqs")
    # print(freqs)
    # x = torch.randn(1, 3, 2, 6)
    # print('x')
    # print(x)
    # print('rotated x')
    # indices = [(0, 0), (0, 2)]
    # print(apply_rotary_embeddings_H(x, freqs[2], indices))
    # print(apply_rotary_embeddings_H(
    #     apply_rotary_embeddings_H(x, freqs[1], indices), 
    #     freqs[1], indices))