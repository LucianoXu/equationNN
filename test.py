
from treenn import *
from randomgen import *
from datagen import *
from model import *

if __name__ == '__main__':
    # Test randomgen

    # x = synthesize_example_thread(5, 18, 10, 10000)
    # print(x)

    ##############################
    
    # freqs = precompute_theta_pos_frequencies(6, 3)
    # x = torch.randn(1, 3, 2, 6)
    # print('x')
    # print(x)
    # print('rotated x')
    # indices = [(0, 0), (0, 2)]
    # print(apply_rotary_embeddings_H(x, freqs[2], indices))
    # print(apply_rotary_embeddings_H(
    #     apply_rotary_embeddings_H(x, freqs[1], indices), 
    #     freqs[1], indices))

    ####################################

    # freqs = precompute_theta_pos_frequencies(6, 3)
    # x = torch.randn(2, 3, 2, 6)
    # posinst = [
    #     ([[1,2],[2]], [[[1, 2], []], [[2], []]]),
    #     ([[1,2],[1]], [[[1, 2], []], [[2], []]]),
    # ]
    # print(apply_rotary_embeddings_instructions(x, freqs, posinst))

    ####################################

    args = ModelArgs()
    args.vocab_size = len(term_tokenizer)
    args.output_size = len(opt_tokenizer)
    model = Transformer(args)

    term = parse("a+(a+b)")
    print(term)
    term_data, pos_instruct = get_model_input_from_term(term, 3, 3, term_tokenizer)

    print(model)
    # print the number of parameters
    for p in model.parameters():
        print(p.shape)
    print()
    print(sum(p.numel() for p in model.parameters()))

    print(model.forward(torch.tensor([term_data]), [pos_instruct]))





