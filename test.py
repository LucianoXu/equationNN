
from treenn import *
from randomgen import *
from datagen import *
from model import *

if __name__ == '__main__':
    # random.seed(42)

    # path = get_head(4)
    # print(path)

    # print()
    # random_rule = [rule_comm, rule_assoc1, rule_assoc2]
    # random_apply_n(path, random_rule, 10)

    # print(path)

    # path.verify(set(random_rule))

    # invpath = path.get_inverse(inverse_table)
    
    # print()
    # print(invpath)
    # print()

    # invpath.verify(set(random_rule))

    # print(get_term_opt_pairs(path))

    ###########################

    # term = parse("((a+b)+a)")
    # opt = rule_comm
    # pos = (0,)

    # term_data, pos_data, target_data = get_single_example_data(
    #     term, opt, pos, term_tokenizer, opt_tokenizer
    # )
    # print(term_data)
    # print(pos_data)
    # print(target_data)

    ##############################
    
    freqs = precompute_theta_pos_frequencies(6, 3)
    print("freqs")
    print(freqs)
    x = torch.randn(1, 3, 2, 6)
    print('x')
    print(x)
    print('rotated x')
    indices = [(0, 0), (0, 2)]
    print(apply_rotary_embeddings_H(x, freqs[2], indices))
    print(apply_rotary_embeddings_H(
        apply_rotary_embeddings_H(x, freqs[1], indices), 
        freqs[1], indices))