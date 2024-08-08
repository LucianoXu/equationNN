
from datagen import *

def test_single_example_data():
    term = parse("((a+b)+a)")
    opt = rule_comm
    pos = (0,)

    term_data, pos_data, target_data = get_single_example_data(
        term, opt, pos, term_tokenizer, opt_tokenizer
    )
    assert term_data == [1, 1, 2, 3, 2]
    assert pos_data == [(), (0,), (0, 0), (0, 1), (1,)]
    assert target_data == [0, 1, 0, 0, 0]
