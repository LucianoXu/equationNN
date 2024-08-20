
from datagen import *

def test_single_example_data():
    term = parse("((a+b)+a)")
    opt = rule_comm
    pos = (0,)

    term_data, pos_list, pos_instruct, target_data = get_single_example_data(
        term=term,
        max_height=3,
        width=2,
        opt=opt,
        pos=pos, 
        term_tokenizer=term_tokenizer, 
        opt_tokenizer=opt_tokenizer
    )
    assert term_data == [3, 3, 4, 5, 4]
    assert pos_list == [(), (0,), (0, 0), (0, 1), (1,)]
    assert pos_instruct == (
        [[1,2,3,4],[2,3]], 
        [[[1,2,3], [4]], [[2], [3]]])
    assert target_data == [0, 1, 0, 0, 0]