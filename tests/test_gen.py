from randomgen import *


def test_random_node():
    for i in range(50):
        ex = gen_expression(6)
        pos, node = ex.get_random_node()
        assert node == ex[pos]

def test_random_apply():
    for i in range(50):
        path = get_head(5)
        random_rule = [rule_comm, rule_assoc1, rule_assoc2]
        random_apply_n(path, random_rule, 10)
        path.verify(set(random_rule))
        
        invpath = path.get_inverse(inverse_table)
        invpath.verify(set(random_rule))
        