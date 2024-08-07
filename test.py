
from treenn import *
from randomgen import *

if __name__ == '__main__':
    # random.seed(42)

    path = get_head(4)
    print(path)

    print()
    random_rule = [rule_comm, rule_assoc1, rule_assoc2]
    random_apply_n(path, random_rule, 10)

    print(path)

    path.verify(set(random_rule))

    invpath = path.get_inverse(inverse_table)
    
    print()
    print(invpath)
    print()

    invpath.verify(set(random_rule))