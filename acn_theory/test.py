
from .scenario import *

def test_gen_example():
    path = gen_example(20)
    print(path)
    print()

    invpath = path.get_inverse(INV_GEN_RULES)
    print(invpath)
