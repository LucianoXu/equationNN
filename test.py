
from treenn import *

if __name__ == '__main__':
    term = parse("(a+b)+(b+a)")
    assert term.apply((), rule_comm) == parse("(b+a)+(a+b)")
    print(term.apply((0,), rule_comm))
    