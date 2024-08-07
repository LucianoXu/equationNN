from __future__ import annotations
from treenn import *
import random

    
def gen_expression(depth: int) -> Tree:
    if depth == 0:
        return Leaf(random.choice(['a', 'b']))
    
    choice = random.choices(['+', 'a', 'b'], [0.8, 0.1, 0.1], k=1)[0]
    if choice == '+':
        left = gen_expression(depth - 1)
        right = gen_expression(depth - 1)
        return InfixBinTree('+', left, right)
    
    else:
        return Leaf(choice)

def get_head(depth : int) -> RewritePath:
    start = Leaf('True')
    e = gen_expression(depth)
    res = RewritePath(start, ((rule_eq_expand, (0,), InfixBinTree('=', e, e)),))
    return res

def random_apply(path: RewritePath, opts: list[TreeOpt], retry: int = 10):
    current = path.current
    for _ in range(retry):
        pos = current.get_random_node()[0]
        opt = random.choice(opts)
        if path.apply(opt, pos):
            break

def random_apply_n(path: RewritePath, opts: list[TreeOpt], n: int, retry: int = 10):
    for _ in range(n):
        random_apply(path, opts, retry)

