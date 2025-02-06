from typing import Optional
from pyualg import Signature, Term, RewriteRule, TermOpt, Subst, Parser
from rewritepath import RewritePath
from scenario import *
import random

def gen_expression(max_height: int) -> Term:
    '''
    Generate an expression tree with height at most max_height
    '''
    if max_height == 1:
        return Term(random.choice(['x', 'y', 'z', 'w', 'u', 'v']))
    
    choice = random.choices(['|', '&', '~', 'x', 'y', 'z', 'w', 'u', 'v'], [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], k=1)[0]

    if choice == '|' or choice == '&':
        left = gen_expression(max_height - 1)
        right = gen_expression(max_height - 1)
        return Term(choice, (left, right))
    elif choice == '~':
        return Term('~', (gen_expression(max_height - 1),))
    else:
        return Term(choice)

def gen_example(max_step: int = 10, max_height: int = 3) -> RewritePath:
    '''
    Generate a path of rewriting operations. The path starts from a random expression tree with height at most max_height.
    The path has at most max_step steps.
    The algorithm will check all possible choices of rules and positions at each step. If no choice is possible, the path will be returned directly.
    '''
    while True:

        # use the equation x = x as the initial term
        # term = gen_expression(max_height)
        term = Term("x")

        path = RewritePath(signature, (), Term('=', (term, term)))

        for _ in range(max_step):
            all_nodes = list(path.current.all_nodes())
            # get the outer product of all_nodes and [r_L2R, r_R2L]
            choices = [(rule, pos) for pos, _ in all_nodes for rule in rule_ls]
            while True:

                # if no choices, return the path directly without further trying
                if not choices:
                    return path
        
                # randomly choose one rule in the gen_rules
                choice = random.choice(choices)
                choices.remove(choice)

                # randomly select one rule and position
                gen_rule, pos = choice

                # remove the direct inverse operation
                if len(path.path)>0 and INV_GEN_RULES[gen_rule] == path.path[-1][1] and pos == path.path[-1][2]:
                    continue

                # generate the given substitution
                given_subst = {}
                for var in INST_VARS[gen_rule]:
                    given_subst[var] = gen_expression(max_height)

                if path.apply(gen_rule, pos, Subst(given_subst), inst_vars=INST_VARS, forbiden_heads=forbidden_heads):
                    break

        # check whether r_OML1 is in the path
        if any([path.path[i][1] == r_OML1 for i in range(len(path.path))]):
            return path
        


if __name__ == "__main__":
    from rewritepath import path_to_examples
    path = gen_example(10, 3)
    print(path)

    examples = path_to_examples(path, signature)
    for example in examples:
        print(example)