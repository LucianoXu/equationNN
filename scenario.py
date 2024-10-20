from typing import Optional
from pyualg import Signature, Term, RewriteRule, TermOpt, Parser
from rewritepath import RewritePath
import random

# a Magma with only one equational theory:
# Equation387[x ◇ y = (y ◇ y) ◇ x]
# See (https://teorth.github.io/equational_theories/implications/?387)

signature = Signature(
    {
        '*' : (2, {'Infix'}),
        '=' : (2, {'Infix'}),
    }
)
parser = Parser(signature)

r_L2R = parser.parse_rewriterule('(X * Y) -> ((Y * Y) * X)')
r_R2L = parser.parse_rewriterule('((Y * Y) * X) -> (X * Y)')

INV_GEN_RULES = {
    r_L2R: r_R2L,
    r_R2L: r_L2R,
}

RULE_NAMES: dict[TermOpt, str] = {
    r_L2R: 'L2R',
    r_R2L: 'R2L',
}


def gen_expression(max_height: int) -> Term:
    '''
    Generate an expression tree with height at most max_height
    '''
    if max_height == 1:
        return Term(random.choice(['x', 'y', 'z', 'w', 'u', 'v']))
    
    choice = random.choices(['*', 'x', 'y', 'z', 'w', 'u', 'v'], [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], k=1)[0]

    if choice == '*':
        left = gen_expression(max_height - 1)
        right = gen_expression(max_height - 1)
        return Term('*', (left, right))
    
    else:
        return Term(choice)

def gen_example(max_step: int = 10, max_height: int = 3) -> RewritePath:
    '''
    Generate a path of rewriting operations. The path starts from a random expression tree with height at most max_height.
    The path has at most max_step steps.
    The algorithm will check all possible choices of rules and positions at each step. If no choice is possible, the path will be returned directly.
    '''
    term = gen_expression(max_height)
    path = RewritePath(signature, (), Term('=', (term, term)))

    for _ in range(max_step):
        all_nodes = list(path.current.all_nodes())
        # get the outer product of all_nodes and [r_L2R, r_R2L]
        choices = [(rule, pos) for pos, _ in all_nodes for rule in [r_L2R, r_R2L]]

        while True:

            # if no choices, return the path directly without further trying
            if not choices:
                return path
    
            # randomly choose one rule in the gen_rules
            choice = random.choice(choices)
            choices.remove(choice)

            # randomly select one node
            gen_rule, pos = choice

            if path.apply(gen_rule, pos):
                break

    return path
    





