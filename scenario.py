from typing import Optional
from pyualg import Signature, Term, RewriteRule, TermOpt, Subst, Parser
from rewritepath import RewritePath
import random

# the problem signature
signature = Signature(
    {
        '|' : (2, {'Infix'}),
        '&' : (2, {'Infix'}),
        '~' : (1, set()),
        '=' : (2, {'Infix'}),
    }
)
parser = Parser(signature)

# Commutativity
# a /\ b = b /\ a
r_commM = parser.parse_rewriterule('(X & Y) -> (Y & X)')

# a \/ b = b \/ a
r_commJ = parser.parse_rewriterule('(X | Y) -> (Y | X)')


# Associativity
# (a /\ b) /\ c = a /\ (b /\ c)
r_assocM1 = parser.parse_rewriterule('((X & Y) & Z) -> (X & (Y & Z))')
r_assocM2 = parser.parse_rewriterule('(X & (Y & Z)) -> ((X & Y) & Z)')

# (a \/ b) \/ c = a \/ (b \/ c)
r_assocJ1 = parser.parse_rewriterule('((X | Y) | Z) -> (X | (Y | Z))')
r_assocJ2 = parser.parse_rewriterule('(X | (Y | Z)) -> ((X | Y) | Z)')


# Absorption Laws
# a /\ (a \/ b) = a
r_absorpM1 = parser.parse_rewriterule('(X & (X | Y)) -> X')
r_absorpM2 = parser.parse_rewriterule('X -> (X & (X | Y))')

# a \/ (a /\ b) = a
r_absorpJ1 = parser.parse_rewriterule('(X | (X & Y)) -> X')
r_absorpJ2 = parser.parse_rewriterule('X -> (X | (X & Y))')
 

# Rules for Complement
# a \/ ~a = 1
# a /\ ~a = 0
# we omit them for now

# ~ ~a = a
r_doubleNeg1 = parser.parse_rewriterule('(~ (~ X)) -> X')
r_doubleNeg2 = parser.parse_rewriterule('X -> (~ (~ X))')

# De Morgan's Laws
# ~(a /\ b) = ~a \/ ~b
r_deMorganM1 = parser.parse_rewriterule('(~ (X & Y)) -> ((~ X) | (~ Y))')
r_deMorganM2 = parser.parse_rewriterule('((~ X) | (~ Y)) -> (~ (X & Y))')

# ~(a \/ b) = ~a /\ ~b
r_deMorganJ1 = parser.parse_rewriterule('(~ (X | Y)) -> ((~ X) & (~ Y))')
r_deMorganJ2 = parser.parse_rewriterule('((~ X) & (~ Y)) -> (~ (X | Y))')

# OML Laws
# a \/ b = ((a \/ b) /\ a) \/ ((a \/ b) /\ ~a)
r_OML1 = parser.parse_rewriterule('(X | Y) -> (((X | Y) & X) | ((X | Y) & (~ X)))')
r_OML2 = parser.parse_rewriterule('(((X | Y) & X) | ((X | Y) & (~ X))) -> (X | Y)')

rule_ls = [r_commM, r_commJ, r_assocM1, r_assocM2, r_assocJ1, r_assocJ2, r_absorpM1, r_absorpM2, r_absorpJ1, r_absorpJ2, r_doubleNeg1, r_doubleNeg2, r_deMorganM1, r_deMorganM2, r_deMorganJ1, r_deMorganJ2, r_OML1, r_OML2]

# calculate the required variables for instantiation of the rewriterules
INST_VARS = {rule: rule.inst_vars(signature) for rule in rule_ls}

forbidden_heads = {'X', 'Y', '='}

INV_GEN_RULES = {
    r_commM: r_commM,
    r_commJ: r_commJ,
    r_assocM1: r_assocM2,
    r_assocM2: r_assocM1,
    r_assocJ1: r_assocJ2,
    r_assocJ2: r_assocJ1,
    r_absorpM1: r_absorpM2,
    r_absorpM2: r_absorpM1,
    r_absorpJ1: r_absorpJ2,
    r_absorpJ2: r_absorpJ1,
    r_doubleNeg1: r_doubleNeg2,
    r_doubleNeg2: r_doubleNeg1,
    r_deMorganM1: r_deMorganM2,
    r_deMorganM2: r_deMorganM1,
    r_deMorganJ1: r_deMorganJ2,
    r_deMorganJ2: r_deMorganJ1,
    r_OML1: r_OML2,
    r_OML2: r_OML1,
}

RULE_NAMES: dict[TermOpt, str] = {
    r_commM: 'commM',
    r_commJ: 'commJ',
    r_assocM1: 'assocM1',
    r_assocM2: 'assocM2',
    r_assocJ1: 'assocJ1',
    r_assocJ2: 'assocJ2',
    r_absorpM1: 'absorpM1',
    r_absorpM2: 'absorpM2',
    r_absorpJ1: 'absorpJ1',
    r_absorpJ2: 'absorpJ2',
    r_doubleNeg1: 'doubleNeg1',
    r_doubleNeg2: 'doubleNeg2',
    r_deMorganM1: 'deMorganM1',
    r_deMorganM2: 'deMorganM2',
    r_deMorganJ1: 'deMorganJ1',
    r_deMorganJ2: 'deMorganJ2',
    r_OML1: 'OML1',
    r_OML2: 'OML2',
}

RULE_NAMES_INV = {v: k for k, v in RULE_NAMES.items()}


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

        term = gen_expression(max_height)
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
        





