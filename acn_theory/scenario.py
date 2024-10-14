from typing import Optional
from pyualg import Signature, Term, RewriteRule, TermOpt
from .rewritepath import RewritePath
import random

# a language with symbols of +, ~ and 0, satisfying:
# + is AC
# a + ~ a = 0
# 0 + a = a
# a + 0 = a

signature = Signature(
    {
        '+' : (2, {'Infix'}),
        '~' : (1, set()),
    }
)

def r_comm(sig: Signature, term: Term) -> Optional[Term]:
    r'''
    x + y -> y + x
    '''
    if term.is_var(sig):
        return None
    if term.head == '+':
        return Term('+', (term.args[1], term.args[0]))
    return None

def r_assoc1C(sig: Signature, term: Term) -> Optional[Term]:
    r'''
    (x + y) + z -> x + (y + z)
    '''
    if term.is_var(sig):
        return None
    if term.head == '+':
        if term.args[0].head == '+':
            return Term('+', (term.args[0].args[0], Term('+', (term.args[0].args[1], term.args[1]))))
    return None

def r_assoc2C(sig: Signature, term: Term) -> Optional[Term]:
    r'''
    x + (y + z) -> (x + y) + z
    '''
    if term.is_var(sig):
        return None
    if term.head == '+':
        if term.args[1].head == '+':
            return Term('+', (Term('+', (term.args[0], term.args[1].args[0])), term.args[1].args[1]))
    return None


def r_add0(sig: Signature, term: Term) -> Optional[Term]:
    r'''
    x + 0 -> x
    '''
    if term.is_var(sig):
        return None
    if term.head == '+':
        if term.args[1].head == '0':
            return term.args[0]
    return None


def r_0add(sig: Signature, term: Term) -> Optional[Term]:
    r'''
    0 + x -> x
    '''
    if term.is_var(sig):
        return None
    if term.head == '+':
        if term.args[0].head == '0':
            return term.args[1]
    return None


def r_add0_inv(sig: Signature, term: Term) -> Optional[Term]:
    return Term('+', (term, Term('0')))

def r_0add_inv(sig: Signature, term: Term) -> Optional[Term]:
    return Term('+', (Term('0'), term))


def r_add_neg(sig: Signature, term: Term) -> Optional[Term]:
    r'''
    x + ~x -> 0
    '''
    if term.is_var(sig):
        return None
    if term.head == '+':
        if term.args[1].head == '~':
            if term.args[0] == term.args[1].args[0]:
                return Term('0')
    return None

def r_add_neg_inv(sig: Signature, term: Term) -> Optional[Term]:
    '''
    return a term that can be rewritten to term by r_add_neg
    '''
    if term == Term('0'):
        rand_term = gen_expression(2)
        return Term('+', (rand_term, Term('~', (rand_term,))))
    return None

def gen_expression(max_height: int) -> Term:
    '''
    Generate an expression tree with height at most max_height
    '''
    if max_height == 1:
        return Term(random.choice(['a', 'b', 'c', 'd']))
    
    choice = random.choices(['+', '~', '0', 'a', 'b', 'c', 'd'], [0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1], k=1)[0]

    if choice == '+':
        left = gen_expression(max_height - 1)
        right = gen_expression(max_height - 1)
        return Term('+', (left, right))
    
    elif choice == '~':
        return Term('~', (gen_expression(max_height - 1),))
    
    else:
        return Term(choice)


INV_GEN_RULES = {
    r_comm: r_comm,
    r_assoc2C: r_assoc1C,
    r_assoc1C: r_assoc2C,
    r_add0_inv: r_add0,
    r_0add_inv: r_0add,
    r_add_neg_inv: r_add_neg
}

RULE_NAMES: dict[TermOpt, str] = {
    r_comm: 'COMM',
    r_assoc1C: 'ASSOC1C',
    r_assoc2C: 'ASSOC2C',
    r_add0: 'ADDr',
    r_0add: 'ADDl',
    r_add_neg: 'ADDNEG',
}

def gen_example(steps: int = 10) -> RewritePath:
    path = RewritePath(signature, (), Term('0'))

    for _ in range(steps):
        while True:
            
            # randomly choose one rule in the gen_rules
            gen_rule = random.choices(
                [r_comm,
                r_assoc2C,
                r_assoc1C,
                r_add0_inv,
                r_0add_inv,
                r_add_neg_inv], 
                [0.02, 0.02, 0.02, 0.02, 0.02, 0.9], k=1)[0]

            # randomly select one node
            pos, node = path.current.get_random_node()

            if path.apply(gen_rule, pos):
                break

    return path
    





