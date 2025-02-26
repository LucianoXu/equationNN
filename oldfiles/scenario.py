from typing import Optional
from pyualg import Signature, Term, RewriteRule, TermOpt, Subst, Parser
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

# Substitution (X : x -> Y)
def r_subst(sig: Signature, term: Term, given_subst: Optional[Subst], forbiden_heads: Optional[set[str]]) -> Optional[tuple[Term, Subst]]:

    # if the term is not a variable, return None
    
    if given_subst is None:
        return None
    
    if "X" not in given_subst.data or "Y" not in given_subst.data:
        return None
    
    x = given_subst["X"]

    if not x.is_var(sig):
        return None
    
    # the substituted variable should be in the term
    if x.head not in term.vars(sig):
        return None

    # if the term is a variable, return the substitution
    subst = Subst({x.head: given_subst['Y']})

    return subst(term), subst




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

rule_ls = [r_subst, r_commM, r_commJ, r_assocM1, r_assocM2, r_assocJ1, r_assocJ2, r_absorpM1, r_absorpM2, r_absorpJ1, r_absorpJ2, r_doubleNeg1, r_doubleNeg2, r_deMorganM1, r_deMorganM2, r_deMorganJ1, r_deMorganJ2, r_OML1, r_OML2]

# calculate the required variables for instantiation of the rewriterules
INST_VARS = {rule: rule.inst_vars(signature) for rule in rule_ls[1:]}
INST_VARS[r_subst] = {'X', 'Y'}

forbidden_heads = {'X', 'Y', '='}

INV_GEN_RULES = {
    r_subst: r_subst,
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
    r_subst: 'subst',
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
