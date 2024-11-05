
from pyualg import Term, Signature
from scenario import signature, parser
import json

from ext_solver.vampire_solver import magma_to_tptp
def test_magma_to_tptp():
    t = parser.parse_term('((x * y) = (x * (x * x)))')
    assert magma_to_tptp(t) == 'm(X, Y) = m(X, m(X, X))'


from ext_solver.vampire_solver import problem_to_tptp
def test_problem_to_tptp():
    A = parser.parse_term('(X = (Y * (Y * (X * Y))))')
    B = parser.parse_term('(x = (x * ((x * x) * x)))')

    res = problem_to_tptp(A, B)
    assert res == \
'''fof(ax, axiom, ![X, Y] : X = m(Y, m(Y, m(X, Y)))).

fof(conj, conjecture, ![X] : X = m(X, m(m(X, X), X))).
'''

from ext_solver.vampire_solver import vampire_solve
def test_vampire_solve():
    conf = json.load(open('config.json'))
    vampire = conf['vampire']
    A = parser.parse_term('((X * Y) = ((Y * Y) * X))')
    B = parser.parse_term('((x * y) = (y * x))')
    assert vampire_solve(vampire, A, B) == True

    B = parser.parse_term('((x * y) = (x * x))')
    assert vampire_solve(vampire, A, B) == False

