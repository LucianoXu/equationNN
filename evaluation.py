
from pyualg import Term
from scenario import parser
from ext_solver import vampire_solve
from math import log
import json

# create the axioms for vampire
Ax_commC = parser.parse_term('((X & Y) = (Y & X))')
Ax_commJ = parser.parse_term('((X | Y) = (Y | X))')
Ax_assocC = parser.parse_term('(((X & Y) & Z) = (X & (Y & Z)))')
Ax_assocJ = parser.parse_term('(((X | Y) | Z) = (X | (Y | Z)))')
Ax_absorpC = parser.parse_term('((X & (X | Y)) = X)')
Ax_absorpJ = parser.parse_term('((X | (X & Y)) = X)')
Ax_doubleNeg = parser.parse_term('((~ (~ X)) = X)')
Ax_deMorganC = parser.parse_term('((~ (X & Y)) = ((~ X) | (~ Y)))')
Ax_deMorganJ = parser.parse_term('((~ (X | Y)) = ((~ X) & (~ Y)))')
Ax_oml = parser.parse_term('((X | Y) = (((X | Y) & X) | ((X | Y) & (~ X))))')

Axioms = [Ax_commC, Ax_commJ, Ax_assocC, Ax_assocJ, Ax_absorpC, Ax_absorpJ, Ax_doubleNeg, Ax_deMorganC, Ax_deMorganJ, Ax_oml]

conf = json.load(open('config.json'))
vampire = conf['vampire']


def intere_fun(vampire_time: float, size: int) -> float:
    '''
    Compute the interestingness of a problem.
    '''
    return log((vampire_time) / size)

def _test_intere(args: tuple[Term, float]) -> float:
    '''
    Test the interestingness function.
    '''
    example, timeout = args
    size = example.size
    res = vampire_solve(vampire, Axioms, example, timeout)
    if res.is_provable:
        vampire_time = res.elapsed_time
    elif res.timeout:
        vampire_time = timeout
    else:
        raise ValueError(f"Vampire failed to solve the problem {str(example)}")
    
    return intere_fun(vampire_time, size)


from multiprocessing import Pool

def test_intere(examples: list[Term], timeout: float = 10) -> list[float]:
    '''
    Use multiprocess to test the interestingness function.
    '''
    # prepare the arguments
    args = [(example, timeout) for example in examples]

    with Pool() as p:
        return p.map(_test_intere, args)


def calc_avg_intere(examples: list[Term], timeout: float = 10) -> float:
    '''
    Calculate the average interestingness of a list of examples.
    '''
    return sum(test_intere(examples, timeout)) / len(examples)


if __name__ == "__main__":
    from rewritepath import path_to_examples
    from gen import gen_example, signature
    from scenario import r_subst

    for _ in range(15):
        path = gen_example(20, 3)
        # print(path)
        # for example in path.path:
        #     if example[1] == r_subst:
        #         input()

        print("Testing interestingness ...")
        interestingness = test_intere([term for term, _, _, _, _ in path.path])
        print(interestingness)


