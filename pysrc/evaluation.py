
from .env import env, Scenario
from .ext_solver import vampire_solve, vampire_solve_mp, VampireResult
from math import log

def intere_fun(vampire_res: VampireResult, size: int) -> float:
    '''
    Compute the interestingness of a problem.
    '''
    # return log((vampire_res.elapsed_time + 0.001) / size)
    return log(vampire_res.generated_clauses / (size + 42))


def test_intere(args: tuple[str, Scenario, env.Equation, float]) -> float:
    '''
    Test the interestingness function.
    Input: algebra, example, timeout
    '''
    vampire, scenario, example, timeout = args
    size = example.size
    res = vampire_solve(vampire, scenario, example, timeout)

    if res.is_true is False:
        raise ValueError("Vampire reports the problem is invalid.")
    
    return intere_fun(res, size)


def test_intere_mp(vampire: str, scenario: Scenario, examples: list[env.Equation], timeout: float = 10) -> list[float]:
    '''
    Use multiprocess to test the interestingness function.
    '''
    vampire_results = vampire_solve_mp(vampire, scenario, examples, timeout)

    return [intere_fun(res, example.size) for res, example in zip(vampire_results, examples)]


def test_intere_mp_args(vampire: str, scenario: Scenario, examples: list[env.Equation], timeout: float = 10) -> list[tuple[int, int, float]]:
    '''
    Use multiprocess to test the interestingness function.

    Output: a list of tuples (size, complexity, interestingness)
    '''
    vampire_results = vampire_solve_mp(vampire, scenario, examples, timeout)

    return [(example.size, res.generated_clauses, intere_fun(res, example.size)) for res, example in zip(vampire_results, examples)]
    

def calc_avg_intere(vampire: str, scenario: Scenario, examples: list[env.Equation], timeout: float = 10) -> float:
    '''
    Calculate the average interestingness of a list of examples.
    '''
    return sum(test_intere_mp(vampire, scenario, examples, timeout)) / len(examples)


if __name__ == "__main__":
    pass
    # from gen import gen_example, signature
    # from scenario import r_subst

    # for _ in range(15):
    #     path = gen_example(20, 3)
    #     # print(path)
    #     # for example in path.path:
    #     #     if example[1] == r_subst:
    #     #         input()

    #     print("Testing interestingness ...")
    #     interestingness = test_intere([term for term, _, _, _, _ in path.path])
    #     print(interestingness)


