from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from subprocess import Popen, PIPE
from .env import env, Scenario
import re

# Vampire Solver

@dataclass
class VampireResult:
    '''
    The result of invoking Vampire.
    '''
    elapsed_time: float
    generated_clauses: int
    is_true: Optional[bool] = None  # use None to represent unknown/timeout

    def __str__(self) -> str:
        return f'VampireResult(elapsed_time={self.elapsed_time}, generated_clauses={self.generated_clauses}, is_true={self.is_true})'
    
    def __repr__(self) -> str:
        return str(self)

generated_clauses_re = re.compile(r'Generated clauses: (\d+)')
time_elapsed_re = re.compile(r'Time elapsed: ([\d.]+) s')

def invoke_vampire(vampire: str|Path, code: str, timeout : float = 10) -> VampireResult:
    '''
    Call Vampire to solve the TPTP format code. Return True if the problem is solvable.

    Returns:
        the running time in seconds if the problem is solvable.
        None if unknown.
    '''

    with Popen([vampire, "--statistics", "full", "-t", f"{timeout}s"], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True) as proc:
        stdout, stderr = proc.communicate(code)

    stderr_cleaned = stderr.strip()
    if (stderr_cleaned and "perf_event_open failed" not in stderr_cleaned):
        raise ValueError(f'When invoking Vampire: {stderr}')
        


    # search for "Generated clauses: ..." in stdout using regex
    match = generated_clauses_re.search(stdout)
    if match:
        generated_clauses = int(match.group(1))
    else:
        raise ValueError(f'Cannot find the number of generated clauses in the output of Vampire: {stdout}\nstderr: {stderr}\ncode: {code}')

    # search for "Time elapsed: ... s" in stdout using regex
    match = time_elapsed_re.search(stdout)
    if match:
        elapsed_time = float(match.group(1))
    else:
        raise ValueError(f'Cannot find the running time in the output of Vampire: {stdout}')

    # decide whether the problem is solvable or timeout
    if "Refutation found" in stdout:
        is_true = True
    elif "Time limit reached!" in stdout:
        is_true = None
    else:
        is_true = False

    # # for debugging
    # print("--------------------CODE----------------------")
    # print(code)
    # print("--------------------STDOUT----------------------")
    # print(stdout)
    # print("--------------------STDERR----------------------")
    # print(stderr)
    # print("--------------------END----------------------")

    return VampireResult(elapsed_time, generated_clauses, is_true)


def vampire_solve(vampire: str|Path, scenario: Scenario, problem: env.Equation, timeout : float = 1) -> VampireResult:
    '''
    Solve the problem using Vampire.
    Return the VampireResult
    '''
    code = env.vampire_problem_encode(scenario.alg, problem, False)
    return invoke_vampire(vampire, code, timeout)


import multiprocessing as mp

def vampire_solve_mp(vampire: str|Path, scenario: Scenario, problems: list[env.Equation], timeout: float = 1) -> list[VampireResult]:
    '''
    Solve the problems using Vampire in parallel.
    '''
    # generate the code
    code_list = [env.vampire_problem_encode(scenario.alg, problem, False) for problem in problems]

    # prepare the arguments
    args = [(vampire, code, timeout) for code in code_list]

    # use multiprocessing to solve the problems
    with mp.Pool() as p:
        return p.starmap(invoke_vampire, args)
    