# transform the implication problem to TPTP format and solve it using Vampire

from dataclasses import dataclass
from enum import Enum
import subprocess
from typing import Optional
from pyualg import Term, Signature
from scenario import signature, parser
from pathlib import Path
import time
from subprocess import Popen, PIPE
import re


def magma_to_tptp(t: Term) -> str:
    '''
    Transform the term t to TPTP format equation.

    Example: 
    Input: x | x = x | (x | x)
    Output: J(x, x) = J(x, J(x, x))
    '''
    if t.is_var(signature):
        return str(t).upper()
    elif t.head == '|':
        return f'j({magma_to_tptp(t.args[0])}, {magma_to_tptp(t.args[1])})'
    elif t.head == '&':
        return f'm({magma_to_tptp(t.args[0])}, {magma_to_tptp(t.args[1])})'
    elif t.head == '~':
        return f'n({magma_to_tptp(t.args[0])})'
    elif t.head == '=':
        return f'{magma_to_tptp(t.args[0])} = {magma_to_tptp(t.args[1])}'
    else:
        raise ValueError(f'Invalid term {t}')

def problem_to_tptp(A: list[Term], B: Term):
    '''
    A and B are equations.
    Transform the implication problem A -> B to TPTP format.
    '''
    res = ""

    for i in range(len(A)):
        eq = A[i]
        eq_vars = [var.upper() for var in eq.vars(signature)]
        eq_vars.sort()

        res += f'fof(ax{i}, axiom, ![{", ".join(eq_vars)}] : {magma_to_tptp(eq)}).\n\n'

    B_vars = [var.upper() for var in B.vars(signature)]
    B_vars.sort()


    res += f'fof(conj, conjecture, ![{", ".join(B_vars)}] : {magma_to_tptp(B)}).\n'

    return res

@dataclass
class VampireResult:
    '''
    The result of invoking Vampire.
    '''
    elapsed_time: float
    generated_clauses: int
    is_true: Optional[bool] = None  # use None to represent unknown/timeout

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
    
    if stderr:
        raise ValueError(f'Error when invoking Vampire: {stderr}')


    # search for "Generated clauses: ..." in stdout using regex
    match = generated_clauses_re.search(stdout)
    if match:
        generated_clauses = int(match.group(1))
    else:
        raise ValueError(f'Cannot find the number of generated clauses in the output of Vampire: {stdout}')

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

def vampire_solve(vampire: str|Path, A: list[Term], B: Term, timeout : float = 1) -> VampireResult:
    '''
    A and B are equations.
    Solve the implication problem A -> B using Vampire.
    Return the running time in seconds if the problem is solvable.
    '''
    code = problem_to_tptp(A, B)
    return invoke_vampire(vampire, code, timeout)



    