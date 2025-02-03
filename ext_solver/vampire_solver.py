# transform the implication problem to TPTP format and solve it using Vampire

from typing import Optional
from pyualg import Term, Signature
from scenario import signature, parser
from pathlib import Path

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

def invoke_vampire(vampire: str|Path, code: str) -> Optional[bool]:
    '''
    Call Vampire to solve the TPTP format code. Return True if the problem is solvable.

    Returns:
        bool, optional: True if the TPTP problem is refutation and False if it is satisfiable. None if unknown.
    '''
    from subprocess import Popen, PIPE

    with Popen([vampire], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True) as proc:
        stdout, stderr = proc.communicate(code)
        # # for debugging
        # print("--------------------CODE----------------------")
        # print(code)
        # print("--------------------STDOUT----------------------")
        # print(stdout)
        # print("--------------------STDERR----------------------")
        # print(stderr)
        # print("--------------------END----------------------")

    if stderr:
        raise ValueError(f'Error when invoking Vampire: {stderr}')
    
    if 'Refutation found' in stdout:
        return True
    elif 'Termination reason: Satisfiable' in stdout:
        return False
    else:
        return None

def vampire_solve(vampire: str|Path, A: list[Term], B: Term) -> Optional[bool]:
    '''
    A and B are equations.
    Solve the implication problem A -> B using Vampire.
    '''
    code = problem_to_tptp(A, B)
    return invoke_vampire(vampire, code)



    