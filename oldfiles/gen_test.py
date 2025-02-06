
from model import *
from toplevel import *
from small_args import SmallArgs
from elab import ELab
from ext_solver import vampire_solve
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


def example_test(
    term: Term, # the problem term

    model: nn.Module,
    context_length: int,

    T: float,

    beams: int, step_limit: int,
):

    # state the problem
    print("Problem: ", term.sig_str(signature))

    # run the beam search
    print(f"Beam Search (beams ={beams}, step_limit = {step_limit})")
    res = beam_search(model, term, beams, step_limit, context_length, T)
    print("Reseult: ", res)


    if res is not None:
        print("State-Action Path:")
        prove(ProofKernel(term), list_agent(res))


    # run vampire
    # test the vampire solver
    if vampire:
        print("Vampire result ... ", end="", flush=True)
        if vampire_solve(vampire, Axioms, term):
            print("Succeed")
        else:
            print("Fail")
    
    input()

def forever_test(
    model: nn.Module,
    context_length: int,
    max_step = 4, max_height = 3, T = 0.4, beams = 20, step_limit = 50):


    # run the toplevel prover and have fun!
    while True:
        path = gen_example(max_step=max_step, max_height=max_height)
        term = path.current

        if term.args[0] == term.args[1]:
            continue

        example_test(term, model, context_length, T, beams, step_limit)

def vampire_test(max_step = 4, max_height = 3):
    while True:
        path = gen_example(max_step=max_step, max_height=max_height)
        term = path.current

        if term.args[0] == term.args[1]:
            continue

        print("Problem: ", term.sig_str(signature))
        print("Vampire result ... ", end="", flush=True)
        res = vampire_solve(vampire, Axioms, term)
        if res.is_provable:
            print(f"Succeed in {res.elapsed_time} seconds")
        elif res.timeout:
            print(f"Timeout in {res.elapsed_time} seconds")
            input()
        else:
            raise ValueError("Vampire failed. Please check the problem.")


if __name__ == '__main__':
    vampire_test(20, 3)

    args = SmallArgs()
    model = Llama3(args, device='mps')
    ELab('ckpt/OML', version_name='latest', model=model)

    forever_test(
        model,
        context_length=args.context_length, max_step = 7, beams = 10, step_limit = 20)

    example_test(term, model, args.context_length, 0.6, 50, 50)