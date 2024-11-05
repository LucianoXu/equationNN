
from model import *
from toplevel import *
from small_args import SmallArgs
from elab import ELab
from ext_solver import vampire_solve
import json

Axiom = parser.parse_term('((X * Y) = ((Y * Y) * X))')

conf = json.load(open('config.json'))
vampire = conf['vampire']


def test_example(
    term: Term, # the problem term

    model: nn.Module,
    context_length: int,

    T: float,

    beams: int, step_limit: int,
):

    model_agent = get_model_agent(model, max_len=context_length, T=T)

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
    print("Vampire result ... ", end="", flush=True)
    if vampire_solve(vampire, Axiom, term):
        print("Succeed")
    else:
        print("Fail")
    
    input()

def forever_test(
    model: nn.Module,
    context_length: int,
    max_step = 4, max_height = 3, T = 0.4, beams = 20, step_limit = 50):


    model_agent = get_model_agent(model, max_len=context_length, T=T)

    # run the toplevel prover and have fun!
    while True:
        path = gen_example(max_step=max_step, max_height=max_height)
        term = path.current

        if term.args[0] == term.args[1]:
            continue

        test_example(term, model, context_length, T, beams, step_limit)


if __name__ == '__main__':
    args = SmallArgs()
    model = Llama3(args, device='cuda')
    ELab('ckpt/VSuper', version_name='latest', model=model)

    # forever_test(
    #     model,
    #     context_length=args.context_length, max_step = 8, beams = 50)

    # SUCCEED 307 x * x = x * (x * x)
    # term = parser.parse_term('((x * x) = (x * (x * x)))')
    # single_test(term)

    # FAIL 4283 x * (x * y) = x * (y * x)
    # term = parser.parse_term('((x * (x * y)) = (x * (y * x)))')

    # 3257 x * x = x * (x * (x * x))
    # term = parser.parse_term('((x * x) = (x * (x * (x * x))))')
    # single_test(term)

    # term = parser.parse_term('((((u * z) * (((z * z) * (z * z)) * (u * u))) * (y * u)) = (((u * u) * y) * (u * z)))')
    # single_test(term)

    term = parser.parse_term('((x * y) = (y * x))')

    # test the example
    test_example(term, model, args.context_length, 0.6, 50, 50)