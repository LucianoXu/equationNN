
from model import *
from toplevel import *
from utilities import *
from small_args import SmallArgs

model_checkpoint = 'small_rl_5.pth'

def forever_test(max_step = 4, max_height = 3, T = 0.4, beams = 20, step_limit = 50):

    args = SmallArgs()
    model = load_model(model_checkpoint, args, 'cuda')

    model_agent = get_model_agent(model, max_len=args.max_seq_len, T=T)

    # run the toplevel prover and have fun!
    while True:
        path = gen_example(max_step=max_step, max_height=max_height)
        term = path.current

        # skip the trivial case
        if term.args[0] == term.args[1]:
            continue

        print("Problem: ", term.sig_str(signature))
        print(f"Beam Search (beams ={beams}, step_limit = {step_limit})")
        res = beam_search(model, term, beams, step_limit, T)
        print("Reseult: ", res)
        input()


        kernel = ProofKernel(term)
        prove(kernel, model_agent)

        input()


def single_test(term: Term, beam_number = 20, step_limit = 50, T = 0.6):
    # load the model
    args = SmallArgs()
    model = load_model(model_checkpoint, args, 'cuda')

    model_agent = get_model_agent(model, max_len=args.max_seq_len, T=T)

    print("Problem: ", term.sig_str(signature))
    print(f"Beam Search (beams ={beam_number}, step_limit = {step_limit})")
    res = beam_search(model, term, beam_number = beam_number, step_limit = step_limit, T=T)
    print("Reseult: ", res)

    # kernel = ProofKernel(term)
    # prove(kernel, model_agent)


if __name__ == '__main__':
    # forever_test(max_step = 6, beams = 20)

    # SUCCEED 307 x * x = x * (x * x)
    # term = parser.parse_term('((x * x) = (x * (x * x)))')
    # single_test(term)

    # FAIL 4283 x * (x * y) = x * (y * x)
    # term = parser.parse_term('((x * (x * y)) = (x * (y * x)))')
    # single_test(term)

    # 3257 x * x = x * (x * (x * x))
    # term = parser.parse_term('((x * x) = (x * (x * (x * x))))')
    # single_test(term)

    # term = parser.parse_term('((((u * z) * (((z * z) * (z * z)) * (u * u))) * (y * u)) = (((u * u) * y) * (u * z)))')
    # single_test(term)

    term = parser.parse_term('((x * y) = (y * x))')
    single_test(term)