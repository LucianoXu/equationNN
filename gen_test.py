
from model import *
from toplevel import *
from utilities import *
from small_args import SmallArgs

model_checkpoint = 'trained_parameters.pth'

def forever_test():

    args = SmallArgs()
    model = load_model('small_rl.pth', args, 'mps')

    model_agent = get_model_agent(model, max_len=args.max_seq_len, T=0.4)

    # run the toplevel prover and have fun!
    while True:
        path = gen_example(3, 3)
        term = path.current

        # skip the trivial case
        if term.args[0] == term.args[1]:
            continue

        kernel = ProofKernel(term)
        prove(kernel, model_agent)

        input()


def single_test(term: Term):
    # load the model
    args = SmallArgs()
    model = load_model('small_rl.pth', args, 'mps')

    model_agent = get_model_agent(model, max_len=args.max_seq_len, T=0.6)

    kernel = ProofKernel(term)
    prove(kernel, model_agent)


if __name__ == '__main__':
    forever_test()

    # SUCCEED 307 x * x = x * (x * x)
    # term = parser.parse_term('((x * x) = (x * (x * x)))')
    # single_test(term)

    # FAIL 4283 x * (x * y) = x * (y * x)
    # term = parser.parse_term('((x * (x * y)) = (x * (y * x)))')
    # single_test(term)

    # 3257 x * x = x * (x * (x * x))
    term = parser.parse_term('((x * x) = (x * (x * (x * x))))')
    single_test(term)