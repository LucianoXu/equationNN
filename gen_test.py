
from model import *
from toplevel import *

model_checkpint = 'trained_parameters.pth'

def forever_test():
    # load the model
    device = 'mps'
    SEQ_LEN = 96

    model_args = ModelArgs()
    model_args.max_seq_len = SEQ_LEN
    model = Transformer(ModelArgs(), device)
    model.load_state_dict(torch.load(model_checkpint, weights_only=True, map_location=device))

    model_agent = get_model_agent(model, max_len=256, T=0.4)

    # run the toplevel prover and have fun!
    while True:
        path = gen_example(5, 3)
        term = path.current

        # skip the trivial case
        if term.args[0] == term.args[1]:
            continue

        kernel = ProofKernel(term)
        prove(kernel, model_agent)

        input()


def single_test(term: Term):
    # load the model
    device = 'mps'
    SEQ_LEN = 96

    model_args = ModelArgs()
    model_args.max_seq_len = SEQ_LEN
    model = Transformer(ModelArgs(), device)
    model.load_state_dict(torch.load(model_checkpint, weights_only=True, map_location=device))

    model_agent = get_model_agent(model, max_len=256, T=0.6)

    kernel = ProofKernel(term)
    prove(kernel, model_agent)


if __name__ == '__main__':
    forever_test()

    # SUCCEED 307 x * x = x * (x * x)
    term = parser.parse_term('((x * x) = (x * (x * x)))')
    # single_test(term)

    # FAIL 4283 x * (x * y) = x * (y * x)
    term = parser.parse_term('((x * (x * y)) = (x * (y * x)))')
    # single_test(term)

    # 3257 x * x = x * (x * (x * x))
    term = parser.parse_term('((x * x) = (x * (x * (x * x))))')
    single_test(term)