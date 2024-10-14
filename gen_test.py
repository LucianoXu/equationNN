
from model import *
from toplevel import *

if __name__ == '__main__':
    # load the model
    device = 'cpu'
    SEQ_LEN = 96

    model_args = ModelArgs()
    model_args.max_seq_len = SEQ_LEN
    model = Transformer(ModelArgs(), device)
    model.load_state_dict(torch.load('trained_parameters.pth', weights_only=True, map_location=device))

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