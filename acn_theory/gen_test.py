from acn_theory.scenario import *
from acn_theory.model import *

def gen_test():

    device = 'cpu'
    SEQ_LEN = 96
    BATCH_SIZE = 8
    DATA_LEN = 1000
    MAX_STEPS = 8


    # Step 2
    model_args = ModelArgs()
    model_args.max_seq_len = SEQ_LEN
    model = Transformer(ModelArgs(), device)
    model.load_state_dict(torch.load('acn_theory/trained_parameters.pth', weights_only=True, map_location=device))


    code = "<SOS>(+ (+ (~ d) (~ (~ d))) (+ (+ a (~ a) 0))<ACT>"
    # code = "<SOS>(+ (+ (~ d) (~ (~ d))) (+ (+ a (~ a"
    # code = "<SOS>(+ (+ (~ (+ a b) (~ (~ (+ b a)))) (+ a (~ a))<ACT>"
    encoding = torch.tensor([encode_example(code)])

    output = []
    # autoregressive generation
    with torch.no_grad():
        model.eval()
        for i in range(SEQ_LEN):
            logits = model(encoding)
            output.append(logits[0][-1].argmax(dim=-1).item())
            encoding = torch.cat((encoding, torch.tensor([[output[-1]]])), dim=1)
            if output[-1] == TOKENS['<EOS>']:
                break

    print(decode_example(output))