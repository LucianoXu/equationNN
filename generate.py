from model import *

PADDING_ID = token2id['<PAD>']
EOS_ID = token2id['<EOS>']

def generate(model, code: str, max_len: int = 256, T: float = 1.0) -> str:
    '''
    Generate code using the model, with temperature scaling.
    
    Parameters:
    - model: the language model used for code generation
    - code: the initial code to prompt the generation
    - max_len: the maximum length of the generated sequence (default is 256)
    - T: the temperature coefficient to control randomness (default is 1.0)
    '''
    device = model.device
    encoding = torch.tensor([tok_encode("<SOS> " +code)], device=device)

    output = []
    
    # autoregressive generation
    with torch.no_grad():
        for i in range(max_len):
            logits = model(encoding)

            # Scale the logits by the temperature
            logits = logits[0, -1, :] / T
            
            # Convert logits to probabilities using softmax
            probabilities = F.softmax(logits, dim=-1)

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            output.append(next_token)

            if output[-1] == token2id['<EOS>']:
                # remove the <EOS> token
                output = output[:-1]
                break
            encoding = torch.cat((encoding, torch.tensor([[output[-1]]], device = device)), dim=1)

    return tok_decode(output)

def batch_predict(model, beams: list[list[int]], T: float = 1.0) -> tuple[list[int], torch.Tensor]:
    '''
    Generate the next tokens for each beam using the model, with temperature scaling.

    Return the next tokens and the probabilities of the next tokens.
    '''
    
    device = model.device
    idx_beams_predict = [len(beam)-1 for beam in beams]
    max_len = max(idx_beams_predict) + 1
    batch_size = len(beams)

    # padding at the end of the beams
    padding_beams = [beam + [PADDING_ID] * (max_len - len(beam)) for beam in beams]
    encoding = torch.tensor(padding_beams, device=device)

    # autoregressive generation
    logits = model(encoding)
    logits = logits[range(batch_size), idx_beams_predict, :] / T

    probabilities = F.softmax(logits, dim=-1)

    # sample the next token from the probability distribution
    next_tokens = torch.multinomial(probabilities, num_samples=1).view(batch_size,).tolist()
    predict_probabilities = probabilities[range(batch_size), next_tokens]

    return next_tokens, predict_probabilities


def batch_generation(model, beams: list[str], T: float = 1.0) -> tuple[list[str], torch.Tensor]:
    '''
    Generate the output for each beam using the model, with temperature scaling.

    Return the output results (without <EOS>) and the log probabilities of the output results.
    '''
    
    input_ids = {i:tok_encode("<SOS> " + beam) for i,beam in enumerate(beams)}
    input_lens = {i:len(input_ids[i]) for i in input_ids}

    outputs = [""] * len(beams)
    log_probs = torch.zeros(len(beams), device = model.device)

    # while there are still beams to predict
    while len(input_ids) > 0:
        indices = list(input_ids.keys())
        next_tokens, predict_probabilities = batch_predict(model, [input_ids[i] for i in input_ids], T)
        predict_probabilities = torch.log(predict_probabilities)

        # iterate through the current beams
        for i in range(len(indices)):
            idx = indices[i]
            # update the log probabilities
            log_probs[idx] += predict_probabilities[i]

            # finish the beams that predict <EOS>
            if next_tokens[i] == EOS_ID:
                outputs[idx] = tok_decode(input_ids[idx][input_lens[idx]:])
                del input_ids[idx]

            # update the beams that are not finished
            else:
                input_ids[idx].append(next_tokens[i])

    return outputs, log_probs


if __name__ == "__main__":
    from scenario import parser
    code1 = '((x * x) = (x * (x * x))) :'
    code2 = '((x * (x * y)) = (x * (y * x))) :'
    code3 = '((x * x) = (x * (x * (x * x)))) :'

    model_checkpint = 'trained_parameters.pth'

    # load the model
    device = 'mps'
    SEQ_LEN = 96

    model_args = ModelArgs()
    model_args.max_seq_len = SEQ_LEN
    model = Transformer(ModelArgs(), device)
    model.load_state_dict(torch.load(model_checkpint, weights_only=True, map_location=device))


    print(batch_generation(model, [code1, code2, code3], 0.3))

    
    
