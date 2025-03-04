from .model import *
from ..env import env, Scenario

def constrained_sample(
    logits: torch.Tensor, 
    valid_next_tokens: list[set[int]], 
    T: float = 1.0
) -> tuple[list[int], torch.Tensor]:
    '''
    Sample the next tokens from the logits, with temperature scaling and constraint enforcement.

    Parameters:
    - logits: (batch_size, vocab_size), the logit values for each token
    - valid_next_tokens: list of sets, where each set contains valid token indices for the corresponding batch item
    - T: the temperature coefficient to control randomness (default is 1.0)
    - num_samples: the number of tokens to sample per batch item (default is 1)

    Returns:
    - A list of lists, where each inner list contains the sampled tokens for the corresponding batch item.
    '''
    # Scale the logits by the temperature
    logits = logits / T

    batch_size = logits.shape[0]
    sampled_tokens = []
    probabilities = []

    for i in range(batch_size):
        valid_tokens = list(valid_next_tokens[i])  # Get valid tokens for batch entry i
        filtered_logits = logits[i, valid_tokens]  # Extract logits of valid tokens

        # Convert logits to probabilities
        sampling_probabilities = F.softmax(filtered_logits, dim=-1)

        # Sample num_samples tokens
        sampled_indice = int(torch.multinomial(sampling_probabilities, num_samples=1, replacement=True).item())

        # Map back to original token indices
        sampled_tokens.append(valid_tokens[sampled_indice])

        # calculate the probabilities
        probabilities.append(sampling_probabilities[sampled_indice])


    return sampled_tokens, torch.stack(probabilities)


def generate(model, scenario: Scenario, code: str, max_len: int = 256, T: float = 1.0) -> str:
    '''
    Generate code using the model, with temperature scaling.
    
    Parameters:
    - model: the language model used for code generation
    - code: the initial code to prompt the generation
    - max_len: the maximum length of the generated sequence (default is 256)
    - T: the temperature coefficient to control randomness (default is 1.0)
    '''
    device = model.device

    # prepare the tokenizer and the next token machine
    ntok_machine = env.NextTokenMachine(scenario.alg)

    # try to push the encoded code to the next token machine
    if not ntok_machine.push_string(code):
        raise Exception("Invalid code pushed to the next token machine: " + code)

    encodings_tensor = torch.tensor([ntok_machine.encodings], device=device)

    output = []
    
    # autoregressive generation
    with torch.no_grad():
        for i in range(max_len):

            # get the possible next tokens and enforece the constraints
            next_tokens = ntok_machine.valid_next_tokens
            if len(next_tokens) == 0:
                raise Exception("No valid next tokens found. Generated code: " + scenario.tokenizer.decode(output))
            
            elif len(next_tokens) == 1:
                # if there is only one possible next token, no need to sample
                next_token = next_tokens.pop()
            
            else:
                # sample the next token
                logits = model(encodings_tensor)
                logits = logits[:, -1, :]
                next_token = constrained_sample(logits, [next_tokens], T)[0][0]

            if not ntok_machine.push_token(next_token):
                raise Exception("Invalid token pushed to the next token machine.")
            output.append(next_token)

            if output[-1] == scenario.EOS_ID:
                # remove the <EOS> token
                output = output[:-1]
                break
            encodings_tensor = torch.cat((encodings_tensor, torch.tensor([[output[-1]]], device = device)), dim=1)

    return scenario.tokenizer.decode(output)


def batch_predict(model, scenario: Scenario, machines: list[env.NextTokenMachine], context_length: int = 256, T: float = 1.0) -> tuple[list[int], torch.Tensor]:
    '''
    Generate the next tokens for each beam using the model, with temperature scaling.

    If one beam is too long, the function will generate according to the last tokens of context length.
    It does not modify the machines' state.
    
    Return the next tokens and the probabilities of the next tokens.
    '''
    
    device = model.device
    beams = [machine.encodings for machine in machines]
    max_len = min(max(len(beam) for beam in beams), context_length)
    batch_size = len(beams)

    # calculate the padding lengths
    pad_seq_lens = [max(0, max_len - len(beam)) for beam in beams]

    # padding at the beginning of the beams
    padded_beams = [[scenario.PAD_ID] * pad_seq_lens[i] + beams[i][-context_length:] for i in range(batch_size)]

    encoding = torch.tensor(padded_beams, device=device)

    # autoregressive generation
    logits = model(encoding, None, pad_seq_lens)
    logits = logits[range(batch_size), -1, :]

    return constrained_sample(logits, [machine.valid_next_tokens for machine in machines], T)



def batch_generation(model, scenario: Scenario, beams: list[str]|list[list[int]]|list[env.NextTokenMachine], context_length: int = 256, T: float = 1.0) -> tuple[list[str], torch.Tensor]:
    '''
    Generate the output for each beam using the model, with temperature scaling.

    Return the output results (without <EOS>) and the log probabilities of the output results.
    '''

    # prepare the next token machines
    if isinstance(beams[0], str):
        machine = env.NextTokenMachine(scenario.alg)
        
        machines = {i:machine.copy() for i,beam in enumerate(beams)}

        for i, beam in enumerate(beams):
            assert isinstance(beam, str)
            if not machines[i].push_string(beam):
                raise Exception("Invalid code pushed to the next token machine: " + beam)
    elif isinstance(beams[0], list):
        machine = env.NextTokenMachine(scenario.alg)
        
        machines = {i:machine.copy() for i,beam in enumerate(beams)}

        for i, beam in enumerate(beams):
            assert isinstance(beam, list)
            if not machines[i].push_encodings(beam):
                raise Exception("Invalid code pushed to the next token machine: " + str(beam))
    else:
        machines = {i:beam for i,beam in enumerate(beams)}  # type: ignore
        machines : dict[int, env.NextTokenMachine]
    
    input_lens = {i:len(machines[i].encodings) for i in machines}

    outputs = [""] * len(beams)
    acc_log_probs = torch.zeros(len(beams), device = model.device)

    # while there are still beams to predict
    while len(machines) > 0:
        indices = list(machines.keys())
        next_tokens, probabilities = batch_predict(
            model, scenario, [machines[i] for i in machines], context_length, T)
        log_probabilities = torch.log(probabilities)

        # iterate through the current beams
        for i in range(len(indices)):

            # get the real index in the batch
            idx = indices[i]

            # check if the beam is too long
            if len(machines[idx].encodings) >= context_length:
                outputs[idx] = scenario.tokenizer.decode(machines[idx].encodings[input_lens[idx]:])
                del machines[idx]

            else:
                # update the log probabilities
                acc_log_probs[idx] += log_probabilities[i]

                # finish the beams that predict <EOS>
                if next_tokens[i] == scenario.EOS_ID:
                    outputs[idx] = scenario.tokenizer.decode(machines[idx].encodings[input_lens[idx]:])
                    del machines[idx]


                # update the beams that are not finished
                else:
                    if not machines[idx].push_token(next_tokens[i]):
                        raise Exception(f"Invalid token {next_tokens[i]} pushed to the next token machine. Machine state:\n{machines[idx].state}")

    return outputs, acc_log_probs


if __name__ == "__main__":
    pass
    
