# evaluate the ability of the model

from pyualg import Term
from proofkernel import solve_kernel_group
from tqdm import tqdm
BATCH_SIZE = 48

def eval(model, examples: list[Term], step_limit: int = 50, T: float = 1.0) -> float:
    '''
    Evaluate the model on a list of examples. The evaluation metric is the average length of the traces.
    '''
    progress_bar = tqdm(total=len(examples) // BATCH_SIZE if len(examples) % BATCH_SIZE == 0 else len(examples) // BATCH_SIZE + 1)

    total_length = 0

    for i in range(0, len(examples), BATCH_SIZE):
        batch_examples = examples[i:i+BATCH_SIZE]
        traces = solve_kernel_group(model, batch_examples, step_limit, T)
        total_length += sum(len(trace) for trace in traces)

        avg_length = total_length / (i+BATCH_SIZE if i+BATCH_SIZE < len(examples) else len(examples))
        progress_bar.desc = f"Evaluating({i+BATCH_SIZE}/{len(examples)}), Average Length: {avg_length:.2f}"
        progress_bar.update(1)

    return total_length / len(examples)

if __name__ == '__main__':
    # load the model
    from model import *

    model_checkpint = 'trained_parameters.pth'

    # load the model
    device = 'mps'
    SEQ_LEN = 96

    model_args = ModelArgs()
    model_args.max_seq_len = SEQ_LEN
    model = Transformer(ModelArgs(), device)
    model.load_state_dict(torch.load(model_checkpint, weights_only=True, map_location=device))

    # construct the examples    
    from data import full_path_examples
    examples = full_path_examples(1000, 5, 3)

    # evaluate the model
    print(eval(model, list(examples), 50, 0.3))

