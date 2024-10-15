# evaluate the ability of the model

from pyualg import Term
from proofkernel import solve_kernel_group
from tqdm import tqdm

from utilities import *

def eval(model, examples: list[Term], step_limit: int = 50, T: float = 1.0, batch_size: int = 48) -> float:
    '''
    Evaluate the model on a list of examples. The evaluation metric is the average length of the traces.
    '''
    progress_bar = tqdm(total=len(examples) // batch_size if len(examples) % batch_size == 0 else len(examples) // batch_size + 1)

    total_length = 0

    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i:i+batch_size]

        with torch.no_grad():
            traces = solve_kernel_group(model, batch_examples, step_limit, T)

        total_length += sum(len(trace) for trace in traces)

        avg_length = total_length / (i+batch_size if i+batch_size < len(examples) else len(examples))
        progress_bar.desc = f"Evaluating({i+batch_size}/{len(examples)}), Average Length: {avg_length:.2f}"
        progress_bar.update(1)

    return total_length / len(examples)

if __name__ == '__main__':
    # load the model
    from model import *
    from small_args import SmallArgs

    model = load_model('small_rl.pth', SmallArgs(), 'mps')
    model.eval()

    # construct the examples    
    from data import full_path_examples
    examples = full_path_examples(500, 5, 3)

    # evaluate the model
    print(eval(model, list(examples), 30, 0.3, batch_size=500))

