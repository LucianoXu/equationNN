# define the proof kernel. It is also the environment simulator for reinforcement learning.
from pyualg import parser
from scenario import *
from tokenizer import *
from generate import batch_generation
from tqdm import tqdm
import torch

class ProofKernel:
    def __init__(self, equation: Term):
        '''
        A proof kernel for a given equation. Notice that the equation should not be already proved.
        '''
        self.sig = signature
        self.equation = equation
        self.step_count = 0
        self.stopped = False

    @property
    def state(self) -> str:
        '''
        return the current state of the environment
        '''
        return self.equation.sig_str(self.sig) + " : "

    @property
    def is_stopped(self) -> bool:
        '''
        return whether the environment is terminated
        '''
        return self.stopped

    def step(self, action: str) -> float:
        '''
        take a step in the environment, change the current state and return the reward
        action: the string representation of the rewriting command, only the part after ':' and before '<EOS>'.
        '''
        self.step_count += 1

        # parse the action
        encoding = tok_encode(action)

        # try to apply the command. any exception will be caught and return
        try:
            # find the rule in RULE_NAME_INV
            if id2token[encoding[0]] not in RULE_NAMES_INV:
                return -10.
            rule = RULE_NAMES_INV[id2token[encoding[0]]]

            # find the first token2id['{'] element in encoding
            subst_start_id = encoding.index(token2id['{'])

            
            pos = tuple(int(id2token[id]) for id in encoding[1:subst_start_id])

            subst = parser.parse_subst(tok_decode(encoding[subst_start_id:]))

            res = self.equation.apply_at(rule, self.sig, pos, subst, forbidden_heads)
        except:
            return -10.

        # if the command is not applicable, return
        if res is None:
            return -10.
        
        # if the command is applicable, conduct the command
        self.equation = res[0]
        
        # check whether the equation is proved, return the corresponding reward
        if self.equation.args[0] == self.equation.args[1]:
            self.stopped = True
            return 0.
        
        
        # default reward
        return -1.





def solve_kernel_group(model, examples: list[Term], step_limit: int = 50, context_length: int = 256, T: float = 1.0) -> list[list[tuple[str, torch.Tensor, float]]]:
    '''
    Solve a group of proof kernels for parallel training and evaluation.

    Args:
        examples: a list of examples, each example is a term.
        step_limit: the maximum number of steps for each example.

    Returns:
        A list of traces. Each trace is a list of tuples (action, log probability, reward).
        Notice that an action is a complete response from the agent, not a single token.
    '''
    
    kernels = [ProofKernel(eq) for eq in examples]

    # proceed the examples
    traces: list[list[tuple[str, torch.Tensor, float]]] = [[] for _ in range(len(kernels))]

    remaining_number = len(kernels)
    progress_bar = tqdm(total=step_limit)
    for _ in range(step_limit):
        # update the description of the progress bar
        progress_bar.desc = f"Solving Kernels({remaining_number}/{remaining_number})"

        # generate the batch
        batch = [kernel.state for kernel in kernels]
        actions, log_probs = batch_generation(model, batch, context_length, T)

        remaining_number = 0
        # proceed the kernels and record the action, probability and reward
        for i, kernel in enumerate(kernels):
            # if the kernel is already stopped, skip
            if kernel.is_stopped:
                continue

            action = actions[i]
            log_prob = log_probs[i]
            reward = kernel.step(action)
            traces[i].append((action, log_prob, reward))

            remaining_number += 1

        progress_bar.update(1)

        # check whether all the kernels are stopped
        if remaining_number == 0:
            progress_bar.close()
            break

    return traces



def beam_search(model, problem: Term, beam_number: int = 20, step_limit: int = 50, context_length: int = 256, T: float = 1.0) -> Optional[list[str]]:
    '''
    Solve one problem by beam search.

    Args:
        problem (Term): a term of the problem to be solved
        beam_number (int): the number of beams
        step_limit (int): the maximum number of steps for searching

    Returns:
        A list of traces. Each trace is a list of tuples (action, log probability, reward).
        Notice that an action is a complete response from the agent, not a single token.
    '''
    
    kernels = [ProofKernel(problem) for _ in range(beam_number)]

    # proceed the examples
    traces: list[list[str]] = [[] for _ in range(len(kernels))]

    remaining_number = len(kernels)
    progress_bar = tqdm(total=step_limit)
    for _ in range(step_limit):
        # update the description of the progress bar
        progress_bar.desc = f"Solving Kernels({remaining_number}/{remaining_number})"

        # generate the batch
        batch = [kernel.state for kernel in kernels]
        actions, _ = batch_generation(model, batch, context_length, T)

        remaining_number = 0
        # proceed the kernels and record the action, probability and reward
        for i, kernel in enumerate(kernels):
            # if the kernel is already stopped, skip
            if kernel.is_stopped:
                return traces[i]

            action = actions[i]
            kernel.step(action)
            traces[i].append(action)

            remaining_number += 1

        progress_bar.update(1)

    return None


if __name__ == '__main__':
    pass
    # from scenario import parser
    # from model import *
    # term1 = parser.parse_term('((x * x) = (x * (x * x)))')
    # term2 = parser.parse_term('((x * x) = (x * (x * (x * x))))')

    # model_checkpint = 'trained_parameters.pth'

    # # load the model
    # device = 'mps'
    # SEQ_LEN = 96

    # model_args = ModelArgs()
    # model_args.max_seq_len = SEQ_LEN
    # model = Transformer(ModelArgs(), device)
    # model.load_state_dict(torch.load(model_checkpint, weights_only=True, map_location=device))

    # traces = solve_kernel_group(model, [term1, term2, term1], 50, 0.3)
    # for trace in traces:
    #     print(len(trace))
    #     print(trace)

