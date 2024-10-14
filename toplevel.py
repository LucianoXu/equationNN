# the toplevel interface for the proof kernel

from typing import Callable
from proofkernel import *
from tokenizer import *
from model import generate

def human_agent(kernel: ProofKernel) -> str:
    '''
    a human agent that interacts with the proof kernel
    '''
    return input()

def get_model_agent(model, max_len: int = 256, T: float = 0.6) -> Callable[[ProofKernel], str]:
    '''
    create the model agent with the model
    '''
    def model_agent(kernel: ProofKernel) -> str:
        # get the model input
        term = kernel.equation
        return generate(model, str(term) + " : ", max_len, T)

    return model_agent


def prove(kernel: ProofKernel, agent: Callable[[ProofKernel], str]):
    '''
    prove the equation in the proof kernel
    UI design:

    Command Language:
        cmd ::= L2R pos | R2L pos
        pos ::= 0 | 1 | pos 0 | pos 1

    Rules:
        L2R: (X * Y) -> ((Y * Y) * X)
        R2L: ((Y * Y) * X) -> (X * Y)

    Step: # | Last Reward: # | Total Reward: # | Equation :
        Action: #
    ...
    Step: # | Last Reward: # | Total Reward: # | Equation : | STOPPED
    '''
    # print the hint
    print(
'''

Command Language:
    cmd ::= L2R pos | R2L pos
    pos ::= 0 | 1 | pos 0 | pos 1

Rules:
    L2R: (X * Y) -> ((Y * Y) * X)
    R2L: ((Y * Y) * X) -> (X * Y)

'''
    )
    last_reward = 0.
    total_reward = 0.
    while True:
        prompt = f'Step: {kernel.step_count}\t| Last Reward: {last_reward:.2f}\t| Total Reward: {total_reward:.2f}\t| {kernel.equation.sig_str(kernel.sig)} : '
        print(prompt)

        if kernel.is_stopped:
            print('\tSTOPPED')
            break

        print(f'\tAction: ', end='')

        # input the action
        action = agent(kernel)
        if agent != human_agent:
            print(action)

        last_reward = kernel.step(action)
        total_reward += last_reward

if __name__ == '__main__':
    # run the toplevel prover and have fun!
    while True:
        path = gen_example(5, 3)
        term = path.current

        # skip the trivial case
        if term.args[0] == term.args[1]:
            continue

        kernel = ProofKernel(term)
        prove(kernel, human_agent)
