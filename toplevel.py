# the toplevel interface for the proof kernel

from proofkernel import *

def prove(kernel: ProofKernel):
    '''
    prove the equation in the proof kernel
    UI design:

    Command Language:
        cmd ::= L2R pos | R2L pos
        pos ::= 0 | 1 | pos 0 | pos 1

    Rules:
        L2R: (X * Y) -> ((Y * Y) * X)
        R2L: ((Y * Y) * X) -> (X * Y)

    Step # | Last Reward # | Total Reward # | Equation : cmd
    ...
    Step # | Last Reward # | Total Reward # | Equation | STOPPED
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
        print(prompt, end='')

        if kernel.is_stopped:
            print('\t| STOPPED')
            break

        # input the action
        action = input()

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
        prove(kernel)
