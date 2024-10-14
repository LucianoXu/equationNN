# define the proof kernel. It is also the environment simulator for reinforcement learning.

from scenario import *
from tokenizer import *

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

        # extract the command
        if encoding[0] == token2id['L2R']:
            rule = r_L2R
        elif encoding[0] == token2id['R2L']:
            rule = r_R2L
        else:
            return -1.
        
        pos = tuple(int(id2token[id]) for id in encoding[1:])

        # try to apply the command. any exception will be caught and return -1
        try:
            res = self.equation.apply_at(rule, self.sig, pos)
        except:
            return -1.

        # if the command is not applicable, return -1
        if res is None:
            return -1.
        
        # if the command is applicable, conduct the command
        self.equation = res
        
        # check whether the equation is proved, return the corresponding reward
        if self.equation.args[0] == self.equation.args[1]:
            self.stopped = True
            return 1./self.step_count
        
        else:
            return 0.
