from typing import Optional
from tqdm import tqdm
from .env import env, Scenario
from .problems import ProofTrace, GenProblemSet, GenProblemFactory, UniqueExamplePipe, LengthRequestPipe, MultiProcessPipe
import random

class SyntaxFuzzerFactory(GenProblemFactory):
    def __init__(self, scenario: Scenario, max_step: int, state_len: int, context_len: int):
        super().__init__(scenario)
        self.max_step = max_step
        self.state_len = state_len
        self.context_len = context_len

    def spawn(self, count: int, verbose = True) -> GenProblemSet:
        '''
        Generate a list of examples using a direct syntax fuzzer. The fuzzer will randomly choose valid actions and apply them to the equation.

        If the intermediate steps exceed the state length or the whole length exceeds the context length, the particular example rewriting will be terminated and the result will be appended to the list.
        '''
        traces : list[ProofTrace] = []
        ntok_machine = env.NextTokenMachine(self.scenario.alg, True)
        random_vars = list(self.scenario.alg.signature.variables)
        
        progress = tqdm(range(count), desc="Syntax Fuzzer examples", leave=False)
        for _ in progress:

            # randomly choose a variable, push in 'x = x :'
            random_var = random.choice(random_vars)
            init_stt = env.parse_proof_state(f"<STT> {random_var} = {random_var} </STT>")
            assert init_stt is not None

            # copy the next token machine
            stt = env.proof_state(init_stt)

            # the proof trace
            trace : list[env.proof_step] = []

            for _ in range(random.randint(1, self.max_step)):

                # push the problem
                ntok_machine_copy = ntok_machine.copy()
                if not ntok_machine_copy.push_string(str(stt)):
                    raise ValueError(f"Cannot push the state into the next token machine: {stt}.")
                
                # check the state length
                state_encodings = ntok_machine_copy.encodings
                if len(state_encodings) > self.state_len:
                    break
                
                act_encodings = []

                # push the next tokens
                while ntok_machine_copy.state != env.NextTokenMachine.State.HALT:
                    choices = ntok_machine_copy.valid_next_tokens
                    if not choices:
                        raise ValueError(f"Cannot find the next token for the state: {stt}.")
                    token = random.choice(list(choices))
                    if not ntok_machine_copy.push_token(token):
                        raise ValueError(f"Cannot push the token {token} to the next state machine:\n{ntok_machine_copy}.")
                    act_encodings.append(token)

                # check the whole length
                if len(state_encodings) + len(act_encodings) > self.context_len:
                    break

                # get the new_state
                if self.scenario.kernel.action_by_code(stt, self.scenario.tokenizer.decode(act_encodings)) != env.ACT_RESULT.SUCCESS:
                    raise Exception(f"Cannot apply the action {self.scenario.tokenizer.decode(act_encodings)} to the state {stt}.")

                # append the equation
                proof_step = env.parse_proof_step(ntok_machine_copy.input)
                if proof_step is None:
                    raise ValueError(f"Cannot parse the proof action: {ntok_machine_copy.input}.")
                    
                trace.append(proof_step)

            traces.append(ProofTrace(self.scenario, trace, stt))

        return GenProblemSet(self.scenario, traces)
    
    def __str__(self) -> str:
        return f"SyntaxFuzzerFactory(max_step={self.max_step}, state_len={self.state_len}, context_len={self.context_len})"


def standard_syntax_fuzzer(scenario: Scenario, max_step: int, state_len: int, context_len: int, nproc: int = 4) -> GenProblemFactory:
    '''
    Create a standard syntax fuzzer.
    '''
    factory = SyntaxFuzzerFactory(scenario, max_step, state_len, context_len)
    factory = LengthRequestPipe(factory, max_step)
    factory = MultiProcessPipe(factory, nproc)
    factory = UniqueExamplePipe(factory)
    return factory
