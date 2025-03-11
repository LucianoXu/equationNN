from typing import Optional
from tqdm import tqdm
from .env import env, Scenario
from .proof import ProofTrace
import random

def gen_examples(scenario: Scenario, count: int, max_step: int) -> list[ProofTrace]:
    '''
    Generate a list of examples using a direct syntax fuzzer.
    '''
    traces : list[ProofTrace] = []
    ntok_machine = env.NextTokenMachine(scenario.alg, True)
    # push in 'x = x :'
    random_var = list(scenario.alg.signature.variables)[0]
    init_eq = env.parse_equation(f"{random_var} = {random_var}")
    assert init_eq is not None

    for _ in tqdm(range(count), desc="Generating examples"):
        # copy the next token machine
        eq = env.Equation(init_eq)

        # the proof trace
        trace : list[env.proof_step] = []

        for _ in range(random.randint(1, max_step)):

            # push the problem
            ntok_machine_copy = ntok_machine.copy()
            if not ntok_machine_copy.push_string(str(eq) + " : "):
                raise ValueError(f"Cannot push the equation into the next token machine: {eq}.")
            
            act_encodings = []

            # push the next tokens
            while ntok_machine_copy.state != env.NextTokenMachine.State.HALT:
                choices = ntok_machine_copy.valid_next_tokens
                if not choices:
                    raise ValueError(f"Cannot find the next token for the equation: {eq}.")
                token = random.choice(list(choices))
                if not ntok_machine_copy.push_token(token):
                    raise ValueError(f"Cannot push the token {token} to the next state machine:\n{ntok_machine_copy}.")
                act_encodings.append(token)

            # remove the last <EOS> token
            act_encodings = act_encodings[:-1]

            # get the equation
            if scenario.kernel.action_by_code(eq, scenario.tokenizer.decode(act_encodings)) != env.ACT_RESULT.SUCCESS:
                raise Exception(f"Cannot apply the action {scenario.tokenizer.decode(act_encodings)} to the equation {eq}.")

            # append the equation
            proof_step = env.parse_proof_step(ntok_machine_copy.finished_session)
            if proof_step is None:
                raise ValueError(f"Cannot parse the proof action: {ntok_machine_copy.finished_session}.")
            trace.append(proof_step)

        traces.append(ProofTrace(scenario, trace, eq))

    return traces
