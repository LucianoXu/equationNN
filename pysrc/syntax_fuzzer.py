from typing import Optional
from tqdm import tqdm
from .env import env, Scenario
import random

def gen_examples(scenario: Scenario, count: int, max_step: int) -> list[env.Equation]:
    '''
    Generate a list of examples using a direct syntax fuzzer.
    '''
    examples : list[env.Equation] = []
    ntok_machine = env.NextTokenMachine(scenario.alg)
    # push in 'x = x :'
    random_var = list(scenario.alg.signature.variables)[0]
    init_eq = env.parse_equation(f"{random_var} = {random_var}")
    assert init_eq is not None

    for _ in tqdm(range(count), desc="Generating examples"):
        # copy the next token machine
        eq = env.Equation(init_eq)
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
        examples.append(eq)

    return examples

if __name__ == '__main__':
    alg_code = '''
    [function]
    & : 2
    | : 2
    ~ : 1

    [variable]
    x y z u v w

    [axiom]
    (AX1) &(x y) = &(y x)
    (AX2) |(x y) = |(y x)
    (AX3) &(x &(y z)) = &(&(x y) z)
    (AX4) |(x |(y z)) = |(|(x y) z)
    (AX5) &(x |(x y)) = x
    (AX6) |(x &(x y)) = x
    (AX7) ~(~(x)) = x
    (AX8) ~(&(x y)) = |(~(x) ~(y))
    (AX9) ~(|(x y)) = &(~(x) ~(y))
    (OML) |(x y) = |(&(|(x y) x) &(|(x y) ~(x)))
    '''
    scenario = Scenario(alg_code)

    # while True:
    #     next_token_machine = env.NextTokenMachine(scenario.alg)
    #     # next_token_machine.push_string("|(~(~(v)) &(~(~(v)) ~)) = |(&(v ~(v)) v)")
    #     next_token_machine.push_string("|(~(~(v)) &(~(~(v)) ~")
    #     print(next_token_machine)
    #     print(next_token_machine.valid_next_tokens)
    #     if (len(next_token_machine.valid_next_tokens) > 1):
    #         break
    # exit(0)

    # eq = env.parse_equation("u = u")
    # scenario.kernel.action_by_code(eq, "SUBST u ~ ( w )")
    # print(eq)
    # exit(0)


    examples = gen_examples(scenario, 100, 10)
    for example in examples:
        print(example)
