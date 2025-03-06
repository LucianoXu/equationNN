from .env import env, Scenario

def parse_examples(scenario: Scenario, file: str):
    with open(file) as f:
        lines = f.readlines()

    # remove those lines that start with %
    lines = [line for line in lines if not line.startswith("%")]

    # remove empty lines
    lines = [line for line in lines if line.strip()]

    examples = []
    for line in lines:
        eq = env.parse_equation(line)
        if eq is None:
            raise ValueError(f"Cannot parse the equation: {line}")

        # check whether the equation is valid
        if not (scenario.sig.term_valid(eq.lhs) and scenario.sig.term_valid(eq.rhs)):
            raise ValueError(f"The equation {eq} is not valid in the signature.")

        examples.append(eq)
        
    return examples