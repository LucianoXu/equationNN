import argparse
from ..syntax_fuzzer import SyntaxFuzzerFactory
from ..env import Scenario

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("syn_gen_examples", help="Generate examples using syntax fuzzer.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("-c", "--count", type=int, default=100, help="Number of examples to generate.")
    parser.add_argument("-m", "--max_step", type=int, default=10, help="Maximum step of the generation.")
    parser.add_argument("--state_len_limit", type=int, default=100, help="State length limit.")
    parser.add_argument("--context_length", type=int, default=150, help="Context length.")
    parser.set_defaults(func=task)

    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    example_factory = SyntaxFuzzerFactory(
        scenario=scenario,
        max_step=parsed_args.max_step,
        state_len=parsed_args.state_len_limit,
        context_len=parsed_args.context_length
    )

    problem_set = example_factory.spawn(parsed_args.count)
    for trace in problem_set.traces:
        print(trace)
        print("\n\n")

    print(f"{parsed_args.count} examples generated.")