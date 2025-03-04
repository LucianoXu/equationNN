import argparse
from ..syntax_fuzzer import gen_examples
from ..env import Scenario

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("syn_gen_examples", help="Generate examples using syntax fuzzer.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to generate.")
    parser.add_argument("--max_step", type=int, default=10, help="Maximum step of the generation.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    examples = gen_examples(scenario, parsed_args.num_examples, parsed_args.max_step)
    for example in examples:
        print(example)
