import argparse
from ..syntax_fuzzer import gen_examples
from ..env import Scenario, env

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("gen_valid_check", help="Check the validity of randomly generated sessions.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("-c", "--count", type=int, default=100, help="Number of examples to generate.")
    parser.add_argument("-m", "--max_step", type=int, default=10, help="Maximum step of the generation.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    while True:
        traces = gen_examples(scenario, parsed_args.count, parsed_args.max_step)
        for trace in traces:
            for step in trace.trace:
                print(step)
                if not env.check_action(scenario.kernel, str(step)):
                    raise Exception("Invalid action")