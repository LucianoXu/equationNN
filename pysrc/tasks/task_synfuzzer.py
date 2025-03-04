import argparse
from ..env import env, Scenario
from ..syntax_fuzzer import gen_examples
from ..evaluation import test_intere_mp_args
import csv

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("synfuzzer", help="Generate examples using syntax fuzzer.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("-c", "--count", type=int, help="Number of examples to generate.", default=100)
    parser.add_argument("-m", "--max_step", type=int, help="Maximum step of the generation.", default=10)
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    examples = gen_examples(scenario, 100, 10)

    for example in examples:
        print(example)

    print(f"{parsed_args.count} examples generated.")


