import argparse
from ..env import Scenario
from ..utilis import parse_examples

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("parse_examples", help="Parse examples from a file.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("example_file", type=str, help="Path to the example file.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    examples = parse_examples(scenario, parsed_args.example_file)
    for example in examples:
        print(example)

    print(f"{len(examples)} examples parsed.")