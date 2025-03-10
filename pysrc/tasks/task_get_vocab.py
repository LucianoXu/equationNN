import argparse
from ..env import env, Scenario
from ..utilis import parse_examples

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("get_vocab", help="Get the vocabulary for the algebra.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    print("Vocabulary:")
    print(scenario.tokenizer.vocab)