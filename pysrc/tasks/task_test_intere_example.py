import argparse
from ..env import env, Scenario
from ..utilis import parse_examples
from ..evaluation import test_intere_mp_args
import csv

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("test_intere_example", help="Test the interestingness of example examples.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("example_file", type=str, help="Path to the example file.")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file.", default="output.csv")
    parser.add_argument("--vampire", type=str, help="Path to the vampire executable.", default="vampire")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    examples = parse_examples(scenario, parsed_args.example_file)

    result = test_intere_mp_args(parsed_args.vampire, scenario, examples)

    with open(parsed_args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Size", "Complexity", "Interestingness"])
        writer.writerows(result)

    print(f"Results saved to {parsed_args.output}.")



