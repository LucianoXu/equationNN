import argparse
from ..env import env, Scenario
from ..utilis import parse_examples

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("ntok_machine", help="The next token machine.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    ntok_machine = env.NextTokenMachine(scenario.alg)

    while ntok_machine.state != env.NextTokenMachine.State.HALT:
        print(ntok_machine)
        next_input = input("Input: ")
        ntok_machine.push_string(next_input)

    print(ntok_machine)
    print("Halted.")