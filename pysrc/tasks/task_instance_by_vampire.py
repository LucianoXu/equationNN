import argparse
from ..env import env, Scenario
from ..ext_solver import vampire_solve

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("instance_by_vampire", help="Try to solve the instance by Vampire.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("--vampire", type=str, default='vampire', help="Path to Vampire.")
    parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout for Vampire.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    while True:
        example_code = input("Input example: ")
        problem = env.parse_equation(example_code)

        if problem is None:
            print("Invalid input.")
            return
        
        print("Vampire solving...")
        result = vampire_solve(parsed_args.vampire, scenario, problem, parsed_args.timeout)

        print(result)