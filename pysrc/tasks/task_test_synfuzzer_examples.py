import argparse
from ..env import env, Scenario
from ..syntax_fuzzer import gen_examples
from ..evaluation import test_intere_mp_args
import csv

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("test_synfuzzer_examples", help="Test the examples by the syntax fuzzer, and calculate the related metrics.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("-c", "--count", type=int, help="Number of examples to generate.", default=100)
    parser.add_argument("-m", "--max_step", type=int, help="Maximum step of the generation.", default=10)
    parser.add_argument("--state_len_limit", type=int, help="State length limit.", default=100)
    parser.add_argument("--context_length", type=int, help="Context length.", default=150)
    parser.add_argument("--timeout", type=float, default=5, help="Timeout for Vampire.")
    parser.add_argument("--vampire", type=str, help="Path to the vampire executable.", default="vampire")
    parser.add_argument("--print_trace", action="store_true", help="Print the generated traces.")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    traces = gen_examples(scenario, count=parsed_args.count, max_step=parsed_args.max_step, state_len=parsed_args.state_len_limit, context_len=parsed_args.context_length)

    # print the traces
    if parsed_args.print_trace:
        for i in range(parsed_args.count):
            print(traces[i])
            print("\n\n")
    
    examples = [trace.final_stt.eq for trace in traces]

    print(f"Generated {len(examples)} examples.")

    # test interestingness
    print("Testing interestingness...")
    intere_result = test_intere_mp_args(parsed_args.vampire, scenario, examples, timeout=parsed_args.timeout)

    # calculate the average interestingness
    total_intere = 0.
    for _, _, intere in intere_result:
        total_intere += intere
    print(f"Average interestingness: {total_intere / len(intere_result)}")

    results = []
    for i in range(len(traces)):
        results.append((str(examples[i]), intere_result[i][0], intere_result[i][1], intere_result[i][2]))

    if parsed_args.output:
        with open(parsed_args.output, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Equation", "Size", "Complexity", "Interestingness"])
            writer.writerows(results)

        print(f"Results saved to {parsed_args.output}.")







