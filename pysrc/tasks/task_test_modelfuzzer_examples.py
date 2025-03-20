import argparse
import torch
from ..env import env, Scenario
from ..utilis import parse_examples
from ..evaluation import test_intere_mp_args
import csv
from ..model import Llama3, MediumArgs
from ..rl import gen_group
from elab import ELab
from ..problems import ProofTrace

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("test_modelfuzzer_examples", help="Test the interestingness of model generated examples.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser.add_argument("-c", "--count", type=int, default=100, help="Number of examples to generate.")
    parser.add_argument("-m", "--max_steps", type=int, default=5, help="Maximum steps of the generation.")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size.")
    parser.add_argument("-t", "--temperature", type=float, default=0.6, help="Temperature for model in RL.")
    parser.add_argument("--timeout", type=float, default=5, help="Timeout for Vampire.")
    parser.add_argument("--state_len_limit", type=int, default=100, help="State length limit.")
    parser.add_argument("--context_length", type=int, default=150, help="Context length.")
    parser.add_argument("--vampire", type=str, help="Path to the vampire executable.", default="vampire")
    parser.add_argument("--print_trace", action="store_true", help="Print the generated traces.")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    # load the model and generate examples
    args = MediumArgs(vocab_size=scenario.tokenizer.get_vocab_size(), context_length=parsed_args.context_length)
    device = parsed_args.device

    gen_model = Llama3(
        model_args = args,
        device = device
    )

    gen_lab = ELab(
        parsed_args.ckpt,
        version_name='latest',
        data = {
            'model': gen_model
        },
        device=device
    )

    gen_model.eval()

    all_gen_traces = []

    with torch.no_grad():
        for _ in range(parsed_args.count // parsed_args.batch_size + 1):
            gen_traces = gen_group(
                gen_model, scenario, 
                parsed_args.batch_size, parsed_args.max_steps, 
                parsed_args.state_len_limit, parsed_args.context_length, parsed_args.temperature)
            
            for trace in gen_traces:
                all_gen_traces.append(trace)

        
    traces = [ProofTrace.from_acts(scenario, trace.init_eq, [step[0] for step in trace.steps]) for trace in all_gen_traces][:parsed_args.count]


    # print the traces
    if parsed_args.print_trace:
        for i in range(parsed_args.count):
            print(traces[i])
            print("\n\n")
    
    examples = [trace.final_stt.eq for trace in traces]
    
    print(f"Generated {len(examples)} examples.")

    # test the interestingness
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