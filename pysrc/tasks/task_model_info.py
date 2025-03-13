
import argparse
from ..rl import *
import elab

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("model_info", help="Print the model information.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser.add_argument("--modelargs", type=str, default='medium', help="Model arguments.")
    parser.add_argument("--vocab_size", type=int, default=20, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=150, help="Context length.")
    parser.set_defaults(func=task)


def task(parsed_args: argparse.Namespace):

    model_args = modelargs_dict[parsed_args.modelargs]
    args = model_args(vocab_size=parsed_args.vocab_size, context_length=parsed_args.context_length)
    device = parsed_args.device

    model = Llama3(
        model_args = args,
        device = device
    )

    print("Model information:")
    print("Device:", device)
    print("Model arguments Name:", parsed_args.modelargs)
    print("Model arguments:", args)
    print("Model size:", elab.get_parameter_size(model))
