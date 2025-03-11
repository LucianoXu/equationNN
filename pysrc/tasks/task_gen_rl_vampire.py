
import argparse
from ..rl import *

def build_parser(subparsers: argparse._SubParsersAction):
    '''
    Example: 
    python main.py gen_rl_vampire --init --num_steps=512 --batch_size=12 --acc_steps=8 --gen_step_limit=12
    '''
    parser = subparsers.add_parser("gen_rl_vampire", help="Execute RL training with Vampire.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("-i", "--init", action="store_true", help="Initialize training models.")
    parser.add_argument("-v", "--version", type=str, help="Version of the model to load. Cannot be used together with -i.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser.add_argument("--modelargs", type=str, default='medium', help="Model arguments.")
    parser.add_argument("--vampire", type=str, default='vampire', help="Path to Vampire.")
    parser.add_argument("--timeout", type=float, default=5, help="Timeout for Vampire.")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size.")
    parser.add_argument("--acc_steps", type=int, default=16, help="Accumulation step.")
    parser.add_argument("--gen_step_limit", type=int, default=5, help="Generation step limit.")
    parser.add_argument("-t", "--temperature", type=float, default=0.6, help="Temperature for model in RL.")
    parser.add_argument("--save_interval", type=int, default=10000, help="Save interval.")
    parser.set_defaults(func=task)


def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    model_args = modelargs_dict[parsed_args.modelargs]
    args = model_args(vocab_size=scenario.tokenizer.get_vocab_size(), context_length=150)
    device = parsed_args.device

    if parsed_args.init:
        # clean the folder ckpt
        import shutil, os
        shutil.rmtree(parsed_args.ckpt, ignore_errors=True)
        os.makedirs(parsed_args.ckpt)
    
    gen_model = Llama3(
        model_args = args,
        device = device
    )

    if parsed_args.version:
        version = parsed_args.version
    else:
        version = 'none' if parsed_args.init else 'latest'


    gen_rl_train_by_vampire(
        gen_model = gen_model,
        scenario = scenario,
        state_len_limit = 100,
        context_length = args.context_length,

        ckpt_folder = parsed_args.ckpt,
        input_version_name = version,

        lr = parsed_args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.99),
        grad_norm_clip=1.0,

        num_steps = parsed_args.num_steps,
        batch_size = parsed_args.batch_size,
        accumulation_step = parsed_args.acc_steps,
        rl_gen_step_limit = parsed_args.gen_step_limit,
        rl_temperature=parsed_args.temperature,
        vampire=parsed_args.vampire,
        timeout=parsed_args.timeout,

        save_interval=parsed_args.save_interval
    )

