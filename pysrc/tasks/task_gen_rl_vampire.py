
import argparse
from ..rl import *

def build_parser(subparsers: argparse._SubParsersAction):
    '''
    Example: 
    python main.py gen_rl_vampire --init --num_steps=512 --batch_size=12 --acc_steps=8 --gen_step_limit=12
    '''
    parser = subparsers.add_parser("gen_rl_vampire", help="Execute RL training with Vampire.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("--ckpt_folder", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("--init", action="store_true", help="Initialize training models.")
    parser.add_argument("--vampire", type=str, default='vampire', help="Path to Vampire.")
    parser.add_argument("--timeout", type=float, default=5, help="Timeout for Vampire.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of steps.")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size.")
    parser.add_argument("--acc_steps", type=int, default=16, help="Accumulation step.")
    parser.add_argument("--gen_step_limit", type=int, default=5, help="Generation step limit.")
    parser.set_defaults(func=task)


def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)

    args = MediumArgs(vocab_size=scenario.tokenizer.get_vocab_size(), context_length=150)
    device = parsed_args.device

    if parsed_args.init:
        # clean the folder ckpt_folder
        import shutil, os
        shutil.rmtree(parsed_args.ckpt_folder, ignore_errors=True)
        os.makedirs(parsed_args.ckpt_folder)
    
    gen_model = Llama3(
        model_args = args,
        device = device
    )

    gen_rl_train_by_vampire(
        gen_model = gen_model,
        scenario = scenario,
        state_len_limit = 100,
        context_length = args.context_length,

        ckpt_folder = parsed_args.ckpt_folder,
        input_version_name = 'none' if parsed_args.init else 'latest',

        lr = 1e-7,
        weight_decay=0.01,
        betas=(0.9, 0.99),
        grad_norm_clip=1.0,

        num_steps = parsed_args.num_steps,
        batch_size = parsed_args.batch_size,
        accumulation_step = parsed_args.acc_steps,
        rl_gen_step_limit = parsed_args.gen_step_limit,
        rl_temperature=0.6,
        vampire=parsed_args.vampire,
        timeout=parsed_args.timeout,

        save_interval=10000
    )

