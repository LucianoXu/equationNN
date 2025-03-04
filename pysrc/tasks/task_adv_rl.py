
import argparse
from ..rl import *

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("adv_rl", help="Execute advanced RL training.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("--ckpt_folder", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("--init", action="store_true", help="Initialize training models.")
    parser.add_argument("--mode", type=str, default='sol', choices=['sol', 'gen'], help="Learning mode (sol/gen).")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # read the algorithm description
    with open(parsed_args.alg_desc) as f:
        alg_code = f.read()

    scenario = Scenario(alg_code)


    scenario = Scenario(alg_code)

    args = MediumArgs(vocab_size=scenario.tokenizer.get_vocab_size(), context_length=150)
    device = parsed_args.device
    
    gen_model = Llama3(
        model_args = args,
        device=device
    )
    sol_model = Llama3(
        model_args = args,
        device=device
    )

    if parsed_args.init:
        # clean the folder ckpt_folder
        import shutil, os
        shutil.rmtree(parsed_args.ckpt_folder, ignore_errors=True)
        os.makedirs(parsed_args.ckpt_folder)
        # initialize the models
        init_sol_gen_models(gen_model, sol_model, parsed_args.ckpt_folder, device)

    adv_rl_train(
        gen_model = gen_model,
        sol_model = sol_model,
        scenario = scenario,
        state_len_limit = 100,
        context_length = args.context_length,

        ckpt_folder = parsed_args.ckpt_folder,
        input_version_name = 'none' if parsed_args.init else 'latest',

        lr = 2e-6,
        weight_decay=0.01,
        betas=(0.9, 0.99),
        grad_norm_clip=1.0,

        learn_model = parsed_args.mode,
        num_steps = 64,
        batch_size = 6,
        accumulation_step = 16,
        rl_gen_step_limit=5,
        rl_sol_step_limit=15,
        rl_temperature=0.6,

        save_interval=10000
    )
