
import argparse
from ..rl import *

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("adv_rl", help="Execute advanced RL training.")
    parser.add_argument("alg_desc", type=str, help="Path to the algorithm description.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("-i", "--init", action="store_true", help="Initialize training models.")

    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser.add_argument("--modelargs", type=str, default='medium', help="Model arguments.")
    parser.add_argument("--starting_mode", type=str, default='sol', choices=['sol', 'gen'], help="Starting mode (sol/gen).")
    parser.add_argument("--num_steps_per_turn", type=int, default=64, help="Number of steps.")
    parser.add_argument("--num_turns", type=int, default=20, help="Number of turns.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size.")
    parser.add_argument("--acc_steps", type=int, default=16, help="Accumulation step.")
    parser.add_argument("--gen_step_limit", type=int, default=5, help="Generation step limit.")
    parser.add_argument("--sol_step_limit", type=int, default=15, help="Solution step limit.")
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
        device=device
    )
    sol_model = Llama3(
        model_args = args,
        device=device
    )

    # initialize the models
    init_sol_gen_models(gen_model, sol_model, parsed_args.ckpt, device)


    adv_rl_train_in_turns(
        gen_model = gen_model,
        sol_model = sol_model,
        scenario = scenario,
        state_len_limit = 100,
        context_length = args.context_length,

        ckpt_folder = parsed_args.ckpt,

        lr = parsed_args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.99),
        grad_norm_clip=1.0,

        starting_model = parsed_args.starting_mode,
        num_steps_per_turn = parsed_args.num_steps_per_turn,
        num_turns = parsed_args.num_turns,
        batch_size = parsed_args.batch_size,
        accumulation_step = parsed_args.acc_steps,
        rl_gen_step_limit = parsed_args.gen_step_limit,
        rl_sol_step_limit = parsed_args.sol_step_limit,
        rl_temperature = parsed_args.temperature,

        save_interval = parsed_args.save_interval
    )
