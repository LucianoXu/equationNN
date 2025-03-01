import argparse
from rl import *

def exe_adv_rl(parsed_args: argparse.Namespace):

    alg_code = '''
    [function]
    * : 2

    [variable]
    x y z u v w

    [axiom]
    (AX1) *(x y) = *(*(y y) x)
    '''

    scenario = Scenario(alg_code)

    args = MediumArgs(vocab_size=scenario.tokenizer.get_vocab_size(), context_length=150)
    device = parsed_args.device

    ckpt_folder = "./ckpt/Magma"
    
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
        shutil.rmtree(ckpt_folder, ignore_errors=True)
        os.makedirs(ckpt_folder)
        # initialize the models
        init_sol_gen_models(gen_model, sol_model, ckpt_folder, device)

    adv_rl_train(
        gen_model = gen_model,
        sol_model = sol_model,
        scenario = scenario,
        state_len_limit = 100,
        context_length = args.context_length,

        ckpt_folder = ckpt_folder,
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

def exe_gen_rl_vampire(parsed_args: argparse.Namespace):

    # alg_code = '''
    # [function]
    # * : 2

    # [variable]
    # x y z u v w

    # [axiom]
    # (AX1) *(x y) = *(*(y y) x)
    # '''

    '''
    fof(ax0, axiom, ![X, Y] : m(X, Y) = m(Y, X)).
    fof(ax1, axiom, ![X, Y] : j(X, Y) = j(Y, X)).
    fof(ax2, axiom, ![X, Y, Z] : m(m(X, Y), Z) = m(X, m(Y, Z))).
    fof(ax3, axiom, ![X, Y, Z] : j(j(X, Y), Z) = j(X, j(Y, Z))).
    fof(ax4, axiom, ![X, Y] : m(X, j(X, Y)) = X).
    fof(ax5, axiom, ![X, Y] : j(X, m(X, Y)) = X).
    fof(ax6, axiom, ![X] : n(n(X)) = X).
    fof(ax7, axiom, ![X, Y] : n(m(X, Y)) = j(n(X), n(Y))).
    fof(ax8, axiom, ![X, Y] : n(j(X, Y)) = m(n(X), n(Y))).
    fof(ax9, axiom, ![X, Y] : j(X, Y) = j(m(j(X, Y), X), m(j(X, Y), n(X)))).
    '''

    alg_code = '''
    [function]
    & : 2
    | : 2
    ~ : 1

    [variable]
    x y z u v w

    [axiom]
    (AX1) &(x y) = &(y x)
    (AX2) |(x y) = |(y x)
    (AX3) &(x &(y z)) = &(&(x y) z)
    (AX4) |(x |(y z)) = |(|(x y) z)
    (AX5) &(x |(x y)) = x
    (AX6) |(x &(x y)) = x
    (AX7) ~(~(x)) = x
    (AX8) ~(&(x y)) = |(~(x) ~(y))
    (AX9) ~(|(x y)) = &(~(x) ~(y))
    (OML) |(x y) = |(&(|(x y) x) &(|(x y) ~(x)))
    '''

    scenario = Scenario(alg_code)

    args = MediumArgs(vocab_size=scenario.tokenizer.get_vocab_size(), context_length=150)
    device = parsed_args.device

    ckpt_folder = "./ckpt/OMLVampire"

    if parsed_args.init:
        # clean the folder ckpt_folder
        import shutil, os
        shutil.rmtree(ckpt_folder, ignore_errors=True)
        os.makedirs(ckpt_folder)
    
    gen_model = Llama3(
        model_args = args,
        device = device
    )

    gen_rl_train_by_vampire(
        gen_model = gen_model,
        scenario = scenario,
        state_len_limit = 100,
        context_length = args.context_length,

        ckpt_folder = ckpt_folder,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main entry for multiple functions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parser for adv_rl_train
    parser_adv_rl = subparsers.add_parser("adv_rl", help="Execute advanced RL training.")
    parser_adv_rl.add_argument("--init", action="store_true", help="Initialize training models.")
    parser_adv_rl.add_argument("--mode", type=str, default='sol', choices=['sol', 'gen'], help="Learning mode (sol/gen).")
    parser_adv_rl.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser_adv_rl.set_defaults(func=exe_adv_rl)

    # Parser for gen_rl_vampire
    parser_gen_rl_vampire = subparsers.add_parser("gen_rl_vampire", help="Execute RL training with Vampire.")
    parser_gen_rl_vampire.add_argument("--init", action="store_true", help="Initialize training models.")
    parser_gen_rl_vampire.add_argument("--vampire", type=str, default='vampire', help="Path to Vampire.")
    parser_gen_rl_vampire.add_argument("--timeout", type=float, default=5, help="Timeout for Vampire.")
    parser_gen_rl_vampire.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser_gen_rl_vampire.add_argument("--num_steps", type=int, default=64, help="Number of steps.")
    parser_gen_rl_vampire.add_argument("--batch_size", type=int, default=6, help="Batch size.")
    parser_gen_rl_vampire.add_argument("--acc_steps", type=int, default=16, help="Accumulation step.")
    parser_gen_rl_vampire.add_argument("--gen_step_limit", type=int, default=5, help="Generation step limit.")
    parser_gen_rl_vampire.set_defaults(func=exe_gen_rl_vampire)

    # Parse arguments
    args = parser.parse_args()
    
    # Call the function associated with the chosen command
    args.func(args)

    # python main.py gen_rl_vampire --init --num_steps=512 --batch_size=12 --acc_steps=8 --gen_step_limit=12
