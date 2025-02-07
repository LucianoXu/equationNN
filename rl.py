# rl training for higher interestingness

from pyualg import Term
from model import *
from scenario import RULE_NAMES_INV, parser, signature, forbidden_heads
import multiprocessing as mp

from evaluation import _test_intere

import json

device = json.load(open('config.json'))['backend']

class GenEnv:
    error_reward = -0.05
    def __init__(self, problem : Term, timeout = 10.):
        self.problem = problem
        
        # note that the reward is the difference between the interestingness of the current equation and the last equation. starting from 0.
        self.last_interestingness = 0.

        self.timeout = timeout

    @property
    def state(self) -> str:
        return self.problem.sig_str(signature) + " : "

    def step(self, action: str) -> float:

        # parse the action
        encoding = tok_encode(action)

        # try to apply the command. any exception will be caught and return
        try:
            # find the rule in RULE_NAME_INV
            if id2token[encoding[0]] not in RULE_NAMES_INV:
                return self.error_reward
            
            rule = RULE_NAMES_INV[id2token[encoding[0]]]

            # find the first token2id['{'] element in encoding
            subst_start_id = encoding.index(token2id['{'])

            
            pos = tuple(int(id2token[id]) for id in encoding[1:subst_start_id])

            subst = parser.parse_subst(tok_decode(encoding[subst_start_id:]))

            res = self.problem.apply_at(rule, signature, pos, subst, forbidden_heads)

        except Exception as e:
            # print(f"An error of type {type(e)} occurred: {e}")
            return self.error_reward

        # if the command is not applicable, return
        if res is None:
            return self.error_reward
        
        # if the command is applicable, conduct the command
        self.problem = res[0]

        # test the interestingness of the current equation
        interestingness = _test_intere((self.problem, self.timeout))

        # calculate the reward
        reward = interestingness - self.last_interestingness
        self.last_interestingness = interestingness
        return reward
    

def process_env(i, shared_envs, action):
    '''
    helping function for parallel processing
    '''
    env = shared_envs[i]
    reward = env.step(action)
    shared_envs[i] = env
    return i, reward

def gen_example_group(model, batch_size: int = 10, step_limit: int = 20, context_length: int = 256, T: float = 1.0) -> list[list[tuple[str, torch.Tensor, float]]]:
    '''
    Generate a group of example traces.

    Args:
        batch_size: the number of examples to generate.
        step_limit: the maximum number of steps for each example.

    Returns:
        A list of traces. Each trace is a list of tuples (action, log probability, reward).
        Notice that an action is a complete response from the agent, not a single token.
    '''
    
    envs = [GenEnv(Term("=", (Term("x"), Term("x")))) for _ in range(batch_size)]

    # proceed the examples
    traces: list[list[tuple[str, torch.Tensor, float]]] = [[] for _ in range(batch_size)]

    remaining_number = len(envs)
    progress_bar = tqdm(total=step_limit)

    with mp.Manager() as manager:
        shared_envs = manager.list(envs)

        with mp.Pool(mp.cpu_count()) as pool:
            for _ in range(step_limit):
                # update the description of the progress bar
                progress_bar.desc = f"Generating in Envs({remaining_number}/{batch_size})"

                # generate the batch
                batch = [env.state for env in shared_envs]
                actions, log_probs = batch_generation(model, batch, context_length, T)

                remaining_number = 0

                # multiprocess version
                reward_results = pool.starmap(process_env, [(i, shared_envs, actions[i]) for i in range(len(shared_envs))])

                # single thread version
                # reward_results = []
                # for i in range(len(envs)):
                #     reward_results.append((i, envs[i].step(actions[i])))

                for i, reward in reward_results:
                    traces[i].append((actions[i], log_probs[i], reward))
                    remaining_number += 1

                progress_bar.update(1)

    return traces

    
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.adamw import AdamW
from tqdm import tqdm
from typing import Optional
from elab import ELab, set_adamw_params, get_grad_norm

from model import ModelArgs, Llama3, token2id, SmallArgs

def rl_train(
        model: Llama3,
        context_length: int,

        ckpt_folder: str,
        input_version_name: str,
        
        
        # optimizer
        lr: float,
        weight_decay: float, 
        betas: tuple[float, float], 
        eps = 1e-8,
        grad_norm_clip: Optional[float] = None,

        # training settings
        num_steps: int = 200, 
        batch_size: int = 10, 
        accumulaton_step: int = 10,
        save_interval: Optional[int] = 10,
        logging: bool = True,

        # reinforcement learning settings
        rl_step_limit: int = 20,
        rl_temperature: float = 0.6,):


    # get device
    device = next(model.parameters()).device

    # Set up the optimizer
    optimizer = AdamW(
        model.parameters(),
        lr = lr, betas = betas, weight_decay=weight_decay,
        eps = eps
    )

    # create/load the checkpoint
    # here t represents the next step number to be executed
    lab = ELab(
        ckpt_folder, 
        version_name=input_version_name,
        model = model,
        optimizer = optimizer,
        default_states={
            't': 1,
        }
    )

    set_adamw_params(optimizer, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

    model.train()
    optimizer.zero_grad()

    t: int = lab.states['t']

    # tensorboard logger
    writer = SummaryWriter(lab.folder_path)

    try:
        step = 0
        while step < num_steps:

            print(f"Step {step + 1}/{num_steps}")


            # note that reward is calculated for each trace
            avg_reward = 0.
            total_pseudo_loss = 0.
            total_SA_pair_count = 0
            torch.cuda.empty_cache()

            try:
                for _ in range(accumulaton_step):
                    # STEP 1: sample the traces
                    # generate the example traces
                    traces = gen_example_group(model, batch_size, rl_step_limit, context_length, rl_temperature)
                    ###########################

                    # STEP 2: calculate the pseudo loss
                    # calculate baseline (average total reward)
                    batch_reward = 0.
                    batch_SA_pair_count = 0
                    for trace in traces:
                        for i in range(len(trace)):
                            batch_reward += trace[i][2]
                        batch_SA_pair_count += len(trace)
                    avg_trace_reward = batch_reward / batch_size

                    # add to total
                    avg_reward += avg_trace_reward / accumulaton_step
                    total_SA_pair_count += batch_SA_pair_count

                    J = torch.tensor(0.0, device=device)

                    for trace in traces:
                        for i in range(len(trace)):
                            _, log_prob, reward_to_go = trace[i]

                            # calculate the reward to go
                            for j in range(i+1, len(trace)):
                                _, _, r = trace[j]
                                reward_to_go += r

                            J -= log_prob * (reward_to_go - avg_trace_reward)

                    total_pseudo_loss += J.item()

                    # STEP 3: Backward pass and optimization
                    J.backward()        # Backward pass

                # for normalization reasons, the pseudo loss is calculated for each state-action pair
                avg_pseudo_loss = total_pseudo_loss / total_SA_pair_count

                # adjust the gradient by total SA pair count
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= total_SA_pair_count

                raw_grad_norm = get_grad_norm(model)

                if grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)


                optimizer.step()       # Update weights
                optimizer.zero_grad()

                # Logging
                print(f"{ckpt_folder}\tStep {t}\tPseudo Loss: {avg_pseudo_loss:.3f}\tAvg Reward: {avg_reward:.3f}\tRaw Grad Norm: {raw_grad_norm:.3f}")
                if logging:
                    writer.add_scalar("pseudo_loss", avg_pseudo_loss, t)
                    writer.add_scalar("avg reward", avg_reward, t)
                    writer.add_scalar("raw grad norm", raw_grad_norm, t)

                t += 1

                if save_interval is not None and t % save_interval == 0:
                    lab.states['t'] = t
                    lab.save(f"RL-{t}")

                step += 1
            
            except torch.OutOfMemoryError:
                print("!!! Out of memory error. Skipping this batch !!! ")
                optimizer.zero_grad()


    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"An error of type {type(e)} occurred: {e}")

    finally:
        if save_interval is not None:
            lab.states['t'] = t
            lab.save(f"RL-{t}")

    print("Training completed and model saved.")

if __name__ == '__main__':
    args = SmallArgs()
    rl_train(
        Llama3(
            model_args = args,
            device=device
        ),
        context_length = args.context_length,

        ckpt_folder = "./ckpt/OMLgen",
        input_version_name = '791',

        lr = 2e-5,
        weight_decay=0.01,
        betas=(0.9, 0.99),
        grad_norm_clip=1.0,

        num_steps = 400,
        batch_size = 6,
        accumulaton_step = 15,
        rl_step_limit=10,
        rl_temperature=0.6,

        save_interval=10000
    )