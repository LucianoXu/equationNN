from model import *
from env import env, Scenario
import multiprocessing as mp

class Trace:
    def __init__(self, equation: env.Equation, steps: list[tuple[str, torch.Tensor, float]]):
        self.equation = equation # the final equation
        self.steps = steps

    def __str__(self):
        return str(self.equation) + " : " + str(self.steps)


class SolveEnv:
    '''
    The environment for the equation solving task.

    The reward is defined as the negative of the length of proof.
    '''
    def __init__(self, scenario: Scenario, problem : env.Equation):
        self.problem = problem
        self.scenario = scenario

    @property
    def state(self) -> str:
        return str(self.problem) + " : "
    
    def step(self, action: str, state_len_limit: int) -> float:
        '''
        Notice that the state length limit include the '<SOS>' token and the ':' token.
        '''

        temp_eq = env.Equation(self.problem)

        res = self.scenario.kernel.action_by_code(temp_eq, action)

        if res != env.ACT_RESULT.SUCCESS:
            return -1.

        # check whether the state length limit is reached
        state_len = len(self.scenario.tokenizer.encode(str(temp_eq))) + 2
        if state_len > state_len_limit:
            return -1.
        
        self.problem = temp_eq

        return -1.


def solve_group(model, scenario, problems: list[str], step_limit: int, state_len_limit: int, context_length: int, T: float) -> list[Trace]:
    '''
    Solve a group of proof kernels for parallel training and evaluation.
    '''
    parsed_problems : list[env.Equation] = [env.parse_equation(problem) for problem in problems]    # type: ignore
    envs = [SolveEnv(scenario, problem) for problem in parsed_problems]
    env_idx_mapping = [i for i in range(len(envs))]
    batch_size = len(envs)

    # proceed the examples
    traces: list[list[tuple[str, torch.Tensor, float]]] = [[] for _ in range(batch_size)]

    progress_bar = tqdm(total=step_limit)


    # NOTICE: multiprocessing acceleration is possible here

    for _ in range(step_limit):

        # check the finished envs
        temp_envs : list[SolveEnv] = []
        for i in range(len(envs)):
            if envs[i].problem.lhs == envs[i].problem.rhs:
                env_idx_mapping = env_idx_mapping[:i] + env_idx_mapping[i+1:]
            else:
                temp_envs.append(envs[i])

        # if all the envs are finished, break
        if len(temp_envs) == 0:
            break

        envs = temp_envs

        finished_count = batch_size - len(envs)

        # update the description of the progress bar
        progress_bar.desc = f"Solving in Envs({finished_count}/{batch_size})"

        # generate the batched solution
        batch = [env.state for env in envs]
        actions, log_probs = batch_generation(model, scenario, batch, context_length, T)

        reward_results : list[float] = []
        for i in range(len(envs)):
            # the generation may be unfinished and the C++ parser will report an warning
            reward_results.append(envs[i].step(actions[i], state_len_limit))

        for i, reward in enumerate(reward_results):
            traces[env_idx_mapping[i]].append((actions[i], log_probs[i], reward))

        progress_bar.update(1)

    return [Trace(problem, trace) for problem, trace in zip(parsed_problems, traces)]



def gen_group(model, 
                      scenario: Scenario,
                      batch_size: int = 10, 
                      step_limit: int = 20, 
                      state_len_limit: int = 128,
                      context_length: int = 256, 
                      T: float = 1.0) -> list[Trace]:
    '''
    Generate a group of example traces.

    Args:
        model: the model to generate the rewriting traces.
        batch_size: the number of examples to generate.
        step_limit: the maximum number of steps for each example.

    Returns:
        A list of traces. Each trace is a list of tuples (action, log probability, reward).
        Notice that an action is a complete response from the agent, not a single token.
    '''

    # create initial equations and environments
    x = list(scenario.sig.variables)[0]
    envs = [SolveEnv(scenario, env.Equation(env.Term(x), env.Term(x))) for _ in range(batch_size)]

    # proceed the examples
    traces: list[list[tuple[str, torch.Tensor, float]]] = [[] for _ in range(batch_size)]

    progress_bar = tqdm(total=step_limit)


    # NOTICE: multiprocessing acceleration is possible here

    for _ in range(step_limit):
        # update the description of the progress bar
        progress_bar.desc = f"Generating in Envs ({batch_size})"

        # generate the batch
        batch = [env.state for env in envs]
        actions, log_probs = batch_generation(model, scenario, batch, context_length, T)

        reward_results : list[float] = []
        for i in range(len(envs)):
            # the generation may be unfinished and the C++ parser will report an warning
            reward_results.append(envs[i].step(actions[i], state_len_limit))

        for i, reward in enumerate(reward_results):
            traces[i].append((actions[i], log_probs[i], reward))

        progress_bar.update(1)

    return [Trace(env.problem, trace) for env, trace in zip(envs, traces)]


import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.adamw import AdamW
from tqdm import tqdm
from typing import Optional
from elab import ELab, set_adamw_params, get_grad_norm

from model import ModelArgs, Llama3, SmallArgs

def rl_train(
        gen_model: Llama3,
        model: Llama3,
        scenario: Scenario,
        state_len_limit: int,
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
        accumulation_step: int = 10,
        save_interval: Optional[int] = 10,
        logging: bool = True,

        # reinforcement learning settings
        rl_gen_step_limit: int = 20,
        rl_sol_step_limit: int = 20,
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
                for _ in range(accumulation_step):
                    # STEP 1: sample the traces
                    # generate the problems
                    with torch.no_grad():
                        gen_traces = gen_group(gen_model, scenario, batch_size, rl_gen_step_limit, state_len_limit, context_length, rl_temperature)
                    
                    equations = [str(trace.equation) for trace in gen_traces]

                    sol_traces = solve_group(model, scenario, equations, rl_sol_step_limit, state_len_limit, context_length, rl_temperature)

                    ###########################

                    # STEP 2: calculate the pseudo loss
                    # calculate baseline (average total reward)
                    batch_reward = 0.
                    batch_SA_pair_count = 0
                    for trace in sol_traces:
                        for i in range(len(trace.steps)):
                            batch_reward += trace.steps[i][2]
                        batch_SA_pair_count += len(trace.steps)
                    avg_trace_reward = batch_reward / batch_size

                    # add to total
                    avg_reward += avg_trace_reward / accumulation_step
                    total_SA_pair_count += batch_SA_pair_count

                    J = torch.tensor(0.0, device=device)

                    for trace in sol_traces:
                        steps = trace.steps
                        for i in range(len(steps)):
                            _, log_prob, reward_to_go = steps[i]

                            # calculate the reward to go
                            for j in range(i+1, len(steps)):
                                _, _, r = steps[j]
                                reward_to_go += r

                            J -= log_prob * (reward_to_go - avg_trace_reward)

                    total_pseudo_loss += J.item()

                    # STEP 3: Backward pass and optimization
                    if any(len(trace.steps) > 0 for trace in sol_traces):
                        J.backward()        # Backward pass
                    else:
                        print("No valid trace generated.")

                if total_SA_pair_count == 0:
                    print("No valid trace generated.")
                    continue

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

                # output
                print(f"{ckpt_folder}\tStep {t}\tPseudo Loss: {avg_pseudo_loss:.3f}\tAvg Reward: {avg_reward:.3f}\tRaw Grad Norm: {raw_grad_norm:.3f}")

                # test generation result

                # Logging
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

    alg_code = '''
    [function]
    * : 2

    [variable]
    x y z u v w

    [axiom]
    (AX1) *(x y) = *(*(y y) x)
    '''

    scenario = Scenario(alg_code)

    args = SmallArgs(vocab_size=scenario.tokenizer.get_vocab_size(), context_length=256)
    device = 'cuda'

    rl_train(
        Llama3(
            model_args = args,
            device=device
        ),
        Llama3(
            model_args = args,
            device=device
        ),
        scenario = scenario,
        state_len_limit = 160,
        context_length = args.context_length,

        ckpt_folder = "./ckpt/Magma",
        input_version_name = 'none',

        lr = 2e-5,
        weight_decay=0.01,
        betas=(0.9, 0.99),
        grad_norm_clip=1.0,

        num_steps = 400,
        batch_size = 192,
        accumulation_step = 1,
        rl_gen_step_limit=6,
        rl_sol_step_limit=30,
        rl_temperature=0.6,

        save_interval=10000
    )