from .model import *
from .env import env, Scenario
from .evaluation import test_intere_mp
from . import syntax_fuzzer

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.adamw import AdamW
from tqdm import tqdm
from typing import Literal, Optional
from elab import ELab, set_adamw_params, get_grad_norm


class Trace:
    def __init__(self, init_eq: env.Equation, steps: list[tuple[str, torch.Tensor, float]], final_eq: env.Equation):
        self.init_eq = init_eq
        self.final_eq = final_eq
        self.steps = steps

    def __str__(self):
        res = "Initial equation: " + str(self.init_eq) + "\n" 
        res += "\n".join([str(step) for step in self.steps]) + "\n"
        res += "Final equation: " + str(self.final_eq) + "\n"
        return res

def interestingness(equation: env.Equation, proof_step: int) -> float:
    '''
    Calculate the interestingness of the equation.

    (The interestingness function is for adversarial reinforcement learning training.)
    '''
    size = equation.lhs.get_term_size() + equation.rhs.get_term_size() + 1
    return 20 * proof_step / size

class GenEnv:
    '''
    The environment for the equation generation task.

    The reward is defined as follows:
    - A reward of the final interestingness is given in the end (in the rl training, instead of step function here)
    - If any step tries to exceed the state length limit, a penalty of -1 is given.
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

        return 0.


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
    
    @property
    def stopped(self) -> bool:
        return self.problem.lhs == self.problem.rhs
    
    def step(self, action: str, state_len_limit: int) -> float:
        '''
        Notice that the state length limit include the '<SOS>' token and the ':' token.
        '''

        temp_eq = env.Equation(self.problem)

        res = self.scenario.kernel.action_by_code(temp_eq, action)

        if res != env.ACT_RESULT.SUCCESS:
            return 0.

        # check whether the state length limit is reached （ +2 because of the <SOS> and <EOS> token)
        state_len = len(self.scenario.tokenizer.encode(str(temp_eq))) + 2
        if state_len > state_len_limit:
            return 0.
        
        self.problem = temp_eq

        # check whether the problem is solved
        if self.stopped:
            return 1.

        return 0.


def solve_group(model, scenario, problems: list[str], step_limit: int, state_len_limit: int, context_length: int, T: float) -> list[Trace]:
    '''
    Solve a group of proof kernels for parallel training and evaluation.
    '''
    parsed_problems : list[env.Equation] = [env.parse_equation(problem) for problem in problems]    # type: ignore
    envs = [SolveEnv(scenario, problem) for problem in parsed_problems]
    # the mapping from the index in remained envs to the index in the original envs
    env_idx_mapping = [i for i in range(len(envs))]
    batch_size = len(envs)

    # proceed the examples
    traces: list[list[tuple[str, torch.Tensor, float]]] = [[] for _ in range(batch_size)]

    progress_bar = tqdm(total=step_limit)

    remaining_envs = envs

    # NOTICE: multiprocessing acceleration is possible here

    for _ in range(step_limit):

        # check the finished envs
        temp_envs : list[SolveEnv] = []
        temp_env_idx_mapping : list[int] = []
        for i in range(len(remaining_envs)):
            # if the problem is solved
            if not remaining_envs[i].stopped:
                temp_env_idx_mapping.append(env_idx_mapping[i])
                temp_envs.append(remaining_envs[i])

        # if all the envs are finished, break
        if len(temp_envs) == 0:
            break

        # the unfinished ones
        remaining_envs = temp_envs
        env_idx_mapping = temp_env_idx_mapping

        finished_count = batch_size - len(remaining_envs)

        # update the description of the progress bar
        progress_bar.desc = f"Solving in Envs({finished_count}/{batch_size})"

        # generate the batched solution
        batch = [env.state for env in remaining_envs]
        actions, log_probs = batch_generation(model, scenario, batch, False, context_length, T)

        reward_results : list[float] = []
        for i in range(len(remaining_envs)):
            # the generation may be unfinished and the C++ parser will report an warning. This is not a problem.
            reward_results.append(remaining_envs[i].step(actions[i], state_len_limit))

        for i, reward in enumerate(reward_results):
            traces[env_idx_mapping[i]].append((actions[i], log_probs[i], reward))

        progress_bar.update(1)

    return [Trace(parsed_problems[i], traces[i], envs[i].problem) for i in range(batch_size)]



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
    init_eq = env.Equation(env.Term(x), env.Term(x))
    envs = [GenEnv(scenario, env.Equation(init_eq)) for _ in range(batch_size)]

    # proceed the examples
    traces: list[list[tuple[str, torch.Tensor, float]]] = [[] for _ in range(batch_size)]

    progress_bar = tqdm(total=step_limit)

    remaining_envs = envs

    # NOTICE: multiprocessing acceleration is possible here

    for _ in range(step_limit):
        # update the description of the progress bar
        progress_bar.desc = f"Generating in Envs ({batch_size})"

        # generate the batch
        batch = [env.state for env in remaining_envs]
        actions, log_probs = batch_generation(model, scenario, batch, True, context_length, T)

        reward_results : list[float] = []
        for i in range(len(remaining_envs)):
            # the generation may be unfinished and the C++ parser will report an warning. This is not a problem.
            print(actions[i])
            reward_results.append(remaining_envs[i].step(actions[i], state_len_limit))

        for i, reward in enumerate(reward_results):
            traces[i].append((actions[i], log_probs[i], reward))

        progress_bar.update(1)

    return [Trace(init_eq, trace, env.problem) for env, trace in zip(envs, traces)]


def construct_pseudo_loss(sol_traces: list[Trace], device: str|torch.device) -> tuple[float, torch.Tensor]:
    '''
    Construct and return the reward and pseudo loss for this batch.
    '''

    # calculate the baseline
    batch_reward = 0.
    for trace in sol_traces:
        for i in range(len(trace.steps)):
            batch_reward += trace.steps[i][2]

    avg_trace_reward = batch_reward / len(sol_traces) # average reward per trace

    J = torch.tensor(0.0, device=device)

    for trace in sol_traces:
        steps = trace.steps
        for i in range(len(steps)):
            _, log_prob, reward_to_go = steps[i]

            # calculate the reward to go
            for j in range(i+1, len(steps)):
                _, _, r = steps[j]
                reward_to_go += r

            J += log_prob * (reward_to_go - avg_trace_reward)

    return avg_trace_reward, J / len(sol_traces)

def init_sol_gen_models(        
        gen_model: Llama3,
        sol_model: Llama3,
        ckpt_folder: str,
        device : str):
    '''
    Initialize the models and optimizers for the reinforcement learning training.
    '''
            
    gen_optim = AdamW(gen_model.parameters())
    ELab(
        ckpt_folder + "/gen",
        version_name="none",
        data = {
            'model': gen_model,
            'optim' : gen_optim,
            't' : 1
        },
        device = device
    ).save("init")

    sol_optim = AdamW(gen_model.parameters())
    ELab(
        ckpt_folder + "/sol",
        version_name="none",
        data = {
            'model': sol_model,
            'optim' : sol_optim,
            't' : 1
        },
        device = device
    ).save("init")

def adv_rl_train(
        gen_model: Llama3,
        sol_model: Llama3,
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
        learn_model : Literal['gen', 'sol'] = 'sol',
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
    device = next(sol_model.parameters()).device

    # Set up the optimizer (according to the learning model)
    optimizer = AdamW(
        sol_model.parameters() if learn_model == 'sol' else gen_model.parameters(),
        lr = lr, betas = betas, weight_decay=weight_decay,
        eps = eps
    )

    # create/load the checkpoint
    # here t represents the next step number to be executed
    if learn_model == 'sol':
        sol_lab = ELab(
            ckpt_folder + "/sol", 
            version_name=input_version_name,
            data = {
                'model': sol_model,
                'optim' : optimizer,
                't' : 1
            },
            device = device
        )
        gen_lab = ELab(
            ckpt_folder + "/gen",
            version_name=input_version_name,
            data = {
                'model': gen_model,
                't' : 1
            },
            device = device
        )
        sol_model.train()
        gen_model.eval()
        t: int = max(sol_lab.data['t'], gen_lab.data['t'])

    else:
        sol_lab = ELab(
            ckpt_folder + "/sol", 
            version_name=input_version_name,
            data = {
                'model': sol_model,
                't' : 1
            },
            device = device
        )
        gen_lab = ELab(
            ckpt_folder + "/gen",
            version_name=input_version_name,
            data = {
                'model': gen_model,
                'optim' : optimizer,
                't' : 1
            },
            device=device
        )
        gen_model.train()
        sol_model.eval()
        t: int = max(sol_lab.data['t'], gen_lab.data['t'])



    # the method to save the labs
    def save_labs():
        sol_lab.data['t'] = t
        gen_lab.data['t'] = t
        if learn_model == 'sol':
            sol_lab.save(f"RL-{t}")
        else:
            gen_lab.save(f"RL-{t}")            
    
    set_adamw_params(optimizer, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

    optimizer.zero_grad()

    # tensorboard logger
    writer = SummaryWriter(ckpt_folder)

    try:
        step = 0
        while step < num_steps:

            print(f"Step {step + 1}/{num_steps}")

            # note that reward is calculated for each trace
            sol_avg_reward = 0.
            sol_total_pseudo_loss = 0.

            gen_avg_reward = 0.
            gen_total_pseudo_loss = 0.

            torch.cuda.empty_cache()

            try:
                for _ in range(accumulation_step):
                    # STEP 1: sampling the episodes
                    # generate the problems
                    gen_traces = gen_group(gen_model, scenario, batch_size, rl_gen_step_limit, state_len_limit, context_length, rl_temperature)
                    
                    # solve the problems
                    equations = [str(trace.final_eq) for trace in gen_traces]
                    sol_traces = solve_group(sol_model, scenario, equations, rl_sol_step_limit, state_len_limit, context_length, rl_temperature)

                    ###########################
                    
                    # STEP 2: Calculate the pseudo loss

                    # (GEN)
                    # add the final reward of interestingness according to the solution result
                    for i in range(len(gen_traces)):
                        original = gen_traces[i].steps[-1]
                        gen_traces[i].steps[-1] = (original[0], original[1], original[2] + interestingness(gen_traces[i].final_eq, len(sol_traces[i].steps)))

                    gen_reward, gen_J = construct_pseudo_loss(gen_traces, device)

                    gen_avg_reward += gen_reward / accumulation_step
                    gen_total_pseudo_loss += gen_J.item()

                    # (SOL)
                    sol_reward, sol_J = construct_pseudo_loss(sol_traces, device)

                    sol_avg_reward += sol_reward / accumulation_step
                    sol_total_pseudo_loss += sol_J.item()

                    ###########################

                    # STEP 3: Backward pass and optimization
                    if learn_model == 'sol':
                        if any(len(trace.steps) > 0 for trace in sol_traces):
                            (-sol_J).backward()        # Backward pass
                        else:
                            print("No valid sol trace generated.")
                    else:
                        if any(len(trace.steps) > 0 for trace in gen_traces):
                            (-gen_J).backward()
                        else:
                            print("No valid gen trace generated.")

                # average the pseudo loss
                sol_avg_pseudo_loss = sol_total_pseudo_loss / accumulation_step
                gen_avg_pseudo_loss = gen_total_pseudo_loss / accumulation_step

                # adjust the gradient by number of accumulation steps
                for param in (sol_model.parameters() if learn_model == 'sol' else gen_model.parameters()):
                    if param.grad is not None:
                        param.grad /= accumulation_step

                if learn_model == 'sol':
                    sol_raw_grad_norm = get_grad_norm(sol_model)
                else:
                    gen_raw_grad_norm = get_grad_norm(gen_model)

                if grad_norm_clip is not None:
                    if learn_model == 'sol':
                        torch.nn.utils.clip_grad_norm_(sol_model.parameters(), grad_norm_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), grad_norm_clip)


                optimizer.step()
                optimizer.zero_grad()

                # test generation result
                with torch.no_grad():
                    demo_traces = gen_group(gen_model, scenario, 10, rl_gen_step_limit, state_len_limit, context_length, rl_temperature)
                    demo_eqs = [str(trace.final_eq) for trace in demo_traces]
                    demo_sols = solve_group(sol_model, scenario, demo_eqs, rl_sol_step_limit, state_len_limit, context_length, rl_temperature)
                    demo_record = ""
                    for i in range(len(demo_traces)):
                        demo_record += f"{demo_eqs[i]}\nSOL-LENGTH {len(demo_sols[i].steps)}, INTERESTINGNESS {interestingness(demo_traces[i].final_eq, len(demo_sols[i].steps))}\n"

                # output
                print(f"{ckpt_folder}\t({learn_model}) Step {t}")
                print(f"Gen Pseudo Loss: {gen_avg_pseudo_loss:.3f}\tGen Avg Reward: {gen_avg_reward:.3f}")
                print(f"Sol Pseudo Loss: {sol_avg_pseudo_loss:.3f}\tSol Avg Reward: {sol_avg_reward:.3f}")
                if (learn_model == 'sol'):
                    print(f"Raw Grad Norm: {sol_raw_grad_norm:.3f}")
                else:
                    print(f"Raw Grad Norm: {gen_raw_grad_norm:.3f}")
                print("Demo:")
                print(demo_record)


                # Logging
                if logging:
                    writer.add_scalar("sol pseudo loss", sol_avg_pseudo_loss, t)
                    writer.add_scalar("sol avg reward", sol_avg_reward, t)
                    writer.add_scalar("gen pseudo loss", gen_avg_pseudo_loss, t)
                    writer.add_scalar("gen avg reward", gen_avg_reward, t)
                    if learn_model == 'sol':
                        writer.add_scalar("sol raw grad norm", sol_raw_grad_norm, t)
                    else:
                        writer.add_scalar("gen raw grad norm", gen_raw_grad_norm, t)
                    writer.add_text("demo", demo_record, t)

                t += 1

                if save_interval is not None and t % save_interval == 0:
                    save_labs()

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
            save_labs()

    print("Training completed and model saved.")



def sol_rl_train_by_fuzzer(
        sol_model: Llama3,
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
        fuzzer_step_limit: int = 10,
        rl_sol_step_limit: int = 20,
        rl_temperature: float = 0.6):


    # get device
    device = next(sol_model.parameters()).device

    # Set up the optimizer (according to the learning model)
    optimizer = AdamW(
        sol_model.parameters(),
        lr = lr, betas = betas, weight_decay=weight_decay,
        eps = eps
    )

    gen_lab = ELab(
        ckpt_folder,
        version_name=input_version_name,
        data = {
            'model': sol_model,
            'optim' : optimizer,
            't' : 1
        },
        device=device
    )

    sol_model.train()
    t: int = gen_lab.data['t']

    # the method to save the labs
    def save_labs():
        gen_lab.data['t'] = t
        gen_lab.save(f"RL-{t}")            
    
    set_adamw_params(optimizer, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

    optimizer.zero_grad()

    # tensorboard logger
    writer = SummaryWriter(ckpt_folder)

    try:
        step = 0
        while step < num_steps:

            print(f"Step {step + 1}/{num_steps}")

            # note that reward is calculated for each trace
            sol_avg_reward = 0.
            sol_total_pseudo_loss = 0.

            torch.cuda.empty_cache()

            try:
                for _ in range(accumulation_step):
                    # STEP 1: sampling the episodes
                    # generate the problems
                    traces = syntax_fuzzer.gen_examples(scenario, batch_size, fuzzer_step_limit)
                    equations = [str(trace.final_eq) for trace in traces]
                    
                    # solve the problems
                    sol_traces = solve_group(sol_model, scenario, equations, rl_sol_step_limit, state_len_limit, context_length, rl_temperature)

                    ###########################
                    
                    # STEP 2: Calculate the pseudo loss
                    sol_reward, sol_J = construct_pseudo_loss(sol_traces, device)

                    sol_avg_reward += sol_reward / accumulation_step
                    sol_total_pseudo_loss += sol_J.item()

                    ###########################

                    # STEP 3: Backward pass and optimization
                    if any(len(trace.steps) > 0 for trace in sol_traces):
                        (-sol_J).backward()
                    else:
                        print("No valid gen trace generated.")

                # average the pseudo loss
                sol_avg_pseudo_loss = sol_total_pseudo_loss / accumulation_step

                # adjust the gradient by number of accumulation steps
                for param in (sol_model.parameters()):
                    if param.grad is not None:
                        param.grad /= accumulation_step

                sol_raw_grad_norm = get_grad_norm(sol_model)

                if grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(sol_model.parameters(), grad_norm_clip)


                optimizer.step()
                optimizer.zero_grad()

                # test generation result
                with torch.no_grad():
                    demo_traces = syntax_fuzzer.gen_examples(scenario, 10, fuzzer_step_limit)
                    demo_eqs = [str(trace.final_eq) for trace in demo_traces]

                    demo_sols = solve_group(sol_model, scenario, demo_eqs, rl_sol_step_limit, state_len_limit, context_length, rl_temperature)

                    demo_record = ""
                    for trace in demo_sols:
                        demo_record += str(trace) + "\n\n"

                # output
                print(f"{ckpt_folder}\t Step {t}")
                print(f"Sol Pseudo Loss: {sol_avg_pseudo_loss:.3f}\tSol Avg Reward: {sol_avg_reward:.3f}")
                print(f"Raw Grad Norm: {sol_raw_grad_norm:.3f}")
                print("Demo:")
                print(demo_record)


                # Logging
                if logging:
                    writer.add_scalar("sol pseudo loss", sol_avg_pseudo_loss, t)
                    writer.add_scalar("sol avg reward", sol_avg_reward, t)
                    writer.add_scalar("sol raw grad norm", sol_raw_grad_norm, t)
                    writer.add_text("demo", demo_record, t)

                t += 1

                if save_interval is not None and t % save_interval == 0:
                    save_labs()

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
            save_labs()

    print("Training completed and model saved.")



def gen_rl_train_by_vampire(
        gen_model: Llama3,
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
        rl_temperature: float = 0.6,
        vampire: str = "vampire",
        timeout: float = 5,):


    # get device
    device = next(gen_model.parameters()).device

    # Set up the optimizer (according to the learning model)
    optimizer = AdamW(
        gen_model.parameters(),
        lr = lr, betas = betas, weight_decay=weight_decay,
        eps = eps
    )

    gen_lab = ELab(
        ckpt_folder,
        version_name=input_version_name,
        data = {
            'model': gen_model,
            'optim' : optimizer,
            't' : 1
        },
        device=device
    )

    gen_model.train()
    t: int = gen_lab.data['t']

    # the method to save the labs
    def save_labs():
        gen_lab.data['t'] = t
        gen_lab.save(f"RL-{t}")            
    
    set_adamw_params(optimizer, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

    optimizer.zero_grad()

    # tensorboard logger
    writer = SummaryWriter(ckpt_folder)

    try:
        step = 0
        while step < num_steps:

            print(f"Step {step + 1}/{num_steps}")

            # note that reward is calculated for each trace
            gen_avg_reward = 0.
            gen_total_pseudo_loss = 0.

            torch.cuda.empty_cache()

            try:
                for _ in range(accumulation_step):
                    # STEP 1: sampling the episodes
                    # generate the problems
                    gen_traces = gen_group(gen_model, scenario, batch_size, rl_gen_step_limit, state_len_limit, context_length, rl_temperature)
                    
                    # solve the problems
                    equations = [trace.final_eq for trace in gen_traces]
                    intere_vals = test_intere_mp(vampire, scenario, equations, timeout)

                    ###########################
                    
                    # STEP 2: Calculate the pseudo loss

                    # (GEN)
                    # add the final reward of interestingness according to the solution result
                    for i in range(len(gen_traces)):
                        original = gen_traces[i].steps[-1]
                        gen_traces[i].steps[-1] = (original[0], original[1], original[2] + intere_vals[i])

                    gen_reward, gen_J = construct_pseudo_loss(gen_traces, device)

                    gen_avg_reward += gen_reward / accumulation_step
                    gen_total_pseudo_loss += gen_J.item()

                    ###########################

                    # STEP 3: Backward pass and optimization
                    if any(len(trace.steps) > 0 for trace in gen_traces):
                        (-gen_J).backward()
                    else:
                        print("No valid gen trace generated.")

                # average the pseudo loss
                gen_avg_pseudo_loss = gen_total_pseudo_loss / accumulation_step

                # adjust the gradient by number of accumulation steps
                for param in (gen_model.parameters()):
                    if param.grad is not None:
                        param.grad /= accumulation_step

                gen_raw_grad_norm = get_grad_norm(gen_model)

                if grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(gen_model.parameters(), grad_norm_clip)


                optimizer.step()
                optimizer.zero_grad()

                # test generation result
                with torch.no_grad():
                    demo_traces = gen_group(gen_model, scenario, 10, rl_gen_step_limit, state_len_limit, context_length, rl_temperature)
                    demo_eqs = [trace.final_eq for trace in demo_traces]
                    demo_intere_vals = test_intere_mp(vampire, scenario, demo_eqs, timeout)
                    demo_record = ""
                    for i in range(len(demo_traces)):
                        demo_record += f"INTERE {demo_intere_vals[i]:.3f}\t{demo_eqs[i]}\n"

                # output
                print(f"{ckpt_folder}\t Step {t}")
                print(f"Gen Pseudo Loss: {gen_avg_pseudo_loss:.3f}\tGen Avg Reward: {gen_avg_reward:.3f}")
                print(f"Raw Grad Norm: {gen_raw_grad_norm:.3f}")
                print("Demo:")
                print(demo_record)


                # Logging
                if logging:
                    writer.add_scalar("gen pseudo loss", gen_avg_pseudo_loss, t)
                    writer.add_scalar("gen avg reward", gen_avg_reward, t)
                    writer.add_scalar("gen raw grad norm", gen_raw_grad_norm, t)
                    writer.add_text("demo", demo_record, t)

                t += 1

                if save_interval is not None and t % save_interval == 0:
                    save_labs()

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
            save_labs()

    print("Training completed and model saved.")
