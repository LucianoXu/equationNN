import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.adamw import AdamW
from tqdm import tqdm
from typing import Optional
from elab import ELab, set_adamw_params, get_grad_norm

from data import full_path_examples, ExampleDataset, get_collate_fn
from model import ModelArgs, Llama3
from tokenizer import token2id
from proofkernel import solve_kernel_group
from small_args import SmallArgs

def rl_train(
        model: Llama3,

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
        save_interval: int = 10,

        # reinforcement learning settings
        max_step: int = 3, 
        max_height: int = 3,
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
        for step in range(num_steps):

            print(f"Step {step + 1}/{num_steps}")


            # STEP 2: calculate the pseudo loss
            total_reward = 0.
            total_pseudo_loss = 0.
            total_grad_norm = 0.

            for _ in range(accumulaton_step):
                # STEP 1: sample the traces
                # generate starting terms
                examples = full_path_examples(batch_size, max_step, max_height)

                # get the traces
                traces = solve_kernel_group(model, list(examples), rl_step_limit, rl_temperature)
                ###########################

                # STEP 2: calculate the pseudo loss
                # calculate baseline (average total reward)
                batch_reward = 0.
                for trace in traces:
                    for i in range(len(trace)):
                        batch_reward += trace[i][2]
                avg_batch_reward = batch_reward / batch_size

                J = torch.tensor(0.0, device=device)

                for trace in traces:
                    for i in range(len(trace)):
                        _, log_prob, reward_to_go = trace[i]

                        # calculate the reward to go
                        for j in range(i+1, len(trace)):
                            _, _, r = trace[j]
                            reward_to_go += r

                        J += log_prob * (reward_to_go - avg_batch_reward)
                J = -J / len(traces)

                total_pseudo_loss += J.item()
                total_reward += batch_reward


                # STEP 3: Backward pass and optimization
                J.backward()        # Backward pass

            raw_grad_norm = get_grad_norm(model)

            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)


            optimizer.step()       # Update weights
            optimizer.zero_grad()

            # Logging
            avg_pseudo_loss = total_pseudo_loss / accumulaton_step
            avg_reward = total_reward / accumulaton_step
            
            
            # log the loss
            print(f"{ckpt_folder}\tStep {t}\tPseudo Loss: {avg_pseudo_loss:.3f}\tAvg Reward: {avg_reward:.3f}\tRaw Grad Norm: {raw_grad_norm:.3f}")
            writer.add_scalar("pseudo_loss", avg_pseudo_loss, t)
            writer.add_scalar("avg reward", avg_reward, t)
            writer.add_scalar("raw grad norm", raw_grad_norm, t)

            t += 1

            if t % save_interval == 0:
                lab.states['t'] = t
                lab.save(f"RL{t}")


    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"An error of type {type(e)} occurred: {e}")

    finally:
        lab.states['t'] = t
        lab.save(f"RL{t}")

    print("Training completed and model saved.")

if __name__ == '__main__':
    rl_train(
        Llama3(
            vocab_size=len(token2id),
            context_length = 160,
            dim=512,
            num_layers=32,
            num_heads=16,
            d_ff=2048,
            device='mps'
        ),
        ckpt_folder = "./ckpt/VSuper",
        input_version_name = 'latest',

        lr = 5e-6,
        weight_decay=0.01,
        betas=(0.9, 0.99),
        grad_norm_clip=1.0,

        num_steps = 200,
        batch_size = 10,
        accumulaton_step = 14,
        rl_step_limit=24,
        rl_temperature=0.6,
        max_step=6
    )