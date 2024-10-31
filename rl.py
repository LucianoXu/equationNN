import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from tqdm import tqdm
from typing import Optional
from train import save_checkpoint, load_checkpoint

from data import full_path_examples, ExampleDataset, get_collate_fn
from model import ModelArgs, Transformer
from proofkernel import solve_kernel_group
from small_args import SmallArgs

def rl_train(
        model_args: ModelArgs, 
        output_path: str, check_point: Optional[str] = None,
        device: str = 'cpu', 
        lr = 5e-5,
        num_steps: int = 200, 
        batch_size: int = 10, 
        batch_mutiplier: int = 10,
        max_step: int = 3, 
        max_height: int = 3,
        rl_step_limit: int = 20,
        rl_temperature: float = 0.6,):

    model = Transformer(model_args, device)
    # Set up the optimizer
    optimizer = AdamW(model.parameters())

    if check_point:
        load_checkpoint(check_point, model, optimizer)

    # Adjust hyperparameters for each parameter group
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = lr * 0.1

    # Custom Training Loop
    model.train()  # Set model to training mode

    try:
        for step in range(num_steps):
            optimizer.zero_grad(set_to_none=True)  # Clear gradients

            print(f"Step {step + 1}/{num_steps}")


            # STEP 2: calculate the pseudo loss
            total_reward = 0.

            for _ in range(batch_mutiplier):
                # STEP 1: sample the traces
                # generate starting terms
                examples = full_path_examples(batch_size, max_step, max_height)
                # get the traces
                traces = solve_kernel_group(model, list(examples), rl_step_limit, rl_temperature)


                # calculate baseline (average total reward)
                for trace in traces:
                    for i in range(len(trace)):
                        total_reward += trace[i][2]
                avg_total_reward = total_reward / len(traces)

                J = torch.tensor(0.0, device=device)

                for trace in traces:
                    for i in range(len(trace)):
                        _, log_prob, reward_to_go = trace[i]

                        # calculate the reward to go
                        for j in range(i+1, len(trace)):
                            _, _, r = trace[j]
                            reward_to_go += r

                        J += log_prob * (reward_to_go - avg_total_reward)
                J = -J / len(traces)


                # STEP 3: Backward pass and optimization
                J.backward()        # Backward pass


            optimizer.step()       # Update weights

            # Logging
            print(f"Average Total Reward: {total_reward / len(traces) / batch_mutiplier}")
            save_checkpoint(model, optimizer, output_path)
            print(f"Model saved to {output_path}.")


    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"An error of type {type(e)} occurred: {e}")

    finally:
        save_checkpoint(model, optimizer, output_path)

    print("Training completed and model saved.")

if __name__ == '__main__':
    model_args = SmallArgs()
    rl_train(model_args,
            output_path = f'small_rl_6.pth',
            check_point = f'small_rl_5.pth',
            device = 'cuda',
            lr = 5e-6,
            num_steps = 200,
            batch_size = 10,
            batch_mutiplier = 14,
            rl_step_limit=24,
            rl_temperature=0.6,
            max_step=6)


    # for max_step in range(4, 10):
    #     rl_train(model_args,
    #         output_path = f'small_rl_{max_step}.pth',
    #         check_point = f'small_rl_{max_step-1}.pth',
    #         device = 'cuda',
    #         num_steps = 200,
    #         batch_size = 20,
    #         batch_mutiplier = 6,
    #         max_step=max_step)