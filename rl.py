import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from tqdm import tqdm
from typing import Optional

from data import full_path_examples, ExampleDataset, get_collate_fn
from model import ModelArgs, Transformer
from proofkernel import solve_kernel_group
from small_args import SmallArgs

def rl_train(
        model_args: ModelArgs, 
        model_path: str, check_point: Optional[str] = None,
        device: str = 'cpu', 
        num_steps: int = 200, 
        batch_size: int = 16, 
        max_step: int = 6, 
        max_height: int = 3,
        rl_step_limit: int = 20,
        rl_temperature: float = 0.6,):

    model = Transformer(model_args, device)
    if check_point:
        model.load_state_dict(torch.load(check_point, map_location=device, weights_only=True))

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Custom Training Loop
    model.train()  # Set model to training mode

    try:
        for step in range(num_steps):
            optimizer.zero_grad(set_to_none=True)  # Clear gradients

            print(f"Step {step + 1}/{num_steps}")

            torch.mps.empty_cache()

            # STEP 1: sample the traces
            # generate starting terms
            examples = full_path_examples(batch_size, max_step, max_height)
            # get the traces
            traces = solve_kernel_group(model, list(examples), rl_step_limit, rl_temperature)

            # STEP 2: calculate the pseudo loss
            print("Calculating average total reward...", end='')
            total_reward = 0.
            # calculate baseline (average total reward)
            for trace in traces:
                for i in range(len(trace)):
                    total_reward += trace[i][2]
            avg_total_reward = total_reward / len(traces)
            print("Done.")

            print("Calculating pseudo loss...", end='')
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
            print("Done.")


            # STEP 3: Backward pass and optimization
            print("Calculating gradients...", end='')
            J.backward()        # Backward pass
            print("Backpropagation...", end='')
            optimizer.step()       # Update weights
            print('Done.')

            # Logging
            print(f"Pseudo Loss: {J.item()}, Average Total Reward: {total_reward / len(traces)}")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"An error of type {type(e)} occurred: {e}")

    finally:
        torch.save(model.state_dict(), model_path)

    print("Training completed and model saved.")

if __name__ == '__main__':
    model_args = SmallArgs()
    rl_train(model_args,
            model_path = 'small_rl.pth',
            check_point = 'small.pth',
            device = 'mps')