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
        batch_size: int = 10, 
        batch_mutiplier: int = 10,
        max_step: int = 3, 
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
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}.")


    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"An error of type {type(e)} occurred: {e}")

    finally:
        torch.save(model.state_dict(), model_path)

    print("Training completed and model saved.")

if __name__ == '__main__':
    model_args = SmallArgs()
    for max_step in range(4, 10):
        rl_train(model_args,
            model_path = f'small_rl_{max_step}.pth',
            check_point = f'small_rl_{max_step-1}.pth',
            device = 'cuda',
            num_steps = 200,
            batch_size = 6,
            batch_mutiplier = 20,
            max_step=max_step)