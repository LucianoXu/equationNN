from typing import Literal, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW
from tqdm import tqdm

from tokenizer import token2id

from data import ExampleDataset, get_collate_fn
from model import ModelArgs, Llama3

from elab import ELab, get_grad_norm

def train(
        model: Llama3,
        context_length: int,
        ckpt_folder: str, 
        load_version_name: str|Literal['latest', 'none'],

        # optimizer
        lr: float,
        weight_decay: float, 
        betas: tuple[float, float], 
        eps = 1e-8,
        grad_norm_clip: Optional[float] = None,

        num_epochs: int = 10, 
        epoch_data_length: int = 100000, 
        batch_size: int = 32, 
        max_step: int = 6, 
        max_height: int = 3,

        save_interval: int = 1000,
        ):
    

    # get device
    device = next(model.parameters()).device

    # build optimizer
    optimizer = AdamW(
        model.parameters(),
        lr = lr, betas = betas, weight_decay=weight_decay,
        eps = eps
    )

    # create/load the checkpoint
    # here t represents the next step number to be executed
    lab = ELab(
        ckpt_folder, 
        version_name=load_version_name,
        model = model,
        optimizer = optimizer,
        default_states={
            't': 1,
        }
    )

    model.train()
    optimizer.zero_grad()

    t: int = lab.states['t']

    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # tensorboard logger
    writer = SummaryWriter(lab.folder_path)

    try:
        for epoch in range(num_epochs):

            print(f"Epoch {epoch + 1}/{num_epochs}")

            # generate training data
            dataset = ExampleDataset(epoch_data_length, max_step, max_height, context_length)
            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=get_collate_fn(str(device)))

            
            # Use tqdm for progress bar
            epoch_iterator = tqdm(train_dataloader, desc="Training", position=0, leave=True)


            for i, batch in enumerate(epoch_iterator):

                # Move batch data to the device
                text, label, mask = batch
                text = text.to(device)
                label = label.to(device)
                mask = mask.to(device)
                
                # Forward pass
                logits = model(text)

                # Flatten the logits and labels to compute loss
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.reshape(-1))
                # Apply mask to the loss
                loss = loss * mask.view(-1).float()
                loss = loss.sum() / mask.sum().float() # Average loss over non-padding tokens
                loss.backward()

                raw_grad_norm = get_grad_norm(model)

                if grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)

                optimizer.step()       # Update weights
                optimizer.zero_grad()  # Clear gradients

                print(f"{ckpt_folder}\tStep {t}\tloss: {loss.item():.3f}")

                # Log the data
                writer.add_scalar("loss", loss.item(), t)
                writer.add_scalar("raw grad norm", raw_grad_norm, t)

                t += 1

                if t % save_interval == 0:
                    lab.states['t'] = t
                    lab.save(version_name=str(t))

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"An error of type {type(e)} occurred: {e}")

    finally:
        lab.states['t'] = t
        lab.save(version_name=str(t))

    print("Training completed and model saved.")

if __name__ == "__main__":
    from small_args import SmallArgs
    args = SmallArgs()
    train(
        Llama3(
            model_args = args,
            device='cuda'
        ),

        context_length = args.context_length,
        ckpt_folder='./ckpt/VSuperF1',
        load_version_name='none',

        lr = 2e-4,
        weight_decay=0.01,
        grad_norm_clip=1.0,
        betas=(0.9, 0.99),


        num_epochs = 10, 
        epoch_data_length = 100000, 
        batch_size = 32, 
        max_step = 6, 
        max_height = 3,

        save_interval = 4854,
    )