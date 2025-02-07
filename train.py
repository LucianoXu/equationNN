from typing import Literal, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW
from tqdm import tqdm
import json
import os

from model import token2id, ModelArgs, Llama3, SmallArgs, MiddleArgs

from ds import ExampleDataset, get_collate_fn

from elab import ELab, get_grad_norm

device = json.load(open('config.json'))['backend']

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def train(
        rank,
        world_size,
        model_args: ModelArgs,
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

    print(f"--------Training on rank {rank}")
    setup(rank, world_size)    

    device = torch.device(f"cuda:{rank}")

    model = Llama3(model_args = model_args, device = device)
    model = DDP(model, device_ids = [rank])

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
            train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=get_collate_fn(str(device)), shuffle=False, sampler=DistributedSampler(dataset))

            
            # Use tqdm for progress bar
            epoch_iterator = tqdm(train_dataloader, desc="Training", position=0, leave=True)


            for i, batch in enumerate(epoch_iterator):

                # Move batch data to the device
                text, label, loss_mask = batch
                text = text.to(device)
                label = label.to(device)
                loss_mask = loss_mask.to(device)
                
                # Forward pass
                logits = model(text)

                # Flatten the logits and labels to compute loss
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.reshape(-1))
                # Apply mask to the loss
                loss = loss * loss_mask.view(-1).float()
                loss = loss.sum() / loss_mask.sum().float() # Average loss over non-padding tokens
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

                if t % save_interval == 0 and rank == 0:
                    lab.states['t'] = t
                    # lab.model = model.module
                    lab.save(version_name=str(t))

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"An error of type {type(e)} occurred: {e}")

    finally:
        lab.states['t'] = t
        if rank == 0:
            # lab.model = model.module
            lab.save(version_name=str(t))

        cleanup()

    print("Training completed and model saved.")

def run_train(
        world_size: int,
        model_args: ModelArgs,
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
    torch.multiprocessing.spawn(train, 
        args=(
            world_size,
            model_args,
            context_length,
            ckpt_folder, 
            load_version_name,

            # optimizer
            lr,
            weight_decay, 
            betas, 
            eps,
            grad_norm_clip,

            num_epochs, 
            epoch_data_length, 
            batch_size, 
            max_step, 
            max_height,

            save_interval,
        ), nprocs=world_size, join=True)

if __name__ == "__main__":
    args = MiddleArgs()
    run_train(
        world_size=torch.cuda.device_count(),

        model_args = args,
        context_length = args.context_length,
        ckpt_folder='./ckpt/OMLgenL',
        load_version_name='none',

        lr = 2e-5,
        weight_decay=0.01,
        grad_norm_clip=1.0,
        betas=(0.9, 0.99),


        num_epochs = 10, 
        epoch_data_length = 100000, 
        batch_size = 64, 
        max_step = 6, 
        max_height = 3,

        save_interval = 5000,
    )