from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from treenn import *
from randomgen import *
from datagen import *
from model import *

from torch.utils.tensorboard import SummaryWriter

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def greedy_decode(model, tree: Tree, max_height, width, max_length, device='cpu'):
    '''
    Greedy decode the tree
    '''
    model.eval()
    with torch.no_grad():
        term_data, pos_list, pos_instruct = get_model_input_from_term(tree, max_height, width, term_tokenizer)
        input, pos_list, pos_instruct, mask = model_input_add_padding(term_data, pos_list, pos_instruct, max_length)

        logits = model.forward(input, mask, pos_instruct)

        # select the most probable operation for every node token
        token_prob, token_choice = torch.max(logits, dim=1)


        # choose the most probable token
        chosen_token = torch.argmax(token_prob[0]).item()
        chosen_opt = token_choice[0][chosen_token].item()

        print(logits)

        return chosen_opt, pos_list[chosen_token]


def get_dataloader(config):

    height, max_height, path_length, n, max_length = config['height'], config['max_height'], config['path_length'], config['n'], config['max_length']
    ds = synthesize_example_thread(height, max_height, path_length, n, max_length=max_length)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * n)
    train_ds_raw, val_ds_raw = ds[:train_ds_size], ds[train_ds_size:]

    train_ds = InverseDataset(train_ds_raw, max_length)
    val_ds = InverseDataset(val_ds_raw, max_length)

    def collate_fn(batch):
        input = [b['input'] for b in batch]
        input_mask = [b['input_mask'] for b in batch]
        pos_list = [b['pos_list'] for b in batch]
        pos_inst = [b['pos_inst'] for b in batch]
        label = [b['label'] for b in batch]

        return {
            'input': torch.stack(input, dim=0),
            'input_mask': torch.stack(input_mask, dim=0),
            'pos_list': pos_list,
            'pos_inst': pos_inst,
            'label': torch.stack(label, dim=0)
        }


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    return train_dataloader, val_dataloader


def train_model(config, modelArgs: ModelArgs):

    # Define the device
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_dataloader(config)

    model = Transformer(modelArgs).to(device)
    
    # print the parameter number of model
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    model_filename = None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')


    # create a weighted cross entropy loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            inputs = batch['input'].to(device) # (B, seq_len)
            masks = batch['input_mask'].to(device) # (B, 1, 1, seq_len)
            pos_instructs = batch['pos_inst']

            logits_output = model.forward(inputs, masks, pos_instructs) # (B, seq_len, opt_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss. the loss only results from the position of operation label
            pred = logits_output.transpose(1, 2)[range(logits_output.shape[0]), label[:, 0]]

            loss = loss_fn(pred, label[:, 1])
            
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # # Run validation at the end of every epoch
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


def load_model(config, modelArgs: ModelArgs):

    device = "cpu"
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_dataloader(config)

    model = Transformer(modelArgs).to(device)
    
    # print the parameter number of model
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Tensorboard

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')


    return model

import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = {
        'height': 3,
        'max_height': 20,
        'path_length': 1,
        'n': 20000,
        'max_length': 100,
        'batch_size': 192,
        'lr': 1e-4,
        'num_epochs': 5,

        'datasource': 'thread',
        'model_basename': 'treenn_',
        'model_folder': 'weights',
        'preload': 'latest',
        'experiment_name': 'runs/treenn',
    }
    args = ModelArgs(
        dim = 32,
        n_layers = 3,
        n_heads = 4,
    )
    args.vocab_size = len(term_tokenizer)
    args.output_size = len(opt_tokenizer)
    train_model(config, args)