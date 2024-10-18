from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from tqdm import tqdm

from data import ExampleDataset, get_collate_fn
from model import ModelArgs, Transformer

def train(
        model_args: ModelArgs, 
        model_path: str, check_point: Optional[str] = None,
        device: str = 'cpu', 
        num_epochs: int = 10, 
        data_len: int = 100000, 
        batch_size: int = 32, 
        max_step: int = 6, 
        max_height: int = 3,):

    model = Transformer(model_args, device)
    if check_point:
        model.load_state_dict(torch.load(check_point, map_location=device, weights_only=True))

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Custom Training Loop
    model.train()  # Set model to training mode

    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    try:
        for epoch in range(num_epochs):

            print(f"Epoch {epoch + 1}/{num_epochs}")

            # generate training data
            dataset = ExampleDataset(data_len, max_step, max_height, model_args.max_seq_len)
            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=get_collate_fn(device))

            
            # Use tqdm for progress bar
            epoch_iterator = tqdm(train_dataloader, desc="Training", position=0, leave=True)

            # Track loss for each epoch
            total_loss = 0

            for i, batch in enumerate(epoch_iterator):

                # Move batch data to the GPU (or CPU)
                text, label, mask = batch
                text = text.to(device)
                label = label.to(device)
                mask = mask.to(device)
                
                # Step 8: Forward pass
                outputs = model(text)

                # Shift the labels so that the model predicts the next token
                outputs = outputs.contiguous()
                label = label.contiguous()

                # Flatten the logits and labels to compute loss
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), label.view(-1))

                # Apply the mask to the loss
                loss = loss * mask.view(-1).float()  # Shape: (batch_size * seq_len)

                # Normalize the loss by the number of valid (non-padding) tokens
                loss = loss.sum() / mask.sum().float()  # Normalize by valid tokens

                # Step 9: Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()        # Backward pass
                optimizer.step()       # Update weights

                # Step 10: Logging
                total_loss += loss.item()
                epoch_iterator.set_postfix(loss=loss.item())

            # Print average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

            # Step 11: Save the model and tokenizer
            torch.save(model.state_dict(), model_path)
            # tokenizer.save_pretrained("./custom_trained_model")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"An error of type {type(e)} occurred: {e}")

    finally:
        torch.save(model.state_dict(), model_path)

    print("Training completed and model saved.")

if __name__ == "__main__":
    from small_args import SmallArgs
    train(
        SmallArgs(),
        model_path='small.pth',
        check_point=None,
        device='cuda'
    )