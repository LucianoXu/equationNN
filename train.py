import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from tqdm import tqdm

from data import ExampleDataset, collate_fn
from model import ModelArgs, Transformer

def train():
    device = 'mps'
    SEQ_LEN = 96
    BATCH_SIZE = 16
    DATA_LEN = 100000
    MAX_STEP = 7
    MAX_HEIGHT = 4
    MODEL_PATH = 'trained_parameters.pth'
    CHECK_POINT = 'trained_parameters.pth'
    # CHECK_POINT = None

    # Step 1
    dataset = ExampleDataset(DATA_LEN, MAX_STEP, MAX_HEIGHT)

    # Step 2
    model_args = ModelArgs()
    model_args.max_seq_len = SEQ_LEN
    model = Transformer(ModelArgs(), device)
    if CHECK_POINT:
        model.load_state_dict(torch.load(CHECK_POINT, map_location=device, weights_only=True))

    # Step 3
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Step 5: Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Step 6: Move model to GPU if available
    device = torch.device(device)
    model.to(device)

    # Step 7: Custom Training Loop
    num_epochs = 3
    model.train()  # Set model to training mode

    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')


    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
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
            torch.save(model.state_dict(), MODEL_PATH)
            # tokenizer.save_pretrained("./custom_trained_model")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        torch.save(model.state_dict(), MODEL_PATH)

    print("Training completed and model saved.")

if __name__ == "__main__":
    train()