import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTargetCrossEntropyLoss(nn.Module):
    """
    A custom loss that extends CrossEntropyLoss to the case where there are multiple valid target tokens.
    For each sample, the loss is defined as:
    
        loss = -log(sum_{i in valid_tokens} softmax(logits)_i)
    
    If the model assigns all probability mass to any of the valid tokens, the loss is zero.
    """
    def __init__(self, reduction='mean'):
        super(MultiTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target: list[list[int]|set[int]] | torch.Tensor):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes) containing raw model outputs.
            target: Either a binary mask tensor of shape (batch_size, num_classes), where valid tokens are marked with 1,
                    or a list of lists of indices, where each inner list contains the valid target token indices for that sample.
        
        Returns:
            The computed loss as a scalar (if reduction is 'mean' or 'sum') or a tensor of losses per sample.
        """
        # Compute log probabilities in a numerically stable way.
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # If target is provided as a list of valid indices, convert it to a binary mask.
        if isinstance(target, list):
            batch_size, num_classes = logits.size()
            target_mask = torch.zeros_like(logits, dtype=torch.float, device=logits.device)
            for i, indices in enumerate(target):
                if isinstance(indices, set):
                    indices = list(indices)
                target_mask[i, indices] = 1.0
        else:
            target_mask = target.float()
        
        # Sum the probabilities over the valid tokens.
        valid_prob = (probs * target_mask).sum(dim=-1)
        
        # Compute the loss as negative log-likelihood.
        loss = -torch.log(valid_prob + 1e-8)  # added epsilon for numerical stability
        
        # Apply reduction.
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Example usage:
if __name__ == '__main__':
    # Dummy logits for a batch of 3 samples over 5 classes.
    logits = torch.tensor([[2.0, 1.0, 0.1, 0.5, 0.2],
                           [0.3, 1.2, 2.1, 0.1, 0.0],
                           [1.5, 0.2, 0.3, 2.0, 0.1]])

    # Suppose for each sample, there are multiple valid target indices.
    # For example, sample 0: valid tokens are indices 0 and 3,
    # sample 1: valid tokens are indices 2,
    # sample 2: valid tokens are indices 0 and 3.
    target_indices = [[0, 3], [0,1,2,3,4], [0, 3]]

    # Initialize the loss function.
    loss_fn = MultiTargetCrossEntropyLoss(reduction='none')
    loss = loss_fn(logits, target_indices)
    print("Loss:", loss)
