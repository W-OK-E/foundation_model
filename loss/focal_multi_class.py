import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Iterable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None, ignore_index = None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, Iterable) and not (type(alpha) == str):
            print("Class weights Set")
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss for multi-class semantic segmentation.
        :param inputs: Predictions (logits) from the model.
                    Shape: (batch_size, num_classes, height, width)
        :param targets: Ground truth labels.
                        Shape: (batch_size, height, width)
        :return: Focal loss value
        """

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Store original spatial dimensions
        batch_size, num_classes, height, width = inputs.shape
        
        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)  # Shape: (B, C, H, W)
        
        # Flatten spatial dimensions for easier computation
        # Reshape to (B*H*W, C)
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        targets_flat = targets.view(-1).long()
        
        # Handle ignore_index if specified
        if self.ignore_index is not None:
            valid_mask = targets_flat != self.ignore_index
            probs = probs[valid_mask]
            targets_flat = targets_flat[valid_mask]
        
        # One-hot encode the targets: (B*H*W, C)
        targets_one_hot = F.one_hot(targets_flat, num_classes=self.num_classes).float()
        
        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs.clamp(min=1e-7))  # Added clamping for numerical stability
        
        # Compute p_t for each sample
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # Shape: (B*H*W,)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets_flat)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss
        
        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss  # Shape: (B*H*W, C)
        loss = loss.sum(dim=1)  # Sum over classes: (B*H*W,)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss.view(batch_size, height, width)  # Reshape back if reduction='none'
