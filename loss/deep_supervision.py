import torch
from torch import nn
import torch.nn.functional as F

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weights):
        """
        Wraps a loss function to support deep supervision.
        
        Args:
            loss: The base loss function to apply at each scale.
            weights: List of weights for each scale, typically decreasing.
        """
        super(DeepSupervisionWrapper, self).__init__()
        self.loss = loss
        self.weights = weights

    def forward(self, x, y):
        """
        Args:
            x: List of predictions at different scales. x[0] is the highest resolution.
            y: Ground truth at full resolution.
        """
        if not isinstance(x, (list, tuple)):
            return self.loss(x, y)

        total_loss = 0
        for i in range(len(x)):
            if self.weights[i] == 0:
                continue
            
            # Prediction shape
            # x[i] shape: (B, C, H_i, W_i) or (B, C, D_i, H_i, W_i)
            target_shape = x[i].shape[2:]
            
            # Downsample target if necessary
            if y.shape[2:] != target_shape:
                # Use nearest neighbor for segmentation masks
                y_down = F.interpolate(y.unsqueeze(1).float() if y.dim() == 3 else y.unsqueeze(1).float(), 
                                       size=target_shape, mode='nearest').squeeze(1).long()
            else:
                y_down = y
            
            total_loss += self.weights[i] * self.loss(x[i], y_down)
            
        return total_loss
