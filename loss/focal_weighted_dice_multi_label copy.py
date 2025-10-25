import torch
import torch.nn as nn


class WeightedDiceFocalLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5,ignore_index = 0):
        super(WeightedDiceFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """Manual focal loss implementation with class weights"""
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Expand weights to match input shape
            weights = self.class_weights.view(1, -1, 1, 1).expand_as(targets)
            focal_loss = focal_loss * weights
            
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets, smooth=1e-5):
        """Dice loss with class weights"""
        inputs = torch.sigmoid(inputs)
        
        # Calculate dice for each class
        dice_scores = []
        for c in range(inputs.shape[1]):  # For each class
            input_c = inputs[:, c].flatten()
            target_c = targets[:, c].flatten()
            
            intersection = (input_c * target_c).sum()
            dice_score = (2. * intersection + smooth) / (input_c.sum() + target_c.sum() + smooth)
            dice_scores.append(dice_score)
        
        # Convert to tensor and apply weights
        dice_scores = torch.stack(dice_scores)
        
        if self.class_weights is not None:
            # Weight the dice scores
            weighted_dice = dice_scores * self.class_weights
            dice_loss = 1 - weighted_dice.mean()
        else:
            dice_loss = 1 - dice_scores.mean()
            
        return dice_loss
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return (1-self.dice_weight) * focal + self.dice_weight * dice