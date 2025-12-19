import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Iterable

class WeightedCrossEntropyDiceLoss(nn.Module):
    """Weighted CrossEntropy + Dice Loss for multiclass segmentation"""
    def __init__(self, alpha=None, dice_weight=0.5, ignore_index = 0,num_classes = 0):
        super(WeightedCrossEntropyDiceLoss, self).__init__()
        if isinstance(alpha, Iterable) and not (type(alpha) == str):
            print("Class weights Set")
            self.class_weights = torch.Tensor(alpha)
        else:
            self.class_weights = alpha
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def cross_entropy_loss(self, inputs, targets):
        """CrossEntropy loss with class weights"""
        # inputs: [B, C, H, W], targets: [B, H, W]
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')
        return ce_loss
    
    def dice_loss(self, inputs, targets, smooth=1e-5):
        """Dice loss with class weights for multiclass"""
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)  # [B, C, H, W]
        
        # Convert targets to one-hot encoding
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Calculate dice for each class
        dice_scores = []
        for c in range(num_classes):
            input_c = probs[:, c].flatten()
            target_c = targets_one_hot[:, c].flatten()
            
            intersection = (input_c * target_c).sum()
            dice_score = (2. * intersection + smooth) / (input_c.sum() + target_c.sum() + smooth)
            dice_scores.append(dice_score)
        
        # Convert to tensor and apply weights
        dice_scores = torch.stack(dice_scores)
        
        if self.class_weights is not None:
            # Apply weights: weighted average of (1 - dice_score) for each class
            losses = 1 - dice_scores
            # print("Unweighted loss:",dice_scores.mean().item())
            weighted_loss = (losses * self.class_weights).sum() / self.class_weights.sum()
            # print("Weighted Loss mean:",weighted_loss.item())
            return weighted_loss
        else:
            return 1 - dice_scores.mean()
    
    def forward(self, inputs, targets):
        print("Shape:",inputs.shape,targets.shape)
        self.class_weights = self.class_weights.to(inputs.device)
        ce = self.cross_entropy_loss(inputs, targets.long())
        dice = self.dice_loss(inputs, targets.long())
        # print("CE Loss:",ce.item()," Dice Loss:",dice.item())
        final_loss = (1-self.dice_weight) * ce + self.dice_weight * dice
        # print("Final Weighted Loss:",final_loss.item())
        return final_loss