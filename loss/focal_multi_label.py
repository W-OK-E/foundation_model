import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Iterable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', 
                 task_type='multi-label', num_classes=None, 
                 ignore_index=None, batch_sz = None, img_sz = None):        
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
        if(isinstance(alpha,Iterable) and type(alpha) != str):
            self.alpha = torch.zeros((batch_sz,num_classes,img_sz[0],img_sz[1]))
            for idx,alp in enumerate(alpha):
                self.alpha[:,idx,:,:] = alp
        else:
            raise
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        


    def forward(self, inputs, targets):
        """
        Focal loss for multi-label classification. 
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        print("="*70)
        print("Inside Mulilabel targets shape:",targets.shape,"Inputs Shape:",inputs.shape)
        print("="*70)
        import ipdb
        ipdb.set_trace()
        probs = torch.sigmoid(inputs)
        
        #Targets are of the shape: B x H x W x N_c changing that
        targets = targets.permute(0,3,1,2)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            print("Alpha is:",self.alpha.shape)
            print("Targets Shape:",targets.shape)
            print("BCE Loss Shape:",bce_loss.shape)
            import ipdb
            ipdb.set_trace()
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss