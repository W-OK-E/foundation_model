import torch
from torch import nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5, do_bg=True):
        super(SoftDiceLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        
        # y is class indices, convert to one-hot
        num_classes = x.shape[1]
        if y.dim() == x.dim() - 1:
            y_onehot = F.one_hot(y.long(), num_classes=num_classes)
            if x.dim() == 4: # 2D
                y_onehot = y_onehot.permute(0, 3, 1, 2)
            else: # 3D
                y_onehot = y_onehot.permute(0, 4, 1, 2, 3)
        else:
            y_onehot = y

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        if loss_mask is not None:
            # loss_mask shape should be same as x excluding channel dim
            x = x * loss_mask
            y_onehot = y_onehot * loss_mask

        # Flatten
        axes = tuple(range(2, x.dim()))
        intersect = torch.sum(x * y_onehot, axes)
        denom = torch.sum(x + y_onehot, axes)
        
        dice = (2. * intersect + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs=None, ce_kwargs=None, weight_ce=1, weight_dice=1):
        super(DC_and_CE_loss, self).__init__()
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {}
        if ce_kwargs is None:
            ce_kwargs = {}
        
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=lambda x: F.softmax(x, dim=1), **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target.long()) if self.weight_ce != 0 else 0
        
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss
