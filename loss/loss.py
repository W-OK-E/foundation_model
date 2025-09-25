import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import roi_align


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):  
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



class weighted_maskrcnn_loss(nn.Module):
    def __init__(self):
        super(weighted_maskrcnn_loss, self).__init__()


    def project_masks_on_boxes(self,gt_masks: torch.Tensor,
                           boxes: torch.Tensor,
                           matched_idxs: torch.Tensor,
                           M: int) -> torch.Tensor:
        """
        Given segmentation masks and the bounding boxes corresponding
        to the location of the masks in the image, this function
        crops and resizes the masks in the position defined by the
        boxes. This prepares the masks for them to be fed to the
        loss computation as the targets.
        """
        matched_idxs = matched_idxs.to(boxes)
        rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
        gt_masks = gt_masks[:, None].to(rois)
        return roi_align(gt_masks, rois, (M, M), 1.)[:, 0]

    def forward(self, mask_logits, 
                proposals, 
                gt_masks, 
                gt_labels, 
                mask_matched_idxs, 
                weight):
        
        discretization_size = mask_logits.shape[-1]
        labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
        mask_targets = [
            self.project_masks_on_boxes(m, p, i, discretization_size)
            for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
        ]

        labels = torch.cat(labels, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0)

        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[torch.arange(labels.shape[0], device=labels.device), labels],
            mask_targets,
            weight
        )
        return mask_loss