import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import roi_align


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index=None):  
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        target = target.long()
        
        # if self.ignore_index is not None:
        #     mask = target != self.ignore_index
        #     input = input[mask]
        #     target = target[mask]

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



class Weighted_Maskrcnn_loss(nn.Module):
    def __init__(self):
        super(Weighted_Maskrcnn_loss, self).__init__()


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


#Was used in Sasvi
class Mask2FormerLoss(nn.Module):
    def __init__(self, classification_loss_fn: nn.Module, mask_loss_fn: nn.Module):
        """
        Args:
            classification_loss_fn: loss function for classification (e.g., nn.CrossEntropyLoss)
            mask_loss_fn: loss function for masks (e.g., nn.BCEWithLogitsLoss or weighted_maskrcnn_loss)
        """
        super(Mask2FormerLoss, self).__init__()
        self.classification_loss_fn = classification_loss_fn
        self.mask_loss_fn = mask_loss_fn

    def forward(self, pred_class_logits, pred_masks, gt_labels, gt_masks, matches):
        """
        Compute classification and mask losses based on Hungarian matching.

        Args:
            pred_class_logits (torch.Tensor): [num_queries, num_classes], class logits.
            pred_masks (torch.Tensor): [num_queries, height, width], predicted masks.
            gt_labels (torch.Tensor): [num_objects], ground truth class labels.
            gt_masks (torch.Tensor): [num_objects, height, width], ground truth masks.
            matches (List[Tuple[int, int]]): List of matched indices (prediction_idx, ground_truth_idx).

        Returns:
            torch.Tensor: Total loss.
        """
        if len(matches) == 0:
            # No matches, return zero loss
            return torch.tensor(0.0, requires_grad=True, device=pred_class_logits.device)

        matched_pred_indices, matched_gt_indices = zip(*matches)

        # Matched predictions and ground truth
        matched_pred_class_logits = pred_class_logits[list(matched_pred_indices)]
        matched_pred_masks = pred_masks[list(matched_pred_indices)]
        matched_gt_labels = gt_labels[list(matched_gt_indices)]
        matched_gt_masks = gt_masks[list(matched_gt_indices)]

        # Ensure gt_masks is float for BCEWithLogitsLoss
        matched_gt_masks = matched_gt_masks.float()

        # Compute classification loss
        classification_loss = self.classification_loss_fn(matched_pred_class_logits, matched_gt_labels)

        # Compute mask loss
        mask_loss = self.mask_loss_fn(matched_pred_masks, matched_gt_masks)

        return classification_loss + mask_loss

class BCEDiceLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        pred: (Batch, Classes, Height, Width) - raw logits
        target: (Batch, Height, Width) - class indices (Long) or (Batch, Classes, Height, Width) for multi-label
        """
        # Convert target to float
        if target.dtype != torch.float32:
            target = target.float()
        
        # For multi-class segmentation, you need to handle this differently
        # Option 1: If binary segmentation (2 classes), take channel 1
        if pred.shape[1] == 2:
            # Use only the positive class logits
            # input = pred[:, 1, :, :]  # Shape: (B, H, W)
            input = pred
            # BCE with logits
            bce = F.binary_cross_entropy_with_logits(input, target)
            
            # Dice loss
            smooth = 1e-5
            input_sigmoid = torch.sigmoid(input)
            num = target.size(0)
            input_flat = input_sigmoid.view(num, -1)
            target_flat = target.view(num, -1)
            intersection = (input_flat * target_flat).sum(1)
            dice = (2. * intersection + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
            dice_loss = 1 - dice.mean()
            
            return 0.5 * bce + dice_loss
        
        # Option 2: If multi-class segmentation (>2 classes)
        else:
            # Convert target from class indices to one-hot
            num_classes = pred.shape[1]
            target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
            
            # BCE with logits
            bce = F.binary_cross_entropy_with_logits(pred, target_one_hot)
            
            # Dice loss
            smooth = 1e-5
            pred_sigmoid = torch.sigmoid(pred)
            num = target.size(0)
            pred_flat = pred_sigmoid.view(num, num_classes, -1)
            target_flat = target_one_hot.view(num, num_classes, -1)
            intersection = (pred_flat * target_flat).sum(2)
            dice = (2. * intersection + smooth) / (pred_flat.sum(2) + target_flat.sum(2) + smooth)
            dice_loss = 1 - dice.mean()
            
            return 0.5 * bce + dice_loss

#The custom implementation is there to just have greater control and understanding of the 
#data type
class CELoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, input, target):
        if(target.dtype != torch.long):
            target = target.long()
        return self.ce(input, target)
