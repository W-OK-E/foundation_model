import torch
from torchmetrics import MetricCollection
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAveragePrecision,
)


class SegmentationMetrics(Metric):
    """
    Computes the Mean IoU and Dice Score for semantic segmentation using TorchMetrics.
    Does it both per-class and overall.

    Args:
        num_classes (int): number of semantic classes.
        class_names (list): list of class names.
        ignore_index (int): ground truth index to ignore in the metrics.
        multi_label (bool): whether the task is multi-label classification.
    """

    def __init__(self, num_classes, class_names, ignore_index=None, multi_label=False, is_3d=False):

        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.multi_label = multi_label
        self.is_3d = is_3d
        
        # Create metric collection with all metrics
        metrics = {
            # Mean metrics
            'miou': MulticlassJaccardIndex(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average='macro'
            ),
            'mf': MulticlassF1Score(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average='macro'
            ),
            
            # Per-class IoU
            'per_class_iou': MulticlassJaccardIndex(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average='none'
            ),
            
            # Per-class F1
            'per_class_f': MulticlassF1Score(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average='none'
            ),
            
            # Per-class Precision
            'per_class_precision': MulticlassPrecision(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average='none'
            ),
            
            # Per-class Recall
            'per_class_recall': MulticlassRecall(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average='none'
            ),
        }
        
        self.metrics = MetricCollection(metrics)

    def to(self, *args, **kwargs):
        self.metrics = self.metrics.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Update all metrics.
        Args:
            pred: B x C x H x W (predicted logits)
            gt: B x H x W (ground truth labels) or B x H x W x C (multi-label)
        """
        pred = pred.detach()
        gt = gt.detach()

        if not self.is_3d:
            assert len(pred.shape) == 4, "pred must be B x C x H x W"    
            # Flatten spatial dimensions: B x C x H x W -> (B*H*W) x C
            B, C, H, W = pred.shape
            pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W) x C
        else:
            assert len(pred.shape) == 5, "pred must be B x C x D x H x W"    
            # Flatten spatial dimensions: B x C x D x H x W -> (B*D*H*W) x C
            B, C, D, H, W = pred.shape
            pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*D*H*W) x C
            
        gt_flat = gt.reshape(-1)  # (B*H*W)
        gt_flat = gt_flat.long()
        self.metrics = self.metrics.to(pred_flat.device)
        self.metrics.update(pred_flat, gt_flat)

    def compute(self):
        """
        Compute all metrics.
        Returns:
            dict: Dictionary containing all computed metrics.
        """
        results = self.metrics.compute()
        
        # Build output dictionary with percentage scaling
        mean_output = {
            "miou": (results['miou'] * 100).item(),
            "mf": (results['mf'] * 100).item(),
        }
        class_output = {}
        # Add per-class metrics
        per_class_iou = results['per_class_iou']
        per_class_f = results['per_class_f']
        per_class_precision = results['per_class_precision']
        per_class_recall = results['per_class_recall']
        
        for class_id in range(self.num_classes):
            class_output[f'class_{class_id}_iou'] = (per_class_iou[class_id] * 100).item()
            class_output[f'class_{class_id}_f'] = (per_class_f[class_id] * 100).item()
            class_output[f'class_{class_id}_precision'] = (per_class_precision[class_id] * 100).item()
            class_output[f'class_{class_id}_recall'] = (per_class_recall[class_id] * 100).item()
        
        return mean_output,class_output

    def reset(self):
        """Reset all metrics."""
        self.metrics.reset()


