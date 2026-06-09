import torch
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (
    MultilabelPrecision,
    MultilabelF1Score,
    MultilabelRecall,
    MultilabelAveragePrecision,
    MultilabelJaccardIndex,
    MultilabelPrecisionAtFixedRecall,
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

    def __init__(self, num_classes, class_names, ignore_index=None, multi_label=True):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.multi_label = multi_label
        
        # Create metric collection with multilabel metrics
        metrics = {
            # Mean metrics
            'miou': MultilabelJaccardIndex(
                num_labels=num_classes,
                average='macro'  # mean IoU across classes
            ),
            'mf': MultilabelF1Score(
                num_labels=num_classes,
                average='macro'
            ),
            'maupr': MultilabelAveragePrecision(
                num_labels=num_classes,
                average='macro'
            ),
            # Per-class IoU
            'per_class_iou': MultilabelJaccardIndex(
                num_labels=num_classes,
                average=None  # per-class
            ),

            # Per-class F1
            'per_class_f': MultilabelF1Score(
                num_labels=num_classes,
                average=None
            ),

            # Per-class Precision
            'per_class_precision': MultilabelPrecision(
                num_labels=num_classes,
                average=None
            ),

            # Per-class Recall
            'per_class_recall': MultilabelRecall(
                num_labels=num_classes,
                average=None
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
            pred: B x C x H x W (predicted logits or probabilities)
            gt: B x C x H x W (ground truth binary masks for multi-label)
        """
        assert len(pred.shape) == 4, "pred must be B x C x H x W"
        assert len(gt.shape) == 4, "gt must be B x C x H x W for multi-label"
        B, C, H, W = pred.shape
        gt = gt.long()
        self.metrics = self.metrics.to(pred.device)
        self.metrics.update(pred, gt)

    def compute(self):
        """
        Compute all metrics.
        Returns:
            dict: Dictionary containing all computed metrics.
        """
        results = self.metrics.compute()
        
        # Mean metrics
        mean_output = {
            "miou": (results['miou'] * 100).item(),
            "mf": (results['mf'] * 100).item(),
            "maupr": (results['maupr'] * 100).item()
        }

        # Per-class metrics
        class_output = {}
        per_class_iou = results['per_class_iou']
        per_class_f = results['per_class_f']
        per_class_precision = results['per_class_precision']
        per_class_recall = results['per_class_recall']
        
        for class_id in range(self.num_classes):
            class_output[f'class_{class_id}_iou'] = (per_class_iou[class_id] * 100).item()
            class_output[f'class_{class_id}_f'] = (per_class_f[class_id] * 100).item()
            class_output[f'class_{class_id}_precision'] = (per_class_precision[class_id] * 100).item()
            class_output[f'class_{class_id}_recall'] = (per_class_recall[class_id] * 100).item()
        
        return mean_output, class_output

    def reset(self):
        """Reset all metrics."""
        self.metrics.reset()


if __name__ == "__main__":
    metrics = SegmentationMetrics(6,list('abcdef'),None,True)
    t1 = torch.load('/home/asavari/foundation_model/datasets/IDRiD/masks_pt/IDRiD_01.pt')
    print("Mask LOaded:",t1.shape)
    t1 = t1.unsqueeze(0).permute(0,3,1,2)
    t2 = t1
    metrics.update(t1,t2)
    mean,cls = metrics.compute()
    print(mean)