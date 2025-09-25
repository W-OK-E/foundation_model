import torch
from torchmetrics import Metric
import numpy as np


class SegmentationMetrics(Metric):
    """
    Computes the Mean IoU and Dice Score for semantic segmentation. Does it both per-class and overall.

    Args:
        num_classes (int): number of semantic classes.
        ignore_index (int): ground truth index to ignore in the metrics.
    """

    def __init__(self, num_classes, class_names, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.conf_matrix = np.zeros((num_classes, num_classes))

    def update(self, pred, gt):
        """
        Update the confusion matrix.
        Args:
            pred: B x T x H x W (predicted logits)
            gt: B x H x W (ground truth labels)
        """
        assert(len(pred.shape) == 4) #B x C x H x W
        assert(len(gt.shape) == 3) #B x H x W

        pred = torch.argmax(pred, dim=1) #B x H x W (predicted labels from logits)

        gt = gt.flatten().cpu().numpy()
        pred = pred.flatten().cpu().numpy()

        if self.ignore_index is not None:
            mask = (gt != self.ignore_index)
            gt = gt[mask]
            pred = pred[mask]

        self.conf_matrix += np.bincount(
            self.num_classes * gt.astype(int) + pred.astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        conf_mat = self.conf_matrix

        # mIoU
        den_iou = np.sum(conf_mat, axis=1) + np.sum(conf_mat, axis=0) - np.diag(conf_mat)
        per_class_iou = np.divide(
            np.diag(conf_mat),
            den_iou,
            out=np.zeros_like(den_iou),
            where=den_iou != 0
        )
        miou = np.nanmean(per_class_iou) * 100

        # Dice Score (Per Class)
        den_dice = np.sum(conf_mat, axis=1) + np.sum(conf_mat, axis=0)
        per_class_dice = np.divide(
            2 * np.diag(conf_mat),
            den_dice,
            out=np.zeros_like(den_dice),
            where=den_dice != 0
        )
        mean_dice = np.nanmean(per_class_dice) * 100

        output = {
            "miou": miou,
            "mean_dice": mean_dice
        }
        for class_id in range(self.num_classes):
            output[f'class_{class_id}_iou'] = per_class_iou[class_id] * 100
            output[f'class_{class_id}_dice'] = per_class_dice[class_id] * 100

        # Reset confusion matrix after compute
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))

        return output
