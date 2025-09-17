import pytorch_lightning as L
import torch
import torch.nn as nn
from PIL import Image
from .network.ElitNet import ElitNet
from hydra.utils import instantiate

class ElitLightModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.network.instance)
        self.loss = instantiate(cfg.loss.instance)
        self.ignore_index = self.loss.ignore_index #If we want to ignore a class in the loss computation
        self.train_metrics = instantiate(cfg.train_metrics)
        self.val_metrics = instantiate(cfg.val_metrics)
        self.test_metrics = instantiate(cfg.test_metrics)

    def training_step(self, batch):
        image,gt_mask = batch
        image,gt_mask = image.float(), gt_mask.long()
        pred = self.model(image)
        loss = self.loss(pred, gt_mask)
        self.train_metrics.update(pred, gt_mask)
        self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        metrics = self.train_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    @torch.no_grad()
    def validation_step(self, batch:list):
        image,gt_mask = batch
        image,gt_mask = image.float(), gt_mask.long()
        pred = self.model(image)    
        loss = self.loss(pred, gt_mask)
        self.val_metrics.update(pred, gt_mask)
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    @torch.no_grad()
    def test_step(self, batch):
        image,gt_mask = batch
        image,gt_mask = image.float(), gt_mask.long()
        pred = self.model(image)
        self.test_metrics.update(pred, gt_mask) #Oh so that is why they have implemented a custom Segmentation loss, so that 
        #at every test step, they can update the confusion matrix.

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute() #And after accumulating the test metrics at every step, they compute the final metrics here.
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
    
    def configure_optimizers(self):
        #So Here the weight decay is not applied to LayerNorm and Biases.
        if self.cfg.optimizer.exclude_bias_from_wd:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.model.parameters())
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
