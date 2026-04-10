import pytorch_lightning as L
import torch
import ipdb
import torch.nn as nn
from PIL import Image

from utils.visualizer import visualize
from .network.ElitNet import ElitNet
from hydra.utils import instantiate

class ElitLightModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model # Model specific config group
        
        self.model = instantiate(self.model_cfg.network.instance)
        
        # Determine if we are in multi-head mode
        self.dataset_classes = getattr(self.model, 'dataset_classes', None)
        self.is_multihead = self.dataset_classes is not None

        if self.is_multihead:
            # Create ModuleDicts for losses and metrics for each dataset
            self.loss = nn.ModuleDict()
            self.train_metrics = nn.ModuleDict()
            self.val_metrics = nn.ModuleDict()
            self.test_metrics = nn.ModuleDict()

            for name, n_cls in self.dataset_classes.items():
                cls_weights = None
                try:
                    # In us_idrid.yaml, weights are under train_dataset[name].cls_weights
                    cls_weights = cfg.dataset.train_dataset[name].get('cls_weights', None)
                except:
                    pass
                
                # Instantiate components per head
                self.loss[name] = instantiate(self.model_cfg.loss.instance, alpha=cls_weights, num_classes=n_cls)
                self.train_metrics[name] = instantiate(self.model_cfg.train_metrics, num_classes=n_cls)
                self.val_metrics[name] = instantiate(self.model_cfg.val_metrics, num_classes=n_cls)
                self.test_metrics[name] = instantiate(self.model_cfg.test_metrics, num_classes=n_cls)
        else:
            # Single-dataset mode instantiation
            num_classes = getattr(cfg.dataset, 'num_classes', 0)
            cls_weights = getattr(cfg.dataset, 'cls_weights', None)
            
            self.loss = instantiate(self.model_cfg.loss.instance, alpha=cls_weights, num_classes=num_classes)
            self.train_metrics = instantiate(self.model_cfg.train_metrics, num_classes=num_classes)
            self.val_metrics = instantiate(self.model_cfg.val_metrics, num_classes=num_classes)
            self.test_metrics = instantiate(self.model_cfg.test_metrics, num_classes=num_classes)

        self.val_steps = 0
        self.viz_image_count = 0

    def training_step(self, batch):
        if isinstance(batch, dict):
            # MULTI-LOADER MODE (from CombinedLoader)
            total_loss = 0
            for dataset_name, d_batch in batch.items():
                image, gt_mask, _ = d_batch
                image, gt_mask = image.float(), gt_mask.long()

                pred = self.model(image, dataset_name=dataset_name)
                # Use dataset-specific loss
                loss = self.loss[dataset_name](pred, gt_mask)
                total_loss += loss
                
                # Log stats with dataset prefix
                self.train_metrics[dataset_name].update(pred.detach(), gt_mask)
                self.log(f"train/{dataset_name}/loss", loss, sync_dist=True)
            
            return total_loss
        
        # SINGLE LOADER MODE
        if len(batch) == 3:
            image, gt_mask, dataset_name = batch
        else:
            image, gt_mask = batch
            dataset_name = None

        image, gt_mask = image.float(), gt_mask.long()

        if self.is_multihead:
            pred = self.model(image, dataset_name=dataset_name)
            metrics = self.train_metrics[dataset_name]
            loss_func = self.loss[dataset_name]
        else:
            pred = self.model(image)
            metrics = self.train_metrics
            loss_func = self.loss
        
        loss = loss_func(pred, gt_mask)
        metrics.update(pred.detach(), gt_mask)
        
        prefix = f"train/{dataset_name}/" if dataset_name else "train/"
        self.log(f"{prefix}loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """Compute and reset training metrics at the end of the epoch."""
        if self.is_multihead:
            for name, metrics in self.train_metrics.items():
                mean_metrics, _ = metrics.compute()
                for m_name, m_val in mean_metrics.items():
                    self.log(f"train/{name}/{m_name}", m_val, sync_dist=True)
                metrics.reset()
        else:
            mean_metrics, _ = self.train_metrics.compute()
            for m_name, m_val in mean_metrics.items():
                self.log(f"train/{m_name}", m_val, sync_dist=True)
            self.train_metrics.reset()

    @torch.no_grad()
    def validation_step(self, batch: list):
        if isinstance(batch, dict):
            # MULTI-LOADER MODE
            total_loss = 0
            for dataset_name, d_batch in batch.items():
                image, gt_mask, _ = d_batch
                image, gt_mask = image.float(), gt_mask.float()
   
                pred = self.model(image, dataset_name=dataset_name)
                # Use dataset-specific loss
                loss = self.loss[dataset_name](pred, gt_mask)
                total_loss += loss
                
                metrics = self.val_metrics[dataset_name]
                metrics.update(pred.detach(), gt_mask.detach())
                
                prefix = f"val/{dataset_name}/"
                self.log(f"{prefix}loss", loss, sync_dist=True, on_step=True, on_epoch=True)
                
                # Compute and log metrics
                mean_metrics, _ = metrics.compute()
                for metric_name, m_val in mean_metrics.items():
                    self.log(f"{prefix}{metric_name}", m_val, sync_dist=True, on_step=True)
                metrics.reset()
            
            self.log("val/loss", total_loss / len(batch), sync_dist=True, on_epoch=True)
            return

        # SINGLE LOADER MODE
        if len(batch) == 3:
            image, gt_mask, dataset_name = batch
        else:
            image, gt_mask = batch
            dataset_name = None

        image, gt_mask = image.float(), gt_mask.float()
        
        if self.is_multihead:
            pred = self.model(image, dataset_name=dataset_name)
            metrics = self.val_metrics[dataset_name]
            loss_func = self.loss[dataset_name]
        else:
            pred = self.model(image)
            metrics = self.val_metrics
            loss_func = self.loss
        
        loss = loss_func(pred, gt_mask)
        metrics.update(pred.detach(), gt_mask.detach())
        
        mean_metrics, class_metrics = metrics.compute()
        
        prefix = f"val/{dataset_name}/" if dataset_name else "val/"
        for metric_name, metric_value in mean_metrics.items():
            self.log(f"{prefix}{metric_name}", metric_value, sync_dist=True, on_step=True)
        
        metrics.reset()
        self.log(f"{prefix}loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        self.val_steps += 1

    def on_validation_epoch_end(self):
        pass
        #TODO: Remove this if the reset at every step works.
        # mean_metrics, class_metrics = self.val_metrics.compute()
        
        # precisions = []
        # for metric_name,metric_value  in class_metrics.items():
        #     if("precision" in metric_name):
        #         precisions.append(metric_value)
        
        # mean_metrics["m_prec"] = sum(precisions)/len(precisions)

        # for metric_name, metric_value in mean_metrics.items():
        #     self.log(
        #         f"val/{metric_name}",
        #         metric_value,
        #         sync_dist=True,
        #         on_step=False
        #     )
        # self.val_metrics.reset()

    @torch.no_grad()
    def test_step(self, batch):
        image,gt_mask = batch
        image,gt_mask = image.float(), gt_mask.long()
        pred = self.model(image)
        self.test_metrics.update(pred.detach(), gt_mask.detach()) #Oh so that is why they have implemented a custom Segmentation loss, so that 
        #at every test step, they can update the confusion matrix.
        mean_metrics, class_metric = self.test_metrics.compute() #And after accumulating the test metrics at every step, they compute the final metrics here.
        for metric_name, metric_value in mean_metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
        self.test_metrics.reset()
    
    @torch.no_grad()
    def on_test_epoch_end(self):
        pass
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        image, image_id = batch
        image = image.float()
        pred = self.model(image)
        pred = torch.argmax(pred, dim=1).cpu().numpy().astype("uint8")
        #Converting each mask to PIL Image and then saving it.
        for i in range(pred.shape[0]):
            mask = Image.fromarray(pred[i])
            mask.save(f"pred_{image_id[i]}.png")
        return pred

    def configure_optimizers(self):
        #So Here the weight decay is not applied to LayerNorm and Biases.
        if self.model_cfg.optimizer.exclude_bias_from_wd:
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
                    "weight_decay": self.model_cfg.optimizer.optim.weight_decay,
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
                self.model_cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.model_cfg.optimizer.optim, self.model.parameters())
        scheduler = instantiate(self.model_cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "monitor":"val/loss", 
                            "frequency": self.trainer.check_val_every_n_epoch}]

    def lr_scheduler_step(self, scheduler, metric):
        if(metric is None):
            print("No Metric, skipping step")
            return
        scheduler.step(metric)


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
