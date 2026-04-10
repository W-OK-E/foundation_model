import os
import hydra
import wandb
import json
import torch
import traceback
import numpy as np
from datetime import datetime
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate
from models.module import ElitLightModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

# Resolver for arithmetic in yaml
OmegaConf.register_new_resolver("eval", eval)

def setup_callbacks(cfg):
    """Local checkpointing and progress bar"""
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(cfg.root_dir, "checkpoints", cfg.experiment_name),
        filename="best_dice_ckpt",
        monitor="val/loss",
        mode="min",
        save_last=True
    )
    return [checkpoint, TQDMProgressBar(refresh_rate=10), LearningRateMonitor()]

@torch.no_grad()
def run_post_training_analysis(cfg, model, datamodule):
    """Simple Visualization and Report after training"""
    print("🚀 Starting Post-Training Analysis...")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    res_dir = os.path.join(cfg.root_dir, "checkpoints", cfg.experiment_name, "results")
    viz_dir = os.path.join(res_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Check if multi-head
    datasets = getattr(model, 'dataset_classes', {cfg.dataset.name: cfg.dataset.num_classes})
    val_loaders = datamodule.val_dataloader()
    if not isinstance(val_loaders, (dict, list)): val_loaders = {cfg.dataset.name: val_loaders}

    report = {}

    for ds_name, loader in val_loaders.items():
        print(f"📊 Processing {ds_name}...")
        metrics_obj = instantiate(cfg.model.val_metrics, num_classes=datasets[ds_name]).to(device)
        
        # Take a few samples for visualization
        samples_saved = 0
        for i, batch in enumerate(loader):
            img, mask, _ = batch
            img, mask = img.to(device), mask.to(device)
            
            # Forward pass
            if hasattr(model.model, 'dataset_classes'):
                pred = model.model(img, dataset_name=ds_name)
            else:
                pred = model.model(img)
            
            metrics_obj.update(pred, mask)
            
            # Save top 5 visualizations
            if samples_saved < 5:
                pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_mask * (255 // datasets[ds_name])).save(os.path.join(viz_dir, f"{ds_name}_{i}.png"))
                samples_saved += 1
        
        mean_res, class_res = metrics_obj.compute()
        report[ds_name] = {"mean": mean_res, "per_class": class_res}
        print(f"✅ {ds_name} IoU: {mean_res.get('miou', 'N/A')}")

    # Save final report
    with open(os.path.join(res_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Update status
    status_path = os.path.join(cfg.root_dir, "checkpoints", cfg.experiment_name, "RUN_STATUS.txt")
    with open(status_path, "w") as f:
        f.write(f"status: SUCCESS\ntimestamp: {datetime.utcnow().isoformat()}Z\n")
    print(f"✨ Analysis complete. Results saved to {res_dir}")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    # Setup WandB
    logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.experiment_name,
        offline=cfg.logger.offline
    )

    # Initialize Model and Data
    model = ElitLightModel(cfg)
    datamodule = instantiate(cfg.datamodule)
    
    # Trainer
    callbacks = setup_callbacks(cfg)
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    try:
        if cfg.mode == "train":
            trainer.fit(model, datamodule=datamodule)
            run_post_training_analysis(cfg, model, datamodule)
        elif cfg.mode == "eval":
            run_post_training_analysis(cfg, model, datamodule)
    except Exception as e:
        print(f"❌ Operation Failed: {e}")
        traceback.print_exc()
        path = os.path.join(cfg.root_dir, "checkpoints", cfg.experiment_name, "RUN_STATUS.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"status: FAILED\ndetails: {str(e)}\n")
        raise e

if __name__ == "__main__":
    main()
