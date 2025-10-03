import os
import sys
import subprocess
import hydra
import wandb
import json
import csv
import torch

from shutil import copyfile
from omegaconf import OmegaConf
from os.path import isfile, join
from hydra.utils import instantiate
from models.module import ElitLightModel
from lightning_fabric.utilities.rank_zero import _get_rank
from pytorch_lightning.callbacks import LearningRateMonitor


# Registering the "eval" resolver allows for advanced config, i.e. basically the values can be dynamic now
# interpolation with arithmetic operations in hydra:
OmegaConf.register_new_resolver("eval", eval)

#To track the wandb experiments
def wandb_init(cfg):
    directory = cfg.checkpoints.dirpath
    if isfile(join(directory, "wandb_id.txt")):
        with open(join(directory, "wandb_id.txt"), "r") as f:
            wandb_id = f.readline()
    else:
        
        rank = _get_rank()
        wandb_id = wandb.util.generate_id()
        print(f"Generated wandb id: {wandb_id}")
        if rank == 0 or rank is None:
            with open(join(directory, "wandb_id.txt"), "w") as f:
                f.write(str(wandb_id))

    return wandb_id

def load_model(cfg, dict_config, wandb_id, callbacks):
    directory = cfg.checkpoints.dirpath
    #This will come in play when we want to test/evaluate the model
    if(cfg.mode == "eval" and isfile(join(directory, "best_dice_dice.ckpt"))):
        checkpoint_path = join(directory, "best_dice_dice.ckpt")
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        model = ElitLightModel.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
        print(f"Loading form checkpoint ... {checkpoint_path}")

    #This makes sure training is resumed from the last checkpoint if available
    elif isfile(join(directory, "last.ckpt")):
        checkpoint_path = join(directory, "last.ckpt")
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        model = ElitLightModel.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
        print(f"Loading form checkpoint ... {checkpoint_path}")
    else:
        checkpoint_path = None
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        log_dict = {"model": dict_config["model"], "dataset": dict_config["dataset"]}
        logger._wandb_init.update({"config": log_dict})
        model = ElitLightModel(cfg.model)

    trainer, strategy = cfg.trainer, cfg.trainer.strategy
    trainer = instantiate(
        trainer, strategy=strategy, logger=logger, callbacks=callbacks,
    )
    return trainer, model, checkpoint_path

def project_init(cfg):
    print("Working directory set to {}".format(os.getcwd()))
    directory = cfg.checkpoints.dirpath
    os.makedirs(directory, exist_ok=True)
    copyfile(".hydra/config.yaml", join(directory, "config.yaml"))


def callback_init(cfg):
    monitor = cfg.checkpoints["monitor"]
    filename = cfg.checkpoints["filename"]
    cfg.checkpoints["monitor"] = monitor 
    cfg.checkpoints["filename"] = filename 
    checkpoint_callback = instantiate(cfg.checkpoints)
    progress_bar = instantiate(cfg.progress_bar)
    lr_monitor = LearningRateMonitor()
    callbacks = [checkpoint_callback, progress_bar, lr_monitor]
    return callbacks

def init_datamodule(cfg):
    datamodule = instantiate(cfg.datamodule)
    return datamodule

def hydra_boilerplate(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    callbacks = callback_init(cfg)
    datamodule = init_datamodule(cfg)
    project_init(cfg)
    wandb_id = wandb_init(cfg)
    trainer, model, ckpt_path = load_model(cfg, dict_config, wandb_id, callbacks)
    return trainer, model, datamodule, ckpt_path


def run_post_training_visualization(cfg):
    """Run visualization script on files listed in datasets/<dataset>/viz.txt.

    This executes visualize.py from the project root so all relative paths
    inside that script remain valid even when Hydra has changed CWD.
    """
    try:
        project_root = cfg.root_dir
        dataset_name = cfg.dataset.name
        cmd = [
            sys.executable,
            os.path.join(project_root, "visualize.py"),
            "--dataset_name",
            dataset_name,
        ]
        # Run from project root to honor relative paths in visualize.py
        subprocess.run(cmd, cwd=project_root, check=True)
        print(f"Post-training visualizations generated for {dataset_name}.")
    except Exception as exc:
        print(f"Visualization step failed: {exc}")


def _select_dataset_by_split(datamodule, split):
    if split == "train":
        return datamodule.train_dataloader().dataset
    if split == "val":
        return datamodule.val_dataloader().dataset
    if split == "test":
        return datamodule.test_dataloader().dataset
    raise ValueError(f"Unsupported split for report: {split}")


def _compute_segmentation_report(model, datamodule, report_cfg):
    model.eval()
    split = report_cfg.split
    dataset = _select_dataset_by_split(datamodule, split)
    loader = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }[split]()

    # Reuse the same metric class configured for test to ensure consistency
    metrics_obj = instantiate(model.cfg.test_metrics)

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, gt = batch[0], batch[1]
            else:
                continue
            images = images.float().to(model.device)
            gt = gt.long().to(model.device)
            logits = model.model(images)
            metrics_obj.update(logits, gt)

    results = metrics_obj.compute()

    # Filter based on requested metrics
    requested = report_cfg.metrics
    if "all" not in requested:
        filtered = {}
        for key, value in results.items():
            if key in requested:
                filtered[key] = value
            elif key.startswith("class_"):
                if "per_class_iou" in requested and key.endswith("_iou"):
                    filtered[key] = value
                if "per_class_dice" in requested and key.endswith("_dice"):
                    filtered[key] = value
        results = filtered

    return results


def run_post_training_report(cfg, model, datamodule):
    report_cfg = cfg.report
    if not report_cfg.enabled:
        return

    results = _compute_segmentation_report(model, datamodule, report_cfg)

    # Determine output directory based on the active hydra run dir and experiment name
    run_dir = os.getcwd()
    out_dir = os.path.join(run_dir, report_cfg.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # File stems based on experiment name and split
    stem = f"{cfg.experiment_name}_{report_cfg.split}"
    json_path = os.path.join(out_dir, f"{stem}.json")
    csv_path = os.path.join(out_dir, f"{stem}.csv")

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV (key,value)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, v])
    print(f"Saved metrics report to {out_dir}")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    trainer, model, datamodule, ckpt_path = hydra_boilerplate(cfg)
    model.datamodule = datamodule
    if cfg.mode == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        # After successful training, generate visualizations on vis.txt images
        run_post_training_visualization(cfg)
        # Generate post-training metrics report
        run_post_training_report(cfg, model, datamodule)
    elif cfg.mode == "eval":
        trainer.test(model, datamodule=datamodule)
    elif cfg.mode == "predict":
        trainer.predict(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
