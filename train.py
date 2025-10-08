import os
import sys
import subprocess
import hydra
import wandb
import json
import csv
import torch
import traceback
from datetime import datetime
import hashlib
import random
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    if(cfg.mode == "eval" and isfile(join(directory, "best_dice_ckpt.ckpt"))):
        checkpoint_path = join(directory, "best_dice_ckpt.ckpt")
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
        print("Instantiating ElitNet")

    trainer, strategy = cfg.trainer, cfg.trainer.strategy
    trainer = instantiate(
        trainer, strategy=strategy, logger=logger, callbacks=callbacks,
    )
    return trainer, model, checkpoint_path

def project_init(cfg):
    print("Working directory set to {}".format(os.getcwd()))
    # Create a per-run subdirectory inside the configured checkpoint dir so
    # multiple runs on the same dataset+model don't overwrite each other.
    base_dir = cfg.checkpoints.dirpath
    directory = base_dir
    os.makedirs(directory, exist_ok=True)
    # copy the active hydra config for reproducibility
    try:
        copyfile(".hydra/config.yaml", join(directory, "config.yaml"))
    except Exception:
        # best-effort: don't crash if hydra metadata isn't present
        pass


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
    if(cfg.mode == "train"):
        project_init(cfg)
    wandb_id = wandb_init(cfg)
    trainer, model, ckpt_path = load_model(cfg, dict_config, wandb_id, callbacks)
    print("Loading Checkpoint from",ckpt_path)
    return trainer, model, datamodule, ckpt_path


def run_post_training_visualization(cfg):
    """Run visualization script on files listed in datasets/<dataset>/viz.txt.

    This executes visualize.py from the project root so all relative paths
    inside that script remain valid even when Hydra has changed CWD.
    """
    # Prefer built-in lightweight visualizer that reads datasets/<dataset>/split.json
    print("Generating Post-Training Visualizations")
    try:
        project_root = cfg.root_dir
        dataset_name = cfg.dataset.name
        _viz_from_split(project_root, dataset_name, cfg)
        print(f"Post-training visualizations generated for {dataset_name}.")
    except Exception as exc:
        print(f"Visualization step failed: {exc}")


def _find_file_recursive(root_dir, basename):
    """Search for a file with name == basename under root_dir (recursively)."""
    for dirpath, _, filenames in os.walk(root_dir):
        if basename in filenames:
            return join(dirpath, basename)
    return None


def _viz_from_split(project_root, dataset_name, cfg, model=None):
    """Generate 1x3 visualizations (Original, Mask, Prediction) for files listed
    in datasets/<dataset>/split.json under the "viz" key.

    Saves PNGs under the run checkpoint directory in a subfolder `viz`.
    If `model` is provided, it will be used to create predictions. Otherwise
    this function only copies the Original+Mask images.
    """
    dataset_dir = join(project_root, "datasets", dataset_name)
    split_path = join(dataset_dir, "split.json")
    if not isfile(split_path):
        raise FileNotFoundError(f"split.json not found for dataset {dataset_name} at {split_path}")

    with open(split_path, "r") as f:
        split = json.load(f)

    viz_list = split.get("viz", [])
    out_dir = join(cfg.checkpoints.dirpath, "viz")
    os.makedirs(out_dir, exist_ok=True)

    for i, fname in enumerate(viz_list):
        # mask file is typically the listed name (e.g. 0000000384_rgb_mask.png)
        mask_path = os.path.join(dataset_dir,f'masks/{fname}')
        img_path = os.path.join(dataset_dir,f'images/{fname}')

        if mask_path is None and img_path is None:
            print(f"Skipping visualization for {fname}: files not found in {dataset_dir}")
            continue

        # Load original image
        orig = None
        if img_path is not None:
            try:
                orig = Image.open(img_path).convert("RGB")
            except Exception:
                orig = None

        # Load mask (may be RGB or single-channel)
        mask = None
        if mask_path is not None:
            try:
                mask = Image.open(mask_path).convert("L")
            except Exception:
                mask = None

        # Prediction
        pred_arr = None
        if model is not None and orig is not None:
            try:
                arr = np.array(orig).astype(np.float32) 
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(model.device)
                model.eval()
                with torch.no_grad():
                    out = model.model(tensor)
                # out can be [B, C, H, W] or [B, 1, H, W]
                if out.dim() == 4 and out.size(1) > 1:
                    pred = out.argmax(1).squeeze(0).cpu().numpy()
                else:
                    out_sig = torch.sigmoid(out)
                    pred = (out_sig.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
                pred_arr = pred
            except Exception as exc:
                print(f"Prediction failed for {fname}: {exc}")
                pred_arr = None

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # Original
        if orig is not None:
            axes[0].imshow(np.array(orig))
        else:
            axes[0].text(0.5, 0.5, "Original not found", ha="center")
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Mask
        if mask is not None:
            axes[1].imshow(np.array(mask))
        else:
            axes[1].text(0.5, 0.5, "Mask not found", ha="center")
        axes[1].set_title("Original Mask")
        axes[1].axis("off")

        # Prediction
        if pred_arr is not None:
            axes[2].imshow(pred_arr)
        else:
            axes[2].text(0.5, 0.5, "Prediction not available", ha="center")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        out_path = join(out_dir, f"viz_{i:03d}_{os.path.splitext(os.path.basename(fname))[0]}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def _write_run_status(directory, status, details=None):
    """Write a simple RUN_STATUS.txt into the run directory with timestamp and optional details."""
    try:
        os.makedirs(directory, exist_ok=True)
        path = join(directory, "RUN_STATUS.txt")
        with open(path, "w") as f:
            f.write(f"status: {status}\n")
            f.write(f"timestamp: {datetime.utcnow().isoformat()}Z\n")
            if details:
                f.write("details:\n")
                f.write(str(details))
    except Exception:
        # best-effort, do not crash training only for status file write
        print("Failed to Write status")
        pass


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
    print("Generating Post Training Report")
    for split in ["test","train","val"]:
        report_cfg.split = split
        results = _compute_segmentation_report(model, datamodule, report_cfg)
        out_dir = os.path.join(cfg.checkpoints.dirpath,"reports")
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
    run_dir = cfg.checkpoints.dirpath
    try:
        if cfg.mode == "train":
            trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
            # After successful training, generate visualizations for viz split
            run_post_training_visualization(cfg)
            # Generate post-training metrics report
            run_post_training_report(cfg, model, datamodule)

        elif cfg.mode == "train_dummy":
            print("Running a dummy session")
            run_post_training_visualization(cfg)
            # Generate post-training metrics report
            # run_post_training_report(cfg, model, datamodule)

        elif cfg.mode == "eval":
            trainer.test(model, datamodule=datamodule)
        elif cfg.mode == "predict":
            trainer.predict(model, datamodule=datamodule)

        # If we reach here, assume run succeeded
        _write_run_status(run_dir, "SUCCESS")
    except Exception as exc:
        # Write failure status and traceback for debugging
        tb = traceback.format_exc()
        _write_run_status(run_dir, "FAILED", details=tb)
        # re-raise so hydra/launcher can see the failure as well
        raise


if __name__ == "__main__":
    main()
