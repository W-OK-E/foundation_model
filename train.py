import os
import hydra
import wandb
import json
import csv
import torch
import warnings
import traceback
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio

from data.transforms import get_transforms
from shutil import copyfile
from utils.visualizer import visualize
from omegaconf import OmegaConf
from os.path import isfile, join
from hydra.utils import instantiate
from models.module import ElitLightModel
from lightning_fabric.utilities.rank_zero import _get_rank
from pytorch_lightning.callbacks import LearningRateMonitor
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings("ignore")

#Sample colors to pick from while visualizations
COLORS = [
    "black",        # 0
    "red",          # 1
    "green",        # 2
    "blue",         # 3
    "yellow",       # 4
    "magenta",      # 5
    "cyan",         # 6
    "orange",       # 7
    "purple",       # 8
    "brown",        # 9
    "pink",         # 10
    "lime",         # 11
    "teal",         # 12
    "navy",         # 13
    "maroon",       # 14
    "olive",        # 15
    "coral",        # 16
    "gold",         # 17
    "turquoise",    # 18
    "violet"        # 19
]


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
        # checkpoint_path = join(directory, "best_dice_ckpt.ckpt")
        # logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        # model = ElitLightModel(cfg.model)
        # ckpt = torch.load(checkpoint_path)
        # model = ElitLightModel.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
        #The best_dice_ckpt.ckpt will be loaded int he _viz_from_split() function
        checkpoint_path = None
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        log_dict = {"model": dict_config["model"], "dataset": dict_config["dataset"]}
        logger._wandb_init.update({"config": log_dict})
        model = ElitLightModel(cfg.model)
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
        print("Instantiating the Model")

    trainer, strategy = cfg.trainer, cfg.trainer.strategy
    trainer = instantiate(
        trainer, strategy=strategy, logger=logger, callbacks=callbacks,
    )
    return trainer, model, checkpoint_path



def denormalize_batch_torch(imgs_norm, mean, std, max_pixel_value=255.0):
    """
    Reverse Albumentations normalization for a batch of PyTorch tensors.
    
    Args:
        imgs_norm: torch.Tensor of shape (B, C, H, W)
        mean, std: list or tensor of per-channel values (len = C)
        max_pixel_value: float, same as used during normalization (default 255.0)
    
    Returns:
        imgs_denorm: torch.ByteTensor of shape (B, H, W, C)
    """
    mean = torch.tensor(mean, device=imgs_norm.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=imgs_norm.device).view(1, -1, 1, 1)
    
    imgs_denorm = (imgs_norm * std + mean) #* max_pixel_value
    imgs_denorm = imgs_denorm.clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy()  # B x H x W x C
    return imgs_denorm


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
    if(cfg.mode != "test"):
        project_init(cfg)
    wandb_id = wandb_init(cfg)
    trainer, model, ckpt_path = load_model(cfg, dict_config, wandb_id, callbacks)
    print("Loading Checkpoint from",ckpt_path)
    return trainer, model, datamodule, ckpt_path


def run_post_training_visualization(cfg,model):
    """Run visualization script on files listed in datasets/<dataset>/viz.txt.

    This executes visualize.py from the project root so all relative paths
    inside that script remain valid even when Hydra has changed CWD.
    """
    # Prefer built-in lightweight visualizer that reads datasets/<dataset>/split.json
    print("Generating Post-Training Visualizations")
    try:
        project_root = cfg.root_dir
        dataset_name = cfg.dataset.name
        _viz_from_split(project_root, dataset_name, cfg,model)
        print(f"Post-training visualizations generated for {dataset_name}.")
    except Exception as exc:
        print(f"Visualization step failed: {exc}")


def _find_file_recursive(root_dir, basename):
    """Search for a file with name == basename under root_dir (recursively)."""
    for dirpath, _, filenames in os.walk(root_dir):
        if basename in filenames:
            return join(dirpath, basename)
    return None

@torch.no_grad()
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

    viz_list = split.get("train", [])[:20]
    out_dir = join(cfg.checkpoints.dirpath, "viz")
    os.makedirs(out_dir, exist_ok=True)
    
    best_model_path = os.path.join(cfg.checkpoints.dirpath,'best_dice_ckpt.ckpt')
    ckpt = torch.load(best_model_path)
    model.load_state_dict(ckpt['state_dict'])

    # for i, fname in enumerate(viz_list):
    #     # mask file is typically the listed name (e.g. 0000000384_rgb_mask.png)
    #     mask_path = os.path.join(dataset_dir,f'masks/{fname}')
    #     img_path = os.path.join(dataset_dir,f'images/{fname}')

    #     if mask_path is None and img_path is None:
    #         print(f"Skipping visualization for {fname}: files not found in {dataset_dir}")
    #         continue

    #     # Load original image
        
    #     orig = iio.imread(img_path)
    #     # if(cfg.dataset.multi_label):
    #     #     print("Multi-Label Visualization yet to be implemented")
    #     #     continue

    #     mask = iio.imread(mask_path)
    #     # if mask.ndim == 3:
    #     #         mask = np.dot(mask[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    #     tf,tf2 = get_transforms(orig.shape[:2])
    #     transformer = tf(image = orig, mask = mask)
    #     orig, mask = transformer["image"], transformer["mask"]
    #     # print("Transformed Mask:",np.unique(mask.cpu().numpy()))
    #     # import ipdb
    #     # ipdb.set_trace()
    #     # Prediction
    #     pred_arr = None
    #     if model is not None and orig is not None and mask is not None:
    #         try:
    #             orig = orig.unsqueeze(0).to(model.device) #We are adding a batch.dimension
    #             model.eval()
    #             with torch.no_grad():
    #                 out = model.model(orig)
    #             print("Output Shape:",out.shape)
    #             # out can be [B, C, H, W] or [B, 1, H, W]
    #             # if out.dim() == 4 and out.size()[1] > 1:
    #             #     out[:,0,:,:] = 0
    #             #     pred = out.argmax(1).squeeze(0).cpu().numpy()
    #             #     # print("Prediction Reshaped to:",pred.shape,np.unique(pred))
    #             #     # import ipdb
    #             #     # ipdb.set_trace()
    #             # else:
    #             #     out_sig = torch.sigmoid(out)
    #             #     pred = (out_sig.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
    #                 # print("Prediction reshaped to:",pred.shape)
    #             # import ipdb
    #             # ipdb.set_trace()
    #             pred = out.argmax(dim = 1)
    #             pred_arr = pred.permute(1,2,0).cpu().numpy()
    #             unique_values, counts = np.unique(pred_arr, return_counts=True)

    #             # Total number of elements in the array
    #             total_elements = pred_arr.size

    #             # Calculate percentage of each unique value
    #             percentages = (counts / total_elements) * 100

    #             # Combine the unique values, counts, and percentages into a structured format
    #             result = list(zip(unique_values, counts, percentages))

    #             # Print the result
    #             for value, count, percentage in result:
    #                 print(f"Value: {value}, Count: {count}, Percentage: {percentage:.2f}%")

    #         except Exception as exc:
    #             print(f"Prediction failed for {fname}: {exc}")
    #             pred_arr = None

    #     # Create figure
        
    #     # Example class names and colors
    #     # visualize(cfg,orig[0],mask,pred_arr,i) #TODO
    #     class_names = cfg.dataset.class_names
    #     colors = COLORS[:len(class_names)]

    #     # Create a discrete colormap
    #     cmap = ListedColormap(colors)
    #     norm = BoundaryNorm(np.arange(len(class_names) + 1) - 0.5, len(class_names))

    #     # Use GridSpec to allocate space: 3 images + 1 for colorbar
    #     fig = plt.figure(figsize=(16, 4))
    #     gs = fig.add_gridspec(1, 4, width_ratios=[1,1,1,0.1], wspace=0.3)

    #     # --- Original ---
    #     ax0 = fig.add_subplot(gs[0, 0])

    #     #Obrain Arrays
    #     orig = orig[0].permute(1,2,0).cpu().numpy()
    #     mask = mask.cpu().numpy()
        
    #     if orig is not None:
    #         ax0.imshow(np.array(orig))
    #     else:
    #         ax0.text(0.5, 0.5, "Original not found", ha="center")
    #     ax0.set_title("Original")
    #     ax0.axis("off")

    #     # --- Mask ---
    #     ax1 = fig.add_subplot(gs[0, 1])
    #     if mask is not None:
    #         im_mask = ax1.imshow(np.array(mask), cmap=cmap, norm=norm)
    #     else:
    #         ax1.text(0.5, 0.5, "Mask not found", ha="center")
    #     ax1.set_title("Ground Truth Mask")
    #     ax1.axis("off")

    #     # --- Prediction ---
    #     ax2 = fig.add_subplot(gs[0, 2])
    #     # print("Unique Values in pred_arr",np.unique(pred_arr))
    #     # print("Cmap looks like:",cmap)
    #     # import ipdb
    #     # ipdb.set_trace()
    #     if pred_arr is not None:
    #         im_pred = ax2.imshow(pred_arr, cmap=cmap, norm=norm)
    #     else:
    #         ax2.text(0.5, 0.5, "Prediction not available", ha="center")
    #     ax2.set_title("Prediction")
    #     ax2.axis("off")

    #     # --- Colorbar in separate axis ---
    #     ax_cbar = fig.add_subplot(gs[0, 3])
    #     cb = plt.colorbar(im_mask, cax=ax_cbar, ticks=range(len(class_names)))
    #     cb.ax.set_yticklabels(class_names)
    #     cb.set_label("Classes")

    #     out_path = join(out_dir, f"viz_{i:03d}_{os.path.splitext(os.path.basename(fname))[0]}.png")
    #     fig.tight_layout()
    #     fig.savefig(out_path)
    #     plt.close(fig)


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

@torch.no_grad()
def _compute_segmentation_report(model, datamodule, report_cfg,cfg):
    model.eval()
    split = report_cfg
    dataset = _select_dataset_by_split(datamodule, split)
    loader = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }[split]()

    # Reuse the same metric class configured for test to ensure consistency
    metrics_obj = instantiate(model.cfg.test_metrics)
    viz_dir = os.path.join(cfg.checkpoints.dirpath,"viz")
    os.makedirs(viz_dir,exist_ok=True)
    loss = instantiate(model.cfg.loss)['instance']
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, gt = batch[0], batch[1]
            else:
                continue
            images = images.float().to(model.device)
            gt = gt.long().to(model.device)
            logits = model.model(images)

            loss(logits, gt)
            images_vis = denormalize_batch_torch(images,cfg.dataset.mean_per_channel,cfg.dataset.std_per_channel) 
            # print("Logits Predicted by the model:",logits[:4,:4,3],logits.shape)
            # import ipdb
            # ipdb.set_trace()
            def dice_score(pred, target, eps=1e-7):
                """
                Compute the Dice coefficient between two binary arrays.

                Args:
                    pred (np.ndarray): Binary prediction array (0s and 1s).
                    target (np.ndarray): Binary ground truth array (0s and 1s).
                    eps (float): Small value to avoid division by zero.

                Returns:
                    float: Dice coefficient in [0, 1].
                """
                pred = np.asarray(pred).astype(np.bool_)
                target = np.asarray(target).astype(np.bool_)

                intersection = np.logical_and(pred, target).sum()
                dice = (2. * intersection + eps) / (pred.sum() + target.sum() + eps)
                return dice
            
            preds = logits.argmax(dim=1)
            if(split == "val"):
                for idx,(pred,target) in enumerate(zip(preds,gt)):
                    preds_array = pred.cpu().numpy()
                    gt_array = target.cpu().numpy()
                    inp_im = images_vis[idx]
                    # print("Preds array look like:",preds_array.shape)
                    fig = plt.figure(figsize=(16, 4))
                    gs = fig.add_gridspec(1, 4, width_ratios=[1,1,1,0.1], wspace=0.3)
                    ax0 = fig.add_subplot(gs[0, 0])
                    ax0.set_title("Input Image")
                    ax1 = fig.add_subplot(gs[0,1])
                    ax1.set_title("Target")
                    ax2 = fig.add_subplot(gs[0,2])
                    ax2.set_title("Prediction")

                    cmap = ListedColormap(COLORS[:len(cfg.dataset.class_names)])
                    print(inp_im.shape)
                    im_inp = ax0.imshow(inp_im)
                    target = ax1.imshow(gt_array,cmap = cmap)
                    im_pred = ax2.imshow(preds_array,cmap = cmap)
                    ax_cbar = fig.add_subplot(gs[0, 3])
                    cb = plt.colorbar(im_pred, cax=ax_cbar, ticks=range(len(cfg.dataset.class_names)))
                    cb.ax.set_yticklabels(cfg.dataset.class_names)
                    cb.set_label("Classes")
                    plt.savefig(os.path.join(viz_dir,f"Vis_{idx:04d}.png"))
                    plt.close()
            # print("Gt shape:",gt.shape,"Dtype:",gt.dtype)
            print(dice_score(preds.cpu().numpy(),gt.cpu().numpy()))
            # import ipdb; ipdb.set_trace()
            metrics_obj.update(logits, gt)

    mean_results,class_results = metrics_obj.compute()
    print("Mean Metrics:",mean_results,"\n",class_results)
    import ipdb
    # ipdb.set_trace()

    return mean_results,class_results

@torch.no_grad()
def run_post_training_report(cfg, model, datamodule):
    report_cfg = cfg.report
    # if not report_cfg.enabled:
    #     return
    print("Generating Post Training Report")
    for split in ["test","train","val"][2:]:
        # report_cfg.split = split
        mean_results,class_results = _compute_segmentation_report(model, datamodule, split,cfg)
        out_dir = os.path.join(cfg.checkpoints.dirpath,"reports",split)
        os.makedirs(out_dir, exist_ok=True)

        # File stems based on experiment name and split
        stem = f"{cfg.experiment_name}_{split}"
        mean_json_path = os.path.join(out_dir, f"mean_{stem}.json")
        class_json_path = os.path.join(out_dir, f"class_{stem}.json")
        csv_path = os.path.join(out_dir, f"{stem}.csv")

        # Save JSON
        with open(mean_json_path, "w") as f:
            json.dump(mean_results, f, indent=2)
        with open(class_json_path,"w") as f:
            json.dump(class_results,f)


        # Save CSV (key,value)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in mean_results.items():
                writer.writerow([k, v])
            for k,v in class_results.items():
                writer.writerow([k,v])
    print(f"Saved metrics report to {out_dir}")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    trainer, model, datamodule, ckpt_path = hydra_boilerplate(cfg)
    model.datamodule = datamodule
    run_dir = cfg.checkpoints.dirpath
    try:
        if cfg.dry_run:
            pass
        if cfg.mode == "train":
            device = model.device
            trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
            # After successful training, generate visualizations for viz split
            model.to(device)
            run_post_training_visualization(cfg,model)
            # Generate post-training metrics report
            run_post_training_report(cfg, model, datamodule)

        elif cfg.mode == "train_dummy":
            print("Running a dummy session")
            run_post_training_visualization(cfg,model)
            # Generate post-training metrics report
            # run_post_training_report(cfg, model, datamodule)

        elif cfg.mode == "eval":
            print("Running Pilot Evaluation")
            device = model.device
            # trainer.test(model, datamodule=datamodule)
            model = model.to(device) #Just stay on the same device
            run_post_training_visualization(cfg,model)
            run_post_training_report(cfg,model,datamodule)

        elif cfg.mode == "predict":
            trainer.predict(model, datamodule=datamodule)

        # If we reach here, assume run succeeded
        _write_run_status(run_dir, "SUCCESS")
    except Exception as exc:
        # Write failure status and traceback for debugging
        torch.cuda.empty_cache()
        tb = traceback.format_exc()
        _write_run_status(run_dir, "FAILED", details=tb)
        # re-raise so hydra/launcher can see the failure as well
        raise


if __name__ == "__main__":
    main()
