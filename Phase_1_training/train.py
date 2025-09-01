import os
import yaml
import tqdm
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchmetrics.segmentation import DiceScore
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from EliteNet import UNet              # Replace with actual model import
from dataset import CustomDataset      # Replace with actual dataset import

from utils.utils import get_reflect_pad_transform

# === Metrics ===
def dice_coefficient(pred, target, eps=1e-6):
    target = target.unsqueeze(1)
    pred = torch.sigmoid(pred)  # for logits output
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def jaccard_index(pred, target, eps=1e-6):
    target = target.unsqueeze(1)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def convert_to_one_hot(labels, num_classes=6):
    """
    Convert (B, 1, H, W) indices to (B, 6, H, W) one-hot
    """
    labels = labels.squeeze(1)  # (B, H, W)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes)  # (B, H, W, 6)
    return one_hot.permute(0, 3, 1, 2).float()  # (B, 6, H, W)


def predictions_to_classes(predictions):
    """
    Convert model output to predicted class indices
    predictions: (B, num_classes, H, W)
    returns: (B, H, W) with predicted class indices
    """
    return torch.argmax(predictions, dim=1)

# === Load config.yaml ===
parser = argparse.ArgumentParser(description="Train UNet with config")
parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file")
args = parser.parse_args()

# === Load config.yaml ===
cfg_path = args.config
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)


# cfg_path = '/mnt/data/omkumar/foundation_phase1/Phase_1_training/configs/idrid.yaml'
# with open(cfg_path, "r") as f:
#     cfg = yaml.safe_load(f)

# === Set random seed and device ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("Number of GPUs visible:",torch.cuda.device_count())
# import sys
# sys.exit(0)
torch.manual_seed(cfg["training"]["seed"])
device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

# === Define transforms ===
final_sz = cfg["dataset"]["image_size"]
train_im_path = os.path.join(cfg["dataset"]["path"],"train","images")

pad_transform = get_reflect_pad_transform(train_im_path,cfg_path)

transform = transforms.Compose([
    pad_transform,        
    transforms.PILToTensor() 
])

# === Create train dataset & loader ===
train_dataset = CustomDataset(
    name=cfg["dataset"]["name"],
    data_dir=cfg["dataset"]["path"],
    img_sz=tuple(cfg["dataset"]["image_size"]),
    is_train=True,
    transform=transform,
    cache_dir=cfg["dataset"]["cache_dir"]
)
train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

# === Optional validation dataset & loader ===
val_loader = None
if cfg.get("validation", {}).get("enabled", False):
    val_dataset = CustomDataset(
        name=cfg["dataset"]["name"],
        data_dir=cfg["validation"]["path"],
        img_sz=tuple(cfg["dataset"]["image_size"]),
        is_train=False,
        transform=transform,
        cache_dir=cfg["validation"]["cache_dir"]
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["validation"]["batch_size"], shuffle=False)

# === Model ===
model = UNet(
    in_c=cfg["model"]["in_channels"],
    n_classes=cfg["dataset"]["num_classes"],
    layers=[4, 8, 16]
).to(device)

# === Loss Function and Metrics ===
loss_fn = getattr(nn, cfg["loss"]["name"])()
dice_score = DiceScore(num_classes = 6,average = "micro")
jack_index = JaccardIndex(task = 'multiclass',num_classes=6,average = "micro")
# === Optimizer ===
optimizer_name = cfg["optimizer"]["name"]
optimizer_class = getattr(optim, optimizer_name)
optimizer_args = {
    "lr": cfg["optimizer"]["lr"],
    "weight_decay": cfg["optimizer"].get("weight_decay", 0)
}
if optimizer_name == "SGD":
    optimizer_args["momentum"] = cfg["optimizer"]["momentum"]

optimizer = optimizer_class(model.parameters(), **optimizer_args)

# === Learning Rate Scheduler ===
scheduler = None
if cfg["lr_scheduler"]["use_scheduler"]:
    scheduler_name = cfg["lr_scheduler"]["name"]
    if scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg["lr_scheduler"]["step_size"], gamma=cfg["lr_scheduler"]["gamma"]
        )
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=cfg["lr_scheduler"]["mode"], factor=cfg["lr_scheduler"]["factor"],
            patience=cfg["lr_scheduler"]["patience"]
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# === Tracking for plots ===
train_losses, val_losses = [], []
train_dice_scores, val_dice_scores = [], []
train_jaccard_scores, val_jaccard_scores = [], []

# === Training Loop ===
epochs = cfg["training"]["epochs"]
log_interval = cfg["logging"]["log_interval"]


best_jaccard = 0.0
best_dice = 0.0

for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
    model.train()
    total_loss, total_dice, total_jaccard = 0, 0, 0

    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        # print("Fetched a batch:",images.size(),masks.size(),images.device,masks.device)
        outputs = model(images)
        # print("Masks Shape before the Squeeze Operation:",masks.shape)
       # Add this right before your loss computation
        masks = masks.squeeze(1)

        # Check if masks have any unexpected dimensions
        if masks.dim() != 3:  # Should be [batch_size, H, W]
            print(f"WARNING: Masks have {masks.dim()} dimensions, expected 3")
        if outputs.dim() != 4:  # Should be [batch_size, num_classes, H, W]
            print(f"WARNING: Outputs have {outputs.dim()} dimensions, expected 4")

        try:
            loss = loss_fn(outputs, masks)
            # print(f"Loss computed successfully: {loss.item()}")
        except Exception as e:
            print(f"Loss computation failed: {e}")
            print(f"Final outputs shape: {outputs.shape}")
            print(f"Final masks shape: {masks.shape}")
            raise e

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        pred_processed = predictions_to_classes(outputs)
        # print("Device Training:",pred_processed.cpu(),masks.cpu())
        # print("SHapes:",pred_processed.shape,masks.shape)
        # print("Pred SHape:",pred_processed.shape,"Mask Shape:",masks.shape)
        # print("Unique Values:",torch.unique(pred_processed),"Masks:",torch.unique(masks))

        total_dice += dice_score(pred_processed.cpu(), masks.cpu())
        total_jaccard += jack_index(pred_processed.cpu(), masks.cpu())

        if (i + 1) % log_interval == 0:
            print(f"[Train] Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    avg_train_dice = total_dice / len(train_loader)
    avg_train_jaccard = total_jaccard / len(train_loader)

    print("Average Training Dice Score:",avg_train_dice,"Average Training Jaccard:",avg_train_jaccard)
    train_losses.append(avg_train_loss)
    train_dice_scores.append(avg_train_dice)
    train_jaccard_scores.append(avg_train_jaccard)

    # === Validation Loop ===
    if val_loader is not None:
        model.eval()
        val_loss, val_dice, val_jaccard = 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                masks = masks.squeeze(1)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                pred_processed = predictions_to_classes(outputs).to(device)
                # print("Device validation:",pred_processed.device,masks.device)
                val_dice += dice_score(pred_processed.cpu(), masks.cpu())
                val_jaccard += jack_index(pred_processed.cpu(), masks.cpu())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_jaccard = val_jaccard / len(val_loader)
        print("Average Validation Dice:",avg_val_dice,"Averate Validation Jaccard:",avg_val_jaccard)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_val_dice)
        val_jaccard_scores.append(avg_val_jaccard)

        print(f"[Val] Epoch [{epoch+1}/{epochs}] Loss: {avg_val_loss:.4f}, "
              f"Dice: {avg_val_dice:.4f}, Jaccard: {avg_val_jaccard:.4f}")


    ref_dice = avg_val_dice if val_loader is not None else avg_train_dice
    ref_jaccard = avg_val_jaccard if val_loader is not None else avg_train_jaccard

    save_base_dir = os.path.join(cfg["logging"]["save_dir"], "ckpts", cfg["logging"]["exp_name"],"Best")
    os.makedirs(save_base_dir, exist_ok=True)

    # Save best Jaccard model
    if ref_jaccard > best_jaccard:
        best_jaccard = ref_jaccard
        torch.save(model.state_dict(), os.path.join(save_base_dir, "best_jaccard.pt"))
        print(f"✅ New best Jaccard: {best_jaccard:.4f} — saved best_jaccard.pt")

    # Save best Dice model
    if ref_dice > best_dice:
        best_dice = ref_dice
        torch.save(model.state_dict(), os.path.join(save_base_dir, "best_dice.pt"))
        print(f"✅ New best Dice: {best_dice:.4f} — saved best_dice.pt")


    # === Step the scheduler ===
    if scheduler:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_train_loss)
        else:
            scheduler.step()

    # === Save model ===
    if (epoch + 1) % cfg["logging"]["save_interval"] == 0:
        os.makedirs(cfg["logging"]["save_dir"], exist_ok=True)
        save_path = os.path.join(cfg["logging"]["save_dir"],"ckpts",cfg["logging"]["exp_name"],"ALL",f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    # === Plot every 100 epochs ===
    if (epoch + 1) % 100 == 0:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 3, 2)
        plt.plot(train_dice_scores, label='Train Dice')
        if val_dice_scores:
            plt.plot(val_dice_scores, label='Val Dice')
        plt.legend()
        plt.title("Dice Coefficient")

        plt.subplot(1, 3, 3)
        plt.plot(train_jaccard_scores, label='Train Jaccard')
        if val_jaccard_scores:
            plt.plot(val_jaccard_scores, label='Val Jaccard')
        plt.legend()
        plt.title("Jaccard Index")

        plt.tight_layout()
        plot_save_dir = os.path.join(cfg["logging"]["save_dir"],"plots",cfg["logging"]["exp_name"])
        os.makedirs(plot_save_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_save_dir, f"metrics_epoch_{epoch+1}.png"))
        plt.close()


