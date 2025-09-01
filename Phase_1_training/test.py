import os
import yaml
import torch
import argparse
import datetime
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from EliteNet import UNet
from dataset import CustomDataset

# === Metrics ===
def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    return ((2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)).mean().item()

def jaccard_index(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()

def predictions_to_classes(predictions):
    """
    Convert model output to predicted class indices
    predictions: (B, num_classes, H, W)
    returns: (B, H, W) with predicted class indices
    """
    return torch.argmax(predictions, dim=1)

# === Parsing CLI Arguments ===
parser = argparse.ArgumentParser(description="Train UNet with config")
parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file")
parser.add_argument("--ckpt_epoch",type=int, help="Epoch for the saved CKPT")
parser.add_argument("--best_jaccard", action='store_true', help="Use checkpoint with best Jaccard index")
parser.add_argument("--best_dice", action='store_true', help="Use checkpoint with best Dice coefficient")

args = parser.parse_args()
#Ensuring that atleast one option has been provided
ckpt_options = [
    args.ckpt_epoch is not None,
    args.best_jaccard,
    args.best_dice
]

if ckpt_options.count(True) != 1:
    raise ValueError("You must provide exactly one of the following: --ckpt_epoch, --best_jaccard, or --best_dice.")

# === Load config.yaml ===
cfg_path = args.config
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("Number of GPUs visible:",torch.cuda.device_count())
# import sys
# sys.exit(0)
torch.manual_seed(cfg["training"]["seed"])
device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

# === Data transform ===
transform = transforms.Compose([
    transforms.Resize(cfg["dataset"]["image_size"]),
    transforms.ToTensor()
])

# === Validation dataset ===
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

# === Loading State Dict ===
ckpt_file = None
if(args.ckpt_epoch is not None):
    ckpt_file = f'/mnt/data/omkumar/foundation_phase1/Phase_1_training/logs/ckpts/US_Nerve_Seg/ALL/model_epoch_{args.ckpt_epoch}.pt'
    save_suff = args.ckpt_epoch
elif(args.best_dice):
    ckpt_file = f'/mnt/data/omkumar/foundation_phase1/Phase_1_training/logs/ckpts/US_Nerve_Seg/Best/best_dice.pt'
    save_suff = "best_dice"
else:
    ckpt_file = f'/mnt/data/omkumar/foundation_phase1/Phase_1_training/logs/ckpts/US_Nerve_Seg/Best/best_jaccard.pt'
    save_suff = "best_jacc"

model.load_state_dict(torch.load(ckpt_file,map_location=device))
model.eval()

# === Evaluation ===
dice_scores = []
jaccard_scores = []

timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
res_save_dir = cfg["logging"]["results"] + f"/session_{timestamp}_epoch_{save_suff}"

os.makedirs(res_save_dir,exist_ok=True)

with torch.no_grad():
    for idx,(images, masks) in tqdm(enumerate(val_loader)):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        outputs_corrected = predictions_to_classes(outputs)
        # print("Masks Shape:",masks.shape)
        # print("Outputs Shape:",outputs.shape)
        # print("Masks Unique values:",torch.unique(masks))
        # print("Outputs Unique Values:",torch.unique(outputs_corrected))

        from utils.utils import calculate_dice_on_each_class
        class_dice_scores,class_jack_scores = calculate_dice_on_each_class(masks,outputs)

        metric_out_file = os.path.join(res_save_dir,f'Metric{idx}.txt')
        with open(metric_out_file,'w') as file:
            for cls in class_dice_scores:
                file.write(f'{cls}\n')
            file.write("\nJaccard Indices")
            for cls in class_jack_scores:
                file.write(f'{cls}\n')

        print("The outputs:",outputs.shape)
        outputs_processed = torch.tensor(predictions_to_classes(outputs)*100,dtype = torch.uint8)
        masks_processed = torch.tensor(masks*100,dtype = torch.uint8)

        pred_save_path  = os.path.join(res_save_dir,f'Pred_Mask_{idx}.png')
        mask_save_path = os.path.join(res_save_dir,f"GT_Mask{idx}.png")
    
        assert outputs_processed.shape == masks_processed.shape

        print(torch.unique(masks_processed[0]))
        if(outputs_processed.ndim > 3):
            Image.fromarray(outputs_processed.squeeze(0).cpu().numpy()).save(pred_save_path)
            Image.fromarray(masks_processed.squeeze(0).cpu().numpy()).save(mask_save_path)
        else:
            for batch in tqdm(range(0,outputs_processed.shape[0],4)):
                pred_save_path  = os.path.join(res_save_dir,f'Pred_Mask_{batch}.png')
                mask_save_path = os.path.join(res_save_dir,f"GT_Mask{batch}.png")
                Image.fromarray(outputs_processed[batch].cpu().numpy()).save(pred_save_path)
                Image.fromarray(masks_processed[batch].cpu().numpy()).save(mask_save_path)
        print("Saved the Images")
        import sys
        sys.exit(0)
        # vutils.save_image(outputs_processed,pred_save_path)
        # vutils.save_image(masks_processed,mask_save_path)        
        # dice_scores.append(dice_score(outputs, masks))
        # jaccard_scores.append(jaccard_index(outputs, masks))

print(f"Dice Score: {sum(dice_scores)/len(dice_scores):.4f}")
print(f"Jaccard Index: {sum(jaccard_scores)/len(jaccard_scores):.4f}")
