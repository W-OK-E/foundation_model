import os
import yaml
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import shutil

from EliteNet import UNet
from dataset import CustomDataset  # Only needed if doing full validation set

# === Load config ===
with open("/mnt/data/omkumar/foundation_phase1/Phase_1_training/configs/idrid.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# === Device ===
device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

# === Load Trained Model ===
model = UNet(
    in_c=cfg["model"]["in_channels"],
    n_classes=cfg["dataset"]["num_classes"],
    layers=[4, 8, 16]
).to(device)

# Load trained weights
checkpoint_path = "/mnt/data/omkumar/foundation_phase1/Phase_1_training/ckpts/IDRID/ALL/model_epoch_500.pt"  # ⬅️ Update this!
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# === Image transform (same as training) ===

fin_size = cfg["dataset"]["image_size"]

if(fin_size is None):
    transform = transforms.Compose([
        # transforms.Resize(cfg["dataset"]["image_size"]),
        transforms.ToTensor()
    ])
else:
    transform = transforms.Compose([
        transforms.Resize(fin_size),
        transforms.ToTensor()
    ])

# === 1. Inference on a single image ===
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)  # Shape: [1, C, H, W]

    with torch.no_grad():
        output = model(input_tensor)  # Shape: [1, num_classes, H, W]
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Shape: [H, W]

    return prediction

# === Example usage ===
single_image_path = "/mnt/data/omkumar/foundation_phase1/Phase_1_Data/IDRID/test/images/IDRiD_55.jpg"  # ⬅️ Replace with your test image
mask = predict_image(single_image_path)

# Save prediction as image

results_dir = cfg['logging']['results']
results_dir = os.path.join(results_dir,cfg['dataset']['name'])

os.makedirs(results_dir,exist_ok=True)

save_fname = 'mask' + Path(single_image_path).name
Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(results_dir,save_fname)) #Saving the mask
shutil.copy(single_image_path,os.path.join(results_dir,Path(single_image_path).name)) #Copying the GT for faster access

print(f"✅ Saved prediction to {save_fname}")
