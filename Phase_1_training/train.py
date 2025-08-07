import os
import yaml
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from EliteNet import UNet              # Replace with your actual model import
from dataset import CustomDataset  # Replace with your actual dataset import

# === Load config.yaml ===
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# === Set random seed and device ===
torch.manual_seed(cfg["training"]["seed"])
device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

# === Define transforms ===
transform = transforms.Compose([
    transforms.Resize(cfg["dataset"]["image_size"]),
    transforms.ToTensor()
])

# === Create datasets & dataloaders ===
train_dataset = CustomDataset(
    name=cfg["dataset"]["name"],
    data_dir=cfg["dataset"]["path"],
    img_sz=tuple(cfg["dataset"]["image_size"]),
    is_train=True,
    transform=transform,
    cache_dir=cfg["dataset"]["cache_dir"]
)

train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

# === Model ===
model = UNet(
    in_c=cfg["model"]["in_channels"],
    n_classes=cfg["dataset"]["num_classes"],
    layers=[4, 8, 16]
).to(device)

# === Loss Function ===
loss_fn = getattr(nn, cfg["loss"]["name"])()

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
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=cfg["lr_scheduler"]["step_size"],
                                              gamma=cfg["lr_scheduler"]["gamma"])
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode=cfg["lr_scheduler"]["mode"],
                                                         factor=cfg["lr_scheduler"]["factor"],
                                                         patience=cfg["lr_scheduler"]["patience"])
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# === Training Loop ===
epochs = cfg["training"]["epochs"]
log_interval = cfg["logging"]["log_interval"]

for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    total_loss = 0

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        # print(images.shape)
        # print(masks.shape)
        # print(masks.max(),masks.min())

        # import sys
        # sys.exit(0)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % log_interval == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Step the scheduler
    if scheduler:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(total_loss)
        else:
            scheduler.step()

    # Save model
    if (epoch + 1) % cfg["logging"]["save_interval"] == 0:
        os.makedirs(cfg["logging"]["save_dir"], exist_ok=True)
        save_path = os.path.join(cfg["logging"]["save_dir"], f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
