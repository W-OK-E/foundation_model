# ELiTNet

Source code for our paper "ATTENTION IN A LITTLE NETWORK IS ALL YOU NEED TO GO GREEN", accepted in ISBI 2023.

## Preparing Datasets

The dataset must be organized in a specific structure for the model to correctly load images and masks during training, validation, and testing.

### Expected Folder Structure
```
datasets/
└── IDRiD/
├── images/
│ ├── image_1.png
│ ├── image_2.png
│ └── ...
├── masks/
│ ├── image_1.png
│ ├── image_2.png
│ └── ...
├── train.txt
├── val.txt
└── test.txt
```
- **images/**: Contains all input images.
- **masks/**: Contains corresponding ground truth masks.
- **train.txt**, **val.txt**, **test.txt**:  
  Each of these text files contains the names of the samples to be used for the respective split.

---

### Important Notes

- The dataset folder (e.g., `IDRiD`) must be placed at the same level as the dataset’s `.yaml` config file.
- Example entry in `train.txt`:
```bash
IDRiD_55.png
IDRiD_56.png
IDRiD_57.png
IDRiD_58.png
```
## Getting Started with Training

Follow these steps to get up and running with the project.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ELiTNet.git
cd ELiTNet
```

### 2. Install UV

Install uv via curl:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This will install uv and set up the environment management system.

### 3. Configure Weights & Biases (wandb)

- Go to https://wandb.ai/ and create an account.

- During account creation, you will be asked to create an organization.This organization name is your entity.

- Create a new project in your WandB dashboard.The project name is your project.

- Open configs/config.yaml and update the logger parameters:
```bash
logger:
  entity: your-entity-name
  project: your-project-name
```

### 4. Login to WandB

Run the training script using uv, which will prompt you to log in to wandb:
```bash
uv run train.py
```
Select Use an existing account when prompted.

Paste your WandB API key (available in your WandB project dashboard).

Once done, your training runs will be tracked in WandB automatically.