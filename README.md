# Foundation Model for Medical Imaging

This repository contains code and configurations for training a foundation model across multiple medical imaging datasets. Currently, it is being trained on:

- **IDRID** (Diabetic Retinopathy)
- **US-Nerve Segmentation** (Ultrasound Nerve Images)

The goal is to build a multi-modal,multi-task and maybe a multi-resolution model using a shared architecture (EliteNet), with considerations for varying input sizes and dataset characteristics.


## Preparing Datasets

SO far the dataset must be organized in a specific structure for the model to correctly load images and masks during training, validation, and testing, we hope to maintain the same uniform structure as we expand to different datasets and more modalities

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