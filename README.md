# Foundation Model for Medical Imaging

This project is at aimed at building, developing and training a foundation model across multiple medical imaging datasets.

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

The virtual environment has been created in the folder /mnt/data/omkumar/foundation_model/.venv
so whatever needs to be run must be run using uv and inside the directory - /mnt/data/omkumar/foundation_model/foundation_phase1


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

## Absolute MUST: 
`/mnt/data/omkumar/foundation_model/foundation_phase1/run.sh` is the primary file that needs to be run to train the model. 
Command Breakdown:
`uv run train.py dataset='cbis' model.name='ELitNet' dataset.multi_label=False`

The dataset parameter value will depend on what the user wants to train on. The options are the names of the config 
files in the `/mnt/data/omkumar/foundation_model/foundation_phase1/configs/dataset/` folder. 