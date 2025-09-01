# Foundation Model for Medical Imaging

This repository contains code and configurations for training a foundation model across multiple medical imaging datasets. Currently, it is being trained on:

- **IDRID** (Diabetic Retinopathy)
- **US-Nerve Segmentation** (Ultrasound Nerve Images)

The goal is to build a multi-modal,multi-task and maybe a multi-resolution model using a shared architecture (EliteNet), with considerations for varying input sizes and dataset characteristics.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone git@github.com:W-OK-E/foundation_model.git
cd foundation_model
```

### 2. Set Up Environment

We recommend using conda to create and manage the environment.
```bash
conda create -n foundation_seg python=3.10 -y
conda activate foundation_seg
pip install -r requirements_clean.txt
```
### 3. Directory Structure

To begin with, we are working with segmentation dataset, and for each dataset, the following folder structure needs to be adopted:

```
<Dataset_Name>
    /train
        /images
        /labels
    /test
        /images
        /labels


Place your training and test data inside the corresponding folders.
```
### 4. Dataset Configuration

- Each dataset requires a dedicated YAML configuration file under the configs/ directory.
- Use configs/idrid.yaml as a reference.
- Paths, image size, number of classes, and other dataset-specific parameters are defined here.

#### Input Shapes and Padding

In each config file, there is a `image_size` attribute, notes on how to set that:

EliteNet downsamples the input image 7 times, each by a factor of 2. Therefore, image dimensions must be a multiple of 128.

| Dataset         | Original Shape  | Required (Padded) Shape |
|------------------|------------------|--------------------------|
| IDRID            | (2848, 4288)     | (2944, 4352)             |
| US-Nerve Seg     | (580, 420)       | (640, 512)               |

> Padding is applied using **reflect padding**.
Therefore what you must do is read data sample from the dataset and check the Original Shape, then calculate the nearest multiple of 128, that becomes your `image_size` attribute in the config file, put that in the yaml file. YOU DONOT HAVE TO DO ANY PADDING OR TRANSFORMS YOURSELVES

- If a dataset cannot be restructured into the expected folder layout (train/images, train/labels, etc.), a custom dataset/dataloader class must be written.

### 5. Training

Run training inside a tmux session to prevent loss on disconnection:

```bash
tmux new -s <session_name> //Session Name can be anything
```
Then run:
```bash
python3 train.py --config configs/idrid.yaml
```
Replace the path to the config file depending on which dataset you're training on.

#### Monitoring Training
You can monitor the training process via the `utils/monitor.py` script but you need to verify that the script is indeed sending messages to your slack app.
You can verify that by running some gibberish as:
```bash
python3 utils/monitor.py 'sfandfoanf'
```
And you should get a message on your slack app.
```bash
python3 utils/monitor.py 'python3 train.py --config <path_to_config>'
```
