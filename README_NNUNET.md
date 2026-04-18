# nnU-Net Integration for Foundation Models

This document outlines the changes and additions made to integrate core philosophies and components from the `nnUNet` (v2) framework into the `foundation_phase1` codebase. These updates enable the creation of highly flexible "foundation models" capable of handling any modality (2D/3D), arbitrary input channels, and dynamic network depths.

## 🚀 Key Features

### 1. Dynamic UNet Architecture
- **Location**: `models/network/DynamicUNet.py`
- **Description**: A fully configurable UNet implementation that can switch between 2D and 3D operations based on a single parameter (`spatial_dims`).
- **Foundation Capabilities**:
    - **Modality Agnostic**: Works with any spatial dimensions.
    - **Channel Agnostic**: Input channels are dynamically determined from the dataset configuration.
    - **Flexible Depth**: Customizable number of stages, features per stage, and convolution/pooling operations.
    - **Deep Supervision**: Optional support for hierarchical output heads (returning a list of tensors) to enhance training stability.

### 2. Advanced Training Components
- **Deep Supervision Wrapper** (`loss/deep_supervision.py`): A wrapper that allows the model to compute loss across multiple output scales, automatically handling the downsampling of ground truth masks.
- **PolyLR Scheduler** (`utils/lr_scheduler.py`): Ported from nnU-Net, this scheduler effectively handles the learning rate decay according to a polynomial power law, which is standard for medical image segmentation.
- **Compound DC + CE Loss** (`loss/compound_losses.py`): Combines Soft Dice Loss and CrossEntropy Loss. This combination is the SOTA standard for robust medical segmentation performance.

### 3. Integrated Hydra Configurations
New configuration files have been added to allow immediate use of the integrated components:
- **Architecture**: `configs/model/network/dynamic_unet.yaml`
- **Loss**: `configs/model/loss/dc_ce.yaml`
- **Scheduler**: `configs/model/lr_scheduler/poly_lr.yaml`
- **Combined Baseline**: `configs/model/dynamic_baseline.yaml`

## 🛠 Usage

### Training with the Foundation Model
To start training using the new dynamic UNet and nnU-Net style training components, use the following command:

```bash
python train.py model=dynamic_baseline dataset=your_dataset_name
```

### Switching to 3D
To switch the architecture and preprocessing for 3D tasks, ensure your dataset is configured as 3D and override the model's spatial dimensions:

```bash
python train.py model=dynamic_baseline dataset=your_dataset_3d model.network.instance.spatial_dims=3
```

### Enabling Deep Supervision
Deep supervision can be enabled via config or CLI. When enabled, the loss is automatically computed across all levels:

```bash
python train.py model=dynamic_baseline model.network.instance.deep_supervision=True
```

## 📝 Modified Files
- `models/module.py`: Updated the `ElitLightModel` (LightningModule) to support multi-scale outputs during training and validation while maintaining standard metric computation on the highest resolution output.

---
*Changes implemented to support the transition towards a flexible foundation model pipeline.*
