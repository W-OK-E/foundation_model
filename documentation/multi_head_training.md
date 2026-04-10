# 🧠 Multi-Head Foundation Model Training

The foundation model architecture has been upgraded to support joint training on multiple datasets simultaneously. This is achieved through a **Shared Backbone** and **Dataset-Specific Heads**.

## 🏗️ Architecture Overview

1.  **Shared Encoder (Backbone)**: Learns high-level features that are generalized across all medical imaging tasks.
2.  **Multiple Decoders (Heads)**: Each dataset has its own dedicated decoder head that converts shared features into task-specific segmentations.
3.  **Multi-Dataset DataModule**: Training batches are drawn from all datasets and routed to their respective heads automatically.

---

## ⚙️ How to Configure Multi-Head Training

To train on multiple datasets, you need to update your Hydra configuration.

### 1. Define the Datasets
Create a new dataset config (e.g., `configs/dataset/multi_head.yaml`):

```yaml
name: foundation_v1
num_classes: # This will be ignored in multi-head mode
  US_Nerve: 2
  IDRiD: 6

# Define the dictionary of datasets
train_dataset:
  US_Nerve:
    _target_: data.data.SEGDataset
    root_dir: ${root_dir}/datasets/US_Nerve
    dataset_name: US_Nerve
  IDRiD:
    _target_: data.data.SEGDataset
    root_dir: ${root_dir}/datasets/Dataset001_IDRiD
    dataset_name: IDRiD

val_dataset:
  US_Nerve: ...
  IDRiD: ...

global_batch_size: 16
```

### 2. Update the Network Config
Update `configs/model/network/elitnet.yaml` to pass the `dataset_classes` map:

```yaml
instance:
  _target_: models.network.ElitNet.ElitNet
  in_channels : 3
  dataset_classes : ${dataset.num_classes}  # Now a dict
  layers : [32, 64, 128, 256]
```

---

## 🔄 Pipeline Changes

### Data Routing
The `SEGDataset` now returns a 3-tuple: `(image, mask, dataset_name)`. 
The `ElitLightModel` uses this `dataset_name` to:
1.  Route the features to the correct **Decoder Head**.
2.  Compute metrics using the correct **Dataset-specific Metrics module**.
3.  Log logs with a prefix: `train/IDRiD/loss`, `train/US_Nerve/loss`.

### Joint Training (CombinedLoader)
We use PyTorch Lightning's `CombinedLoader` in `max_size_cycle` mode. This pulls a batch from every dataset in each training step and aggregates the losses. This ensures the model is always being updated by all tasks.

---

## 📝 Best Practices
- **Balanced Batches**: Try to keep image resolutions similar or use padding/resizing via the `dataset.json` metadata.
- **Normalization**: Each dataset can still have its own `mean` and `std` calculated via the planning script; the pipeline will handle them individually.
- **Weights**: You can still save the best model based on a shared metric or specific dataset performance.
