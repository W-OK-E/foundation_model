# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='brats3d' model.name='elitnet_3d' dataset.multi_label=False 
