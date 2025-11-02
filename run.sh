# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='cholec' model.name='focal_dice_weighted_subset6' dataset.multi_label=False 
#*niser*#