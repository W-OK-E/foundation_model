# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='idrid' model.name='focal_with_bg_set2' dataset.multi_label=False 
