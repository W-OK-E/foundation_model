# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='idrid' model.name='focal_multi_label' dataset.multi_label=True 
