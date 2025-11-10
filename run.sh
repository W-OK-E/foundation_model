# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='cholec' model.name='Unext_Conv2d' dataset.multi_label=False 
