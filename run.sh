# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='cataract' model.name='elit_conv2d' dataset.multi_label=False 
