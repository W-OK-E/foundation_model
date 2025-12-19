# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='inbreast' model.name='INBreast' dataset.multi_label=False 
