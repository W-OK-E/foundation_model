# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='idrid' model.name='IDRID_128' dataset.multi_label=False 
