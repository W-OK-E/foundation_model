export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='us_idrid' model.name='ELitNet_Shared_Head' dataset.multi_label=False 
