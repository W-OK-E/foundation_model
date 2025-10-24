# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='cholec' model.name='batch_overfit_set2' dataset.multi_label=False 
