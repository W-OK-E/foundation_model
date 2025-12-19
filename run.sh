# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='cataract' model.name='Cataracts_set2' dataset.multi_label=False