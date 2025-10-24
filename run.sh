# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='cholec' model.name='focal_batched' dataset.multi_label=False dry_run=True
