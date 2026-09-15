# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='cholec_8k_3d' model.name='elitnet_3d' dataset.multi_label=False 
