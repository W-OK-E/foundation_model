# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='cholec' model.name='1fps_large_Elitnet' dataset.multi_label=False 
