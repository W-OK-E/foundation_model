export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE='disabled'
uv run train.py dataset='US_Nerve' 
