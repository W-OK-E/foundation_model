export CUDA_VISIBLE_DEVICES=0,1
uv run train.py dataset='US_Nerve' mode='predict'