# export CUDA_VISIBLE_DEVICES=1
# export WANDB_MODE='disabled'
uv run train.py dataset='brats3d' model.name='ELitNet3D' dataset.multi_label=False 
