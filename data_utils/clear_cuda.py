import torch
import gc

# # Delete references to the model and other tensors
# del model
# del optimizer
# del loss_variable

# Run Python's garbage collector
gc.collect()

# Clear the CUDA memory cache
torch.cuda.empty_cache()
