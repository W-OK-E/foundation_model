import os

# path to saving models
models_dir = 'SSIM_Laplace'

# path to saving loss plots
losses_dir = 'losses'

# path to the data directories
data_dir = '/home/omkumar/Denoising/UNET/KGP_Data'


# maximun number of synthetic words to generate
num_synthetic_imgs = 18000
train_percentage = 0.8
 # False for trainig from scratch or testing, True for loading a previously saved weight
ckpt= '/home/omkumar/Denoising/UNET/L1_Frozen2/lr_1e-05_w_0.51/L1_Frozen301.pth'
lr = 1e-5          # learning rate
epochs = 500 # epochs to train for 

# batch size for train and val loaders
batch_size = 1 # try decreasing the batch_size if there is a memory error

# log interval for training and validation
log_interval = 25

resume = False

test_dir = os.path.join(data_dir, val_dir, noisy_dir)
res_dir = 'Single_Result'
quick_res_dir = 'quick_results'
test_bs = 1
