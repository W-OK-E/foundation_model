import h5py
import numpy as np
import SimpleITK as sitk
import subprocess
import scipy.ndimage
import nibabel as nib

def load_h5_volume(filepath):
    with h5py.File(filepath, 'r') as f:
        volume = f['data'][:]  # Adjust the key as needed
    return volume

def register_to_atlas(volume_np, atlas_path):
    img = sitk.GetImageFromArray(volume_np)
    atlas_img = sitk.ReadImage(atlas_path)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(atlas_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    out_img = resampler.Execute(img)
    return sitk.GetArrayFromImage(out_img)

def skull_strip(input_nifti, output_nifti):
    subprocess.run(['hd-bet', '-i', input_nifti, '-o', output_nifti, '-mode', 'fast'], check=True)
    # Read output_nifti in your pipeline

def resize_volume(volume_np, target_shape=(240,240,155)):
    factors = [float(t)/s for t, s in zip(target_shape, volume_np.shape)]
    resized = scipy.ndimage.zoom(volume_np, factors, order=1)  # Linear interpolation
    return resized

def extract_patches(volume, patch_size=(128,128,128), stride=(64,64,64)):
    patches = []
    z_max, y_max, x_max = volume.shape
    ps_z, ps_y, ps_x = patch_size
    for z in range(0, z_max-ps_z+1, stride[0]):
        for y in range(0, y_max-ps_y+1, stride[1]):
            for x in range(0, x_max-ps_x+1, stride[2]):
                patch = volume[z:z+ps_z, y:y+ps_y, x:x+ps_x]
                patches.append(patch)
    return patches

def save_nifti(volume_np, filepath):
    img = nib.Nifti1Image(volume_np, affine=np.eye(4))
    nib.save(img, filepath)

# Example complete routine
volume_np = load_h5_volume('volume_1_slice_0.h5')
registered = register_to_atlas(volume_np, 'SRI24_T1.nii.gz')
skull_stripped = skull_strip_simple(registered)
resized = resize_volume(skull_stripped)
patches = extract_patches(resized)
save_nifti(resized, 'preprocessed_volume.nii.gz')

