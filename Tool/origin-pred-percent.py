import os
import numpy as np
import nibabel as nib
import h5py
from PIL import Image


def calculate_voxel_ratios_nifti(original_folder, predicted_folder):
    unmarked_aneurysm_voxels = 0
    mislabeled_aneurysm_voxels = 0
    total_voxels = 0

   
    for original_file, predicted_file in zip(sorted(os.listdir(original_folder)), sorted(os.listdir(predicted_folder))):
        
        original_labels = nib.load(os.path.join(original_folder, original_file)).get_fdata()
        predicted_labels = nib.load(os.path.join(predicted_folder, predicted_file)).get_fdata()
        
        total_voxels += np.sum(original_labels == 1)

        
        unmarked_aneurysm_voxels += np.sum((original_labels == 1) & (predicted_labels == 0))

        
        mislabeled_aneurysm_voxels += np.sum((original_labels == 0) & (predicted_labels == 1))

    
    unmarked_aneurysm_ratio = unmarked_aneurysm_voxels / total_voxels
    mislabeled_aneurysm_ratio = mislabeled_aneurysm_voxels / total_voxels

    return unmarked_aneurysm_ratio, mislabeled_aneurysm_ratio


def calculate_voxel_ratios_h5(original_folder, predicted_folder):
    unmarked_aneurysm_voxels = 0
    mislabeled_aneurysm_voxels = 0
    total_voxels = 0

   
    for original_file, predicted_file in zip(sorted(os.listdir(original_folder)), sorted(os.listdir(predicted_folder))):
        
        with h5py.File(os.path.join(original_folder, original_file), 'r') as f:
            original_labels = f['label'][:]
        with h5py.File(os.path.join(predicted_folder, predicted_file), 'r') as f:
            predicted_labels = f['label'][:]  
      
        total_voxels += np.sum(original_labels == 1)

        
        unmarked_aneurysm_voxels += np.sum((original_labels == 1) & (predicted_labels == 0))

        
        mislabeled_aneurysm_voxels += np.sum((original_labels == 0) & (predicted_labels == 1))

    
    unmarked_aneurysm_ratio = unmarked_aneurysm_voxels / total_voxels
    mislabeled_aneurysm_ratio = mislabeled_aneurysm_voxels / total_voxels

    return unmarked_aneurysm_ratio, mislabeled_aneurysm_ratio

original_folder = "/home/xsq/xsq/data_myself/GLIA-DATA/dwh-H-h5/"
predicted_folder = "/home/xsq/xsq/data_myself/GLIA-DATA/dwh-noise-h5/"

unmarked_ratio, mislabeled_ratio = calculate_voxel_ratios_h5(original_folder, predicted_folder)
print("unlabel ia voxel percent:", unmarked_ratio)
print("error label ia voxel percent:", mislabeled_ratio)
