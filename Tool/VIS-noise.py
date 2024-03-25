import glob
import os
import re
import h5py
import numpy as np
import shutil
import scipy.ndimage
import random
import SimpleITK as sitk
import nibabel as nib

def VIS(test_list_path, ori_Dir, nifti_Dir):
    with open(test_list_path, 'r') as f:
        test_image_list = f.readlines()
    num = 0
    for case_name in test_image_list:
        case_name = case_name.replace('\n', '')
        print(case_name)
        ori_path = os.path.join(ori_Dir, case_name+'.h5')
        h5f_ori = h5py.File(ori_path, 'r')
        image_ori = h5f_ori['image']
        label_ori = h5f_ori['label']

        image_path = os.path.join(nifti_Dir, case_name+'_image'+'.nii.gz')
        image_ori_itk = sitk.GetImageFromArray(image_ori.astype(np.uint8))
        image_ori_itk.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(image_ori_itk, image_path)

        label_path = os.path.join(nifti_Dir, case_name+'_label'+'.nii.gz')
        label_ori_itk = sitk.GetImageFromArray(label_ori.astype(np.uint8))
        label_ori_itk.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(label_ori_itk, label_path)
        num += 1

def convert_h5_to_nifti(input_file, output_folder):
    
    with h5py.File(input_file, 'r') as f:
        image = f['image'][:]
        label = f['label'][:]

    label = label.astype(np.uint16)
    nii_image = nib.Nifti1Image(image, np.eye(4))
    nii_label = nib.Nifti1Image(label, np.eye(4))

    
    nib.save(nii_image, os.path.join(output_folder, 'image.nii.gz'))
    nib.save(nii_label, os.path.join(output_folder, 'label.nii.gz'))




if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    convert_h5_to_nifti('/home/xsq/xsq/data_myself/my data/dwh-H-h5/Tr_265.h5', '/home/xsq/xsq/data_myself/my data/bb')
    #vis_list_path = '//home/xsq/xsq/data_myself/my data/aa/vis_list.txt'
    #ori_Dir = '/home/xsq/xsq/data_myself/my data/aa/'
    #nifti_Dir = '/home/xsq/xsq/data_myself/my data/bb/'
    
    #VIS(vis_list_path, ori_Dir, nifti_Dir)
