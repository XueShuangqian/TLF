"""
@author: Zhe XU
Data Preprocessing for IRCAD Probability Map Dataset
1. slice extraction for training set
2. Normalization
3. Convert to h5 file
"""


import glob
import os
import re
import h5py
import numpy as np
import SimpleITK as sitk
import nibabel as nib


def findidx(file_name):
    # find the idx
    cop = re.compile("[^0-9]")
    idx = cop.sub('', file_name)
    return idx


def contact_process(test_img_Dir, test_prob_Dir, msk_baseDir):
    test_img_path = sorted(glob.glob(test_img_Dir))
    for case in test_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path, better than the re, sub code
        idx = findidx(case)

        prob_path = os.path.join(test_prob_Dir, 'Tr_' + str(idx) + '.nii.gz')
        print(prob_path)
        prob_itk = sitk.ReadImage(prob_path)
        prob = sitk.GetArrayFromImage(prob_itk)

        label_file_name = 'Tr_' + str(idx) + '.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            print('image shape:', image.shape)
            print('mask shape:', mask.shape)
            image = image.astype(np.float32)
            prob = prob.astype(np.float32)

            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            f = h5py.File(
                '/home/xsq/xsq/data_myself/img-prob-h/Tr_{}.h5'.format(str(idx)), 'w')
            concat_img = np.concatenate((image, prob), axis=0)
            print(concat_img.shape)
            f.create_dataset('image', data=concat_img, compression="gzip")
            f.create_dataset('label', data=mask, compression="gzip")
            f.close()
    print("Converted test concatenated IRCAD volumes to h5 files")


if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train_img_Dir =  '/home/xsq/xsq/data_myself/p-img/*.nii.gz'
    train_prob_Dir = '/home/xsq/xsq/data_myself/p-img-prob/'

    # New ROI vessel
    msk_baseDir = '/home/xsq/xsq/data_myself/p-label/'
    contact_process(train_img_Dir, train_prob_Dir,  msk_baseDir)




