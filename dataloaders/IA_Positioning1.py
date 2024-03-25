import numpy as np
import glob
import re
import os
from tqdm import tqdm
import h5py
import SimpleITK as sitk
import nibabel as nib
from skimage.filters import frangi, hessian, sato

output_size =[128, 128, 128]

def enhence(img_Dir, label_Dir):
    img_path = sorted(glob.glob(img_Dir))
    for case in img_path:  
        print(case)
        img_itk = sitk.ReadImage(case)  # read img
        spacing = img_itk.GetSpacing()
        image = sitk.GetArrayFromImage(img_itk) # img to numpy
        
        idx = findidx(case) # get img idx
        label_file_name = 'Tr_' + str(idx)[:] + '.nii.gz' # get label file name     
        label_path = os.path.join(label_Dir, label_file_name) # get label file path
        label_itk = sitk.ReadImage(label_path)  # read label
        label = sitk.GetArrayFromImage(label_itk)

        image = VesselEnhance(image, type='sato')
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        print("img shape2", image.shape)
        img_itk.SetSpacing(spacing)
        #img_itk.SetSpacing((1.0, 1.0, 1.0))
        label_itk = sitk.GetImageFromArray(label.astype(np.float32))
        label_itk.SetSpacing(spacing)
        #label_itk.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img_itk, '/home/xsq/xsq/data_myself/Prob/HQ-prob/Tr_{}.nii.gz'.format(str(idx)[:]))
        #sitk.WriteImage(label_itk,
                            #'/home/xsq/xsq/data_myself/p-label-prob/Tr_{}.nii.gz'.format(str(idx)[:]))
    print("Converted val IRCAD volumes to ROI")

def CT_nifti(img_Dir, label_Dir):
    img_path = sorted(glob.glob(img_Dir))
    for case in img_path:  
        print(case)
        img_itk = sitk.ReadImage(case)  # read img
        spacing = img_itk.GetSpacing()
        image = sitk.GetArrayFromImage(img_itk) # img to numpy
        
        idx = findidx(case) # get img idx
        label_file_name = 'Tr' + str(idx)[:] + '.nii.gz' # get label file name     
        label_path = os.path.join(label_Dir, label_file_name) # get label file path
        label_itk = sitk.ReadImage(label_path)  # read label
        label = sitk.GetArrayFromImage(label_itk)
        #label = (label == 255).astype(np.uint8) # 0,1

        image = CT_normalize(image) # Normalize the image
        w, h, d = label.shape


        tempL = np.nonzero(label)

        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2



        minx = max(minx - px, 0)
        maxx = min(maxx + px, w)
        miny = max(miny - py, 0)
        maxy = min(maxy + py, h)
        minz = max(minz - pz, 0)
        maxz = min(maxz + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        if minx <= 20 :
             image = image[minx:minx+128, miny-20:miny+108, minz-20:minz+108]
             label = label[minx:minx+128, miny-20:miny+108, minz-20:minz+108]
        else :
             image = image[minx-20:minx+108, miny-20:miny+108, minz-20:minz+108]
             label = label[minx-20:minx+108, miny-20:miny+108, minz-20:minz+108]
        print("label shape", label.shape)

        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.SetSpacing(spacing)
        #img_itk.SetSpacing((1.0, 1.0, 1.0))
        label_itk = sitk.GetImageFromArray(label.astype(np.float32))
        label_itk.SetSpacing(spacing)
        #label_itk.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img_itk, '/home/xsq/xsq/data_myself/LQ-img-H/Tr_{}.nii.gz'.format(str(idx)[:]))
        sitk.WriteImage(label_itk,
                            '/home/xsq/xsq/data_myself/LQ-label-H/Tr_{}.nii.gz'.format(str(idx)[:]))
    print("Converted val IRCAD volumes to ROI")

def nifti_covert_h5(img_Dir, label_Dir):
    img_path = sorted(glob.glob(img_Dir))
    for case in img_path:  
        print(case)
        img_itk = sitk.ReadImage(case)  # read img
        spacing = img_itk.GetSpacing()
        image = sitk.GetArrayFromImage(img_itk) # img to numpy
        
        idx = findidx(case) # get img idx
        label_file_name = 'Tr_' + str(idx)[:] + '.nii.gz' # get label file name     
        label_path = os.path.join(label_Dir, label_file_name) # get label file path
        label_itk = sitk.ReadImage(label_path)  # read label
        label = sitk.GetArrayFromImage(label_itk)

        image = CT_normalize(image) # Normalize the image

        print(image.shape)
        print('---------------')
        f = h5py.File(
                '/home/xsq/xsq/data_myself/H5/LQ-Prob-H-h5/Tr_{}.h5'.format(str(idx)[:]), 'w')

        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.create_dataset('spacing', data=np.array(spacing), compression="gzip")
        f.close()
    print("IA finished")


def covert_h5_1(img_Dir, label_Dir):

    img_path = sorted(glob.glob(img_Dir))
    for case in img_path:  
        print(case)
        img_itk = sitk.ReadImage(case)  # read img
        spacing = img_itk.GetSpacing()
        image = sitk.GetArrayFromImage(img_itk) # img to numpy
        
        idx = findidx(case) # get img idx
        label_file_name = 'Tr' + str(idx)[1:] + '.nii.gz' # get label file name     
        label_path = os.path.join(label_Dir, label_file_name) # get label file path
        label_itk = sitk.ReadImage(label_path)  # read label
        label = sitk.GetArrayFromImage(label_itk)
        #label = (label == 255).astype(np.uint8) # 0,1

        image = CT_normalize(image) # Normalize the image
        d, w, h = label.shape


        tempL = np.nonzero(label)

        minz, maxz = np.min(tempL[0]), np.max(tempL[0])
        minx, maxx = np.min(tempL[1]), np.max(tempL[1])
        miny, maxy = np.min(tempL[2]), np.max(tempL[2])

        pz = max(output_size[0] - (maxz - minz), 0) // 2
        px = max(output_size[1] - (maxx - minx), 0) // 2
        py = max(output_size[2] - (maxy - miny), 0) // 2

        minx = max(minx - np.random.randint(5, 10) - px, 0)
        maxx = min(maxx + np.random.randint(5, 10) + px, w)
        miny = max(miny - np.random.randint(5, 10) - py, 0)
        maxy = min(maxy + np.random.randint(5, 10) + py, h)
        minz = max(minz - np.random.randint(0, 5) - pz, 0)
        maxz = min(maxz + np.random.randint(0, 5) + pz, d)

        #minx = max(minx - px, 0)
        #maxx = min(maxx + px, w)
        #miny = max(miny - py, 0)
        #maxy = min(maxy + py, h)
        #minz = max(minz - pz, 0)
        #maxz = min(maxz + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minz:maxz, minx:maxx, miny:maxy]
        label = label[minz:maxz, minx:maxx, miny:maxy]
        print("label shape", label.shape)
        f = h5py.File(
                '/home/xsq/xsq/TSS-CL/my_data/train_total/all-LQ_h5/Tr_{}.h5'.format(str(idx)[1:]), 'w')

        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()
    print("IA Positioning finished")

def covert_h5_2(img_Dir, label_Dir):

    img_path = sorted(glob.glob(img_Dir))
    for case in img_path:  
        print(case)
        img_itk = sitk.ReadImage(case)  # read img
        spacing = img_itk.GetSpacing()
        image = sitk.GetArrayFromImage(img_itk) # img to numpy
        
        idx = findidx(case) # get img idx
        label_file_name = 'Tr' + str(idx)[:] + '.nii.gz' # get label file name     
        label_path = os.path.join(label_Dir, label_file_name) # get label file path
        label_itk = sitk.ReadImage(label_path)  # read label
        label = sitk.GetArrayFromImage(label_itk)
        #label = (label == 255).astype(np.uint8) # 0,1

        image = CT_normalize(image) # Normalize the image
        d, w, h = label.shape


        tempL = np.nonzero(label)

        minz, maxz = np.min(tempL[0]), np.max(tempL[0])
        minx, maxx = np.min(tempL[1]), np.max(tempL[1])
        miny, maxy = np.min(tempL[2]), np.max(tempL[2])

        pz = max(output_size[0] - (maxz - minz), 0) // 2
        px = max(output_size[1] - (maxx - minx), 0) // 2
        py = max(output_size[2] - (maxy - miny), 0) // 2



        minx = max(minx - px, 0)
        maxx = min(maxx + px, w)
        miny = max(miny - py, 0)
        maxy = min(maxy + py, h)
        minz = max(minz - pz, 0)
        maxz = min(maxz + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        if minz <= 20 :
             image = image[minz:minz+128, minx-20:minx+108, miny-20:miny+108]
             label = label[minz:minz+128, minx-20:minx+108, miny-20:miny+108]
        else :
             image = image[minz-20:minz+108, minx-20:minx+108, miny-20:miny+108]
             label = label[minz-20:minz+108, minx-20:minx+108, miny-20:miny+108]
        print("label shape", label.shape)
        f = h5py.File(
                '/home/xsq/xsq/data_myself/cc/Tr_{}.h5'.format(str(idx)[1:]), 'w')

        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()
    print("IA Positioning finished")

def covert_h5_3(img_Dir, label_Dir):

    img_path = sorted(glob.glob(img_Dir))
    for case in img_path:  
        print(case)
        img_itk = sitk.ReadImage(case)  # read img
        spacing = img_itk.GetSpacing()
        image = sitk.GetArrayFromImage(img_itk) # img to numpy
        
        idx = findidx(case) # get img idx
        label_file_name = 'Tr' + str(idx)[:] + '.nii.gz' # get label file name     
        label_path = os.path.join(label_Dir, label_file_name) # get label file path
        label_itk = sitk.ReadImage(label_path)  # read label
        label = sitk.GetArrayFromImage(label_itk)
        #label = (label == 255).astype(np.uint8) # 0,1

        image = CT_normalize(image) # Normalize the image
        w, h, d = label.shape


        tempL = np.nonzero(label)

        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2



        minx = max(minx - px, 0)
        maxx = min(maxx + px, w)
        miny = max(miny - py, 0)
        maxy = min(maxy + py, h)
        minz = max(minz - pz, 0)
        maxz = min(maxz + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        if minx <= 20 :
             image = image[minx:minx+128, miny-20:miny+108, minz-20:minz+108]
             label = label[minx:minx+128, miny-20:miny+108, minz-20:minz+108]
        else :
             image = image[minx-20:minx+108, miny-20:miny+108, minz-20:minz+108]
             label = label[minx-20:minx+108, miny-20:miny+108, minz-20:minz+108]
        print("label shape", label.shape)
        f = h5py.File(
                '/home/xsq/xsq/data_myself/Process/Crop-H-h5/Tr_{}.h5'.format(str(idx)[:]), 'w')

        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.create_dataset('spacing', data=np.array(spacing), compression="gzip")
        f.close()
    print("IA Positioning finished")

def CT_normalize(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    # Default: [0, 400]
    MIN_BOUND = -75.0
    MAX_BOUND = 250.0

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data

def findidx(file_name):
    # find the idx
    cop = re.compile("[^0-9]")
    idx = cop.sub('', file_name)
    return idx

def VesselEnhance(img, type):
    if type == 'sato':
        filter_img = sato(img, sigmas=range(1, 4, 1), black_ridges=False, mode='constant')
    elif type == 'frangi':
        filter_img = frangi(img, sigmas=range(1, 4, 1), scale_range=None,
                            scale_step=None, alpha=0.5, beta=0.5, gamma=5, black_ridges=False, mode='constant', cval=1)
    return filter_img

if __name__ == '__main__':
    img_Dir = '/home/xsq/xsq/data_myself/my data/image/*.nii.gz'
    label_Dir = '/home/xsq/xsq/data_myself/my data/label/'
    covert_h5_3(img_Dir, label_Dir)
    #covert_nifti(img_Dir, label_Dir)
    #nifti_covert_h5(img_Dir, label_Dir)
    #covert_nifti(img_Dir, label_Dir)
    #CT_nifti(img_Dir, label_Dir)
    #enhence(img_Dir, label_Dir)
