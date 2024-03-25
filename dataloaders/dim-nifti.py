import pydicom
import nibabel as nib
import numpy as np
import os
import gzip
import shutil

def dicom2nifti(dicom_dir, output_filename):
    
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in os.listdir(dicom_dir)]
    
 
    slices.sort(key=lambda x: int(x.InstanceNumber))

  
    image = np.stack([s.pixel_array for s in slices])

    
    img_nifti = nib.Nifti1Image(image, np.eye(4))

   
    nib.save(img_nifti, output_filename + '.nii')

   
    with open(output_filename + '.nii', 'rb') as f_in:
        with gzip.open(output_filename + '.nii.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    
    os.remove(output_filename + '.nii')


dicom2nifti('/home/xsq/xsq/TSS-CL/code/data/MASKS_DICOM/portalvein', '/home/xsq/xsq/TSS-CL/code/data/portalvein')
