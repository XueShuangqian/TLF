import glob
import h5py
from monai.transforms import Compose, AddChanneld, NormalizeIntensityd, EnsureTyped, RandGridDistortiond, RandSpatialCropd


h5_files = glob.glob('/home/xsq/xsq/data_myself/GLIA-DATA/dwh-H-h5/*.h5')


train_aug = Compose([
    #AddChanneld(keys=['image', 'label']),
    NormalizeIntensityd(keys=['image']),
    EnsureTyped(keys=['image', 'label']),
    RandGridDistortiond(keys=['image', 'label'], mode=['bilinear', 'nearest'], distort_limit=0.1),
    #RandSpatialCropd(keys=['image', 'label'], roi_size=[128, 128, 128], random_size=False),
])


for i, h5_file in enumerate(h5_files):
    with h5py.File(h5_file, 'r') as f:
        data = {'image': f['image'][:], 'label': f['label'][:]}
    enhanced_data = train_aug(data)
    with h5py.File(f'/home/xsq/xsq/data_myself/GLIA-DATA/RCPS-AUG/Tr_{i+1:03}.h5', 'w') as f:
        f.create_dataset('image', data=enhanced_data['image'].numpy())
        f.create_dataset('label', data=enhanced_data['label'].numpy())
