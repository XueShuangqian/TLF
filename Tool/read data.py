import h5py


with h5py.File('/home/xsq/xsq/TSS-CL/my_data/Contact/HQ-contact/Tr_001.h5', 'r') as f:
   
    for key in f.keys():
        print('Key:', key)
        print('Type:', type(f[key]))
        print('Shape:', f[key].shape)
        print('Data type:', f[key].dtype)









