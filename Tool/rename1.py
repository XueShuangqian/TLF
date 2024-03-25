import os

src_dir = "/home/xsq/xsq/TSS-CL/my_data/whd/aa/"
new_dir = "/home/xsq/xsq/TSS-CL/my_data/whd/train-50-50/"

files = os.listdir(src_dir)
files.sort()

for i, filename in enumerate(files):
    name , ext = os.path.splitext(filename)
    new_name = "Tr_{:03d}.h5".format(i+1)
    os.rename(os.path.join(src_dir, filename), os.path.join(new_dir, new_name))
