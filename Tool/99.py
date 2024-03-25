import os
import shutil

names = []
with open("/home/xsq/xsq/data_myself/Process/final.txt", "r") as f:
    for line in f:
        name = line.strip() 
        names.append(name + ".nii.gz")

src_dir = "/home/xsq/xsq/data_myself/Process/H-label/"
dst_dir = "/home/xsq/xsq/data_myself/Process/low-half-H-label/"

for filename in os.listdir(src_dir):
    if filename in names:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.move(src_path, dst_path)
        #shutil.copyfile(src_path, dst_path)
