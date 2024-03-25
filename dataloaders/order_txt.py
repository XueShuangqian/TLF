list = []
with open('/home/xsq/xsq/data_myself/GLIA-DATA/dwh-H-h5/aa.txt', 'r') as f:
    for line in f:
        list.append(line.strip())
 
with open("/home/xsq/xsq/data_myself/GLIA-DATA/dwh-H-h5/train.txt", "w") as f:
    for item in sorted(list):
        f.writelines(item)
        f.writelines('\n')
    f.close()


