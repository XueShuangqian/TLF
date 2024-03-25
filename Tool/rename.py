import re
import os


path = "/home/xsq/xsq/GLIA-Net-master/data1_flair" #输入你要更改文件的目录



originalname = 'T1c_subtrMeanDivStd' #123是要查找文件名里包含123的文件
replacename = '' #321是要被替换的字符串，如果就是删除originalname，那么replacename = ''就可以
def main1(path1):
    files = os.listdir(path1)  # 得到文件夹下的所有文件名称
    for file in files: #遍历文件夹
        if os.path.isdir(path1 + '/' + file):
            main1(path1 + '/' + file)
        else:
            files2 = os.listdir(path1 + '/')
            for file1 in files2:
                if originalname in file1:
                    #用‘’替换掉 X变量
                    #n = str(path1 + '/' + file1.replace(file,file.split(".")[0]+file))
                    n = str(path1 + '/' + file1.replace(originalname,replacename))
                    n1 = str(path1 + '/' + str(file1))
                    try:
                        os.rename(n1, n)
                    except IOError:
                        continue
main1(path)
