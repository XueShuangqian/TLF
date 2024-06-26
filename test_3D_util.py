import math
import random
import time

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm
import cleanlab

def test_single_case_dwh(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    d, w, h = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if d < patch_size[0]:
        d_pad = patch_size[0]-d
        add_pad = True
    else:
        d_pad = 0
    if w < patch_size[1]:
        w_pad = patch_size[1]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[2]:
        h_pad = patch_size[2]-h
        add_pad = True
    else:
        h_pad = 0
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    if add_pad:
        image = np.pad(image, [(dl_pad, dr_pad), (wl_pad, wr_pad),
                               (hl_pad, hr_pad)], mode='constant', constant_values=0)
    dd, ww, hh = image.shape

    sz = math.ceil((dd - patch_size[0]) / stride_z) + 1
    sx = math.ceil((ww - patch_size[1]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[2]) / stride_xy) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for z in range(0, sz):
        zs = min(stride_z*z, dd-patch_size[0])
        for x in range(0, sx):
            xs = min(stride_xy * x, ww-patch_size[1])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh-patch_size[2])
                test_patch = image[zs:zs+patch_size[0],
                                   xs:xs+patch_size[1], ys:ys+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)['out']
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, zs:zs+patch_size[0], xs:xs+patch_size[1], ys:ys+patch_size[2]] \
                    = score_map[:, zs:zs+patch_size[0], xs:xs+patch_size[1], ys:ys+patch_size[2]] + y
                cnt[zs:zs+patch_size[0], xs:xs+patch_size[1], ys:ys+patch_size[2]] \
                    = cnt[zs:zs+patch_size[0], xs:xs+patch_size[1], ys:ys+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[dl_pad:dl_pad+d,
                              wl_pad:wl_pad+w, hl_pad:hl_pad+h]
        score_map = score_map[:, dl_pad:dl_pad +
                              d, wl_pad:wl_pad+w, hl_pad:hl_pad+h]
    return label_map

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map

def test_single_case1(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)['out']
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jaccard = metric.binary.jc(pred, gt)
    #ravd = abs(metric.binary.ravd(pred, gt))
    if 0 == np.count_nonzero(pred):
       return np.array([dice, jaccard, 50, 50])
    else:
       hd = metric.binary.hd95(pred, gt)
       asd = metric.binary.asd(pred, gt)
    return np.array([dice, jaccard, hd, asd])


def test_all_case(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=18, stride_z=4, test_save_path=None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    #image_list = random.sample(image_list, 30)
    total_metric = np.zeros((num_classes-1, 4))
    metric_dice = []
    metric_jac = []
    metric_hd = []
    metric_asd = []
    print("Testing begin")
    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for image_path in image_list:
            print("Processing image: {}".format(image_path))
            ids = image_path.split("/")[-1].replace(".h5", "")
            h5f = h5py.File(image_path, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:] 
            prediction = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

            metric = calculate_metric_percase(prediction == 1, label == 1)
            print(metric)
            total_metric[0, :] += metric
            metric_dice.append(metric[0])
            metric_jac.append(metric[1])
            metric_hd.append(metric[2])
            metric_asd.append(metric[3])
            f.writelines("{},{},{},{},{}\n".format(
                ids, metric[0], metric[1], metric[2], metric[3]))

            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(pred_itk, test_save_path +
                            "/{}.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(image)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, test_save_path +
                            "/{}_img.nii.gz".format(ids))

            lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
            lab_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(lab_itk, test_save_path +
                            "/{}_lab.nii.gz".format(ids))
        f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
            image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
    f.close()
    print("Testing end")
    average = total_metric / len(image_list)
    std = [np.std(metric_dice), np.std(metric_jac), np.std(metric_hd), np.std(metric_asd)]
    return average, std


def test_all_case1(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=18, stride_z=4, test_save_path=None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))
    metric_dice = []
    metric_jac = []
    metric_hd = []
    metric_asd = []
    sum_metric = np.zeros((1, 4))
    average_list = []
    num = 10
    print("Testing begin")
    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for i in range(num):
            print("Testing roung {}".format(i+1))
            image_list_i = random.sample(image_list, 30)
            total_metric = np.zeros((1, 4))
            for image_path in image_list_i:
               ids = image_path.split("/")[-1].replace(".h5", "")
               h5f = h5py.File(image_path, 'r')
               image = h5f['image'][:]
               label = h5f['label'][:]
               prediction = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

               metric = calculate_metric_percase(prediction == 1, label == 1)
               print(metric)
               total_metric[0, :] += metric
               metric_dice.append(metric[0])
               metric_jac.append(metric[1])
               metric_hd.append(metric[2])
               metric_asd.append(metric[3])
               f.writelines("{},{},{},{},{}\n".format(
                  ids, metric[0], metric[1], metric[2], metric[3]))

               pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
               pred_itk.SetSpacing((1.0, 1.0, 1.0))
               #pred_itk.SetSpacing(pred_itk)
               sitk.WriteImage(pred_itk, test_save_path +
                            "/{}_pred.nii.gz".format(ids))

               img_itk = sitk.GetImageFromArray(image)
               img_itk.SetSpacing((1.0, 1.0, 1.0))
               sitk.WriteImage(img_itk, test_save_path +
                            "/{}_img.nii.gz".format(ids))

               lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
               lab_itk.SetSpacing((1.0, 1.0, 1.0))
               sitk.WriteImage(lab_itk, test_save_path +
                            "/{}_lab.nii.gz".format(ids))
            f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
                 image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
            
            sum_metric += total_metric
            average = total_metric / len(image_list_i)
            average_list.append(average)
            
    f.close()
    print("Testing end")
    average_array = np.array(average_list)
    std = np.std(average_array, axis=0)
    average = [np.mean(metric_dice), np.mean(metric_jac), np.mean(metric_hd), np.mean(metric_asd)]
    return average, std

def test_all_case2(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=18, stride_z=4, test_save_path=None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))
    metric_dice = []
    metric_jac = []
    metric_hd = []
    metric_asd = []
    sum_metric = np.zeros((1, 4))
    average_list = []
    num = 10
    print("Testing begin")
    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for i in range(num):
            print("Testing roung {}".format(i+1))
            image_list_i = random.sample(image_list, 30)
            total_metric = np.zeros((1, 4))
            for image_path in image_list_i:
               ids = image_path.split("/")[-1].replace(".h5", "")
               h5f = h5py.File(image_path, 'r')
               image = h5f['image'][:]
               label = h5f['label'][:]
               prediction = test_single_case1(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

               metric = calculate_metric_percase(prediction == 1, label == 1)
               print(metric)
               total_metric[0, :] += metric
               metric_dice.append(metric[0])
               metric_jac.append(metric[1])
               metric_hd.append(metric[2])
               metric_asd.append(metric[3])
               f.writelines("{},{},{},{},{}\n".format(
                  ids, metric[0], metric[1], metric[2], metric[3]))

               pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
               pred_itk.SetSpacing((1.0, 1.0, 1.0))
               #pred_itk.SetSpacing(pred_itk)
               sitk.WriteImage(pred_itk, test_save_path +
                            "/{}_pred.nii.gz".format(ids))

               #img_itk = sitk.GetImageFromArray(image)
               #img_itk.SetSpacing((1.0, 1.0, 1.0))
               #sitk.WriteImage(img_itk, test_save_path +
                #            "/{}_img.nii.gz".format(ids))

#               lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
 #              lab_itk.SetSpacing((1.0, 1.0, 1.0))
  #             sitk.WriteImage(lab_itk, test_save_path +
   #                         "/{}_lab.nii.gz".format(ids))
            f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
                 image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
            
            sum_metric += total_metric
            average = total_metric / len(image_list_i)
            average_list.append(average)
            
    f.close()
    print("Testing end")
    average_array = np.array(average_list)
    std = np.std(average_array, axis=0)
    average = [np.mean(metric_dice), np.mean(metric_jac), np.mean(metric_hd), np.mean(metric_asd)]
    return average, std

def test_all_case3(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=18, stride_z=4, test_save_path=None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    #image_list = random.sample(image_list, 30)
    total_metric = np.zeros((num_classes-1, 4))
    metric_dice = []
    metric_jac = []
    metric_hd = []
    metric_asd = []
    print("Testing begin")
    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for image_path in image_list:
            print("Processing image: {}".format(image_path))
            ids = image_path.split("/")[-1].replace(".h5", "")
            h5f = h5py.File(image_path, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:] 
            prediction = test_single_case1(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

            metric = calculate_metric_percase(prediction == 1, label == 1)
            print(metric)
            total_metric[0, :] += metric
            metric_dice.append(metric[0])
            metric_jac.append(metric[1])
            metric_hd.append(metric[2])
            metric_asd.append(metric[3])
            f.writelines("{},{},{},{},{}\n".format(
                ids, metric[0], metric[1], metric[2], metric[3]))

            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(pred_itk, test_save_path +
                            "/{}.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(image)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, test_save_path +
                            "/{}_img.nii.gz".format(ids))

            lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
            lab_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(lab_itk, test_save_path +
                            "/{}_lab.nii.gz".format(ids))
        f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
            image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
    f.close()
    print("Testing end")
    average = total_metric / len(image_list)
    std = [np.std(metric_dice), np.std(metric_jac), np.std(metric_hd), np.std(metric_asd)]
    return average, std


    
