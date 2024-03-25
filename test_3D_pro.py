import argparse
import os
import shutil
from glob import glob

import torch
from networks.net_factory_3d import net_factory_3d
from test_3D_util import test_all_case, test_all_case1, test_all_case2,test_all_case3

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/xsq/xsq/data_myself/GLIA-DATA/dwh-H-h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='TSS_CL_hard_168', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='Projectorvnet', help='model_name')
parser.add_argument('--gpu', type=str, default='2',
                    help='gpu id')


def Inference(FLAGS):
    snapshot_path = "/home/xsq/xsq/TSS-CL/dwh-result/TLF/model_TLF_70_30_rcps/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "/home/xsq/xsq/TSS-CL/dwh-result/TLF/model_TLF_70_30_rcps/{}/{}_Prediction".format(FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes).cuda()
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    #save_mode_path = os.path.join(snapshot_path, 'iter_8000_dice_0.6061.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric, std = test_all_case2(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(128, 128, 128), stride_xy=18, stride_z=4, test_save_path=test_save_path)
    return avg_metric, std
    #test_all_case2(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test_h5.txt", num_classes=num_classes,
                               #patch_size=(128, 128, 128), stride_xy=64, stride_z=64, test_save_path=test_save_path)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    avg_metric, std = Inference(FLAGS)
    print('dice, jc, hd, asd:', avg_metric)
    print('std:', std)
