import argparse
import logging
import os
import random
import shutil
import sys
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.la_heart import (LA_heart, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from base.base_modules import TensorBuffer, NegativeSamplingPixelContrastiveLoss
from utils import losses, metrics, ramps, VAT1, tau
from val_3D import test_all_case, test_all_case1

# Confident Learning module
import cleanlab

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/xsq/xsq/data_myself/GLIA-DATA/train_70_30', help='Name of Experiment')##### Change dataset
parser.add_argument('--val_path', type=str,
                    default='/home/xsq/xsq/data_myself/GLIA-DATA/val', help='Name of Experiment')
parser.add_argument('--root_prob_path', type=str,
                    default='/home/xsq/xsq/data_myself/GLIA-DATA/RCPS-AUG', help='Name of Experiment')

parser.add_argument('--exp', type=str,
                    default='TSS_CL_hard', help='experiment_name')#### Change exp name
parser.add_argument('--model', type=str,
                    default='Projectorvnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=20000,  help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1',
                    help='gpu id')
parser.add_argument('--pretrain_model', type=str, default=None, help='pretrain model')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--clean_labeled_num', type=int, default=168,##### Accordingly change HQ setting
                    help='labeled data')
parser.add_argument('--total_sample', type=int, default=240,
                    help='total samples')
# costs-
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')

# CL
parser.add_argument('--CL_type', type=str,
                    default='both', help='CL implement type')
parser.add_argument('--weak_weight', type=float,
                    default=5.0, help='weak_weight')
parser.add_argument('--refine_type', type=str,
                    default="uncertainty_smooth", help='refine types')               

# entropy and consistency
parser.add_argument('--use_consistency', type=int,
                    default=1, help='see if use the perturbed consistency (image-only information)') 
parser.add_argument('--use_entropymin', type=int,# degradation
                    default=0, help='see if use the entropy minimization')      


args = parser.parse_args()
ema_model_idx = 0

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model1, ema_model2, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    #ema_model = random.choice([ema_model1, ema_model2]) # random choose ema_model updata
    global ema_model_idx
    if ema_model_idx == 0:
       ema_model = ema_model1
    else:
       ema_model = ema_model2
    ema_model_idx = (ema_model_idx + 1)%2

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    train_prob_data_path = args.root_prob_path
    batch_size = args.batch_size
    patch_size = args.patch_size
    max_iterations = args.max_iterations
    num_classes = 2
    pretrain_model = args.pretrain_model
    

    def create_model(ema=False):
        net = net_factory_3d(net_type=args.model,
                             in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    if pretrain_model:
       model.load_state_dict(torch.load(pretrain_model))
       print("loaded Pretrained Model")

    ema_model1 = create_model(ema=True)
    ema_model2 = create_model(ema=True)

    db_train = LA_heart(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    db_prob_train = LA_heart(base_dir=train_prob_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.clean_labeled_num))
    LQ_labeled_idxs = list(range(args.clean_labeled_num, args.total_sample))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, LQ_labeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader_prob = DataLoader(db_prob_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model1.train()
    ema_model2.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)
    focal_loss = losses.FocalLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, (sampled_batch, sampled_prob_batch) in enumerate(zip(trainloader, trainloader_prob)):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            LQ_labeled_volume_batch = volume_batch[args.labeled_bs:]
            LQ_label_batch = label_batch[args.labeled_bs:]
            HQ_labeled_volume_batch = volume_batch[:args.labeled_bs]
            HQ_label_batch = label_batch[:args.labeled_bs]

            noise = torch.clamp(torch.randn_like(LQ_labeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = LQ_labeled_volume_batch + noise 
           
            #prob
            volume_prob_batch, label_prob_batch = sampled_prob_batch['image'], sampled_prob_batch['label']
            volume_prob_batch, label_prob_batch = volume_prob_batch.cuda(), label_prob_batch.cuda()

            LQ_labeled_volume_prob_batch = volume_prob_batch[args.labeled_bs:]
            LQ_label_prob_batch = label_prob_batch[args.labeled_bs:]
            HQ_labeled_volume_prob_batch = volume_prob_batch[:args.labeled_bs]
            HQ_label_prob_batch_batch = label_prob_batch[:args.labeled_bs]

            noise = torch.clamp(torch.randn_like(LQ_labeled_volume_prob_batch) * 0.1, -0.2, 0.2)
            ema_inputs_prob = LQ_labeled_volume_prob_batch + noise
            
            with torch.no_grad():
                ema_output1 = ema_model1(ema_inputs)['out']
                ema_output1_soft = torch.softmax(ema_output1, dim=1)
                ema_output2 = ema_model2(ema_inputs)['out']
                ema_output2_soft = torch.softmax(ema_output2, dim=1)  # ([2, 2, 128, 128, 128]) (b, n, d, w, h)
                all_outputs_noise_soft = torch.stack([ema_output1_soft, ema_output2_soft], dim=0) # ([3, 2, 2, 128, 128, 128])
                all_outputs_mean_noise_soft = torch.mean(all_outputs_noise_soft, dim=0) # ([2, 2, 128, 128, 128]) (b, d, w, h)
                all_outputs = torch.argmax(all_outputs_mean_noise_soft, dim=1)

                all_out = torch.stack([ema_output1, ema_output2], dim=0)
                all_out_one = torch.mean(all_out, dim=0)

                all_out1 = torch.stack([ema_output1_soft, ema_output2_soft], dim=0)
                all_out_one1 = torch.mean(all_out1, dim=0)

                #prob
                ema_output1_prob = ema_model1(ema_inputs_prob)['out']                
                ema_output1_prob_soft = torch.softmax(ema_output1_prob, dim=1)
                ema_output2_prob = ema_model2(ema_inputs_prob)['out']                
                ema_output2_prob_soft = torch.softmax(ema_output2_prob, dim=1)
                all_outputs_noise_prob_soft = torch.stack([ema_output1_prob_soft, ema_output2_prob_soft], dim=0) # ([3, 2, 2, 128, 128, 128])
                all_outputs_mean_noise_prob_soft = torch.mean(all_outputs_noise_prob_soft, dim=0) # ([2, 2, 128, 128, 128]) (b, d, w, h)
                all_outputs_prob = torch.argmax(all_outputs_mean_noise_prob_soft, dim=1)


            r_vat = VAT1.vat(ema_model1, ema_model2, LQ_labeled_volume_batch, all_outputs_mean_noise_soft)

            outputs_r = model(LQ_labeled_volume_batch+r_vat)['out']
            outputs_r_soft = torch.softmax(outputs_r, dim=1)
            outputs_r_onehot = torch.argmax(outputs_r_soft, dim=1)

            outputs = model(HQ_labeled_volume_batch)['out']
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_onehot = torch.argmax(outputs_soft, dim=1)

            # contrastive loss
            contrastive_loss = NegativeSamplingPixelContrastiveLoss(sample_num=240,
                                                                     bidirectional=True, temperature=0.1)
            project_HQ_negative = TensorBuffer(buffer_size=4, concat_dim=0)
            project_LQ_negative = TensorBuffer(buffer_size=4, concat_dim=0)
            map_HQ_negative = TensorBuffer(buffer_size=4, concat_dim=0)
            map_LQ_negative = TensorBuffer(buffer_size=4, concat_dim=0)
##
            project_HQ_t1 = model(HQ_labeled_volume_batch)['project']            
            project_HQ_t2 = model(HQ_labeled_volume_prob_batch)['project']
            project_LQ_t1 = model(LQ_labeled_volume_batch)['project']
            project_LQ_t2 = model(LQ_labeled_volume_prob_batch)['project']
            
            map_HQ_positive = model(HQ_labeled_volume_batch)['project_map']
            map_LQ_positive = model(LQ_labeled_volume_batch)['project_map']
            map_HQ_negative.update(map_LQ_positive)
            map_LQ_negative.update(map_HQ_positive)

            project_HQ_negative.update(project_LQ_t1)
            project_LQ_negative.update(project_HQ_t1)


            contrastive_HQ_loss = torch.utils.checkpoint.checkpoint(
                contrastive_loss, project_HQ_t1, project_HQ_t2, project_HQ_negative.values, map_HQ_positive, map_HQ_negative.values)
            contrastive_LQ_loss = torch.utils.checkpoint.checkpoint(
                contrastive_loss, project_LQ_t1, project_LQ_t2, project_LQ_negative.values, map_LQ_positive, map_LQ_negative.values)
            contrastive_loss = contrastive_HQ_loss + contrastive_LQ_loss

            # S-Err Lts
            
            outputs_onehot_np = outputs_r_onehot.cpu().detach().numpy()
            ema_output_soft_np = all_outputs_mean_noise_soft.cpu().detach().numpy()
            ema_output_soft_np_accumulated = np.swapaxes(ema_output_soft_np, 1, 2)
            ema_output_soft_np_accumulated = np.swapaxes(ema_output_soft_np_accumulated, 2, 3)
            ema_output_soft_np_accumulated = np.swapaxes(ema_output_soft_np_accumulated, 3, 4)
            ema_output_soft_np_accumulated = ema_output_soft_np_accumulated.reshape(-1, num_classes)
            ema_output_soft_np_accumulated = np.ascontiguousarray(ema_output_soft_np_accumulated)

            outputs_onehot_np_accumulated = outputs_onehot_np.reshape(-1).astype(np.uint8)
			
            assert outputs_onehot_np_accumulated.shape[0] == ema_output_soft_np_accumulated.shape[0]

            CL_type = args.CL_type
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = ce_loss(outputs_r, all_outputs)

            try:
                #print('Potential Noise Num:', cleanlab.count.num_label_issues(outputs_onehot_np_accumulated, ema_output_soft_np_accumulated))                
                if CL_type in ['both']:
                    noise = cleanlab.filter.find_label_issues(outputs_onehot_np_accumulated, ema_output_soft_np_accumulated, filter_by='both', n_jobs=1)
                elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                    noise = cleanlab.filter.find_label_issues(outputs_onehot_np_accumulated, ema_output_soft_np_accumulated, filter_by=CL_type, n_jobs=1)            
              
                confident_maps_np = noise.reshape(-1, patch_size[0], patch_size[1], patch_size[2]).astype(np.uint8)
                confident_maps = torch.from_numpy(confident_maps_np).cuda(outputs_r_soft.device.index)

                mask = (confident_maps == 1).float() 
                mask = torch.unsqueeze(mask, 1) 
                mask = torch.cat((mask, mask), 1)

                consistency_loss1 = torch.sum(mask*consistency_dist)/(torch.sum(mask)+1e-16)
                consistency_loss2 = ce_loss(outputs_r, all_outputs_prob)
                consistency_loss3 = ce_loss(all_out_one, all_outputs_prob)
                consistency_loss = consistency_loss1 + consistency_loss2 + consistency_loss3

            except Exception as e:
                print('fail to identify errors in this batch; change to typical CR loss')
                consistency_loss = torch.mean((outputs_soft[args.labeled_bs:] - noisy_ema_output_soft) ** 2)



            # supervised loss
            loss_ce = ce_loss(outputs, label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(outputs_soft, label_batch[:args.labeled_bs].unsqueeze(1))
            
            # focal loss
            loss_focal = focal_loss(outputs, label_batch[:args.labeled_bs][:].long())

            supervised_loss = 0.5 * (loss_dice + loss_ce) + loss_focal 
            
            # consistency loss
            #consistency_weight = get_current_consistency_weight(iter_num//150)
            #consistency_loss = torch.mean((outputs_r_soft - all_outputs_mean_noise_soft) ** 2)
            #consistency_loss = ce_loss(outputs_r, all_outputs)
          			
            noisy_label_batch = LQ_label_batch
            with torch.no_grad():            
                ema_output1_no_noise = ema_model1(LQ_labeled_volume_batch)['out']
                ema_output1_soft_no_noise = torch.softmax(ema_output1_no_noise, dim=1)
                ema_output2_no_noise = ema_model2(LQ_labeled_volume_batch)['out']
                ema_output2_soft_no_noise = torch.softmax(ema_output2_no_noise, dim=1)
                outputs1 = model(LQ_labeled_volume_batch)['out']
                outputs1_soft = torch.softmax(outputs1, dim=1)
                all_outputs_no_noise_soft = torch.stack([outputs1_soft, ema_output1_soft, ema_output2_soft], dim=0) 
                all_outputs_mean_no_noise_soft = torch.mean(all_outputs_no_noise_soft, dim=0)# (b, d, w, h)

            # 1: tensor to npy()
            masks_np = noisy_label_batch.cpu().detach().numpy()
            ema_output_soft_np = all_outputs_mean_no_noise_soft.cpu().detach().numpy()
            #print("ema_output_soft_np shape", ema_output_soft_np.shape)

            # 2: identify the noise map
            ema_output_soft_np_accumulated_0 = np.swapaxes(ema_output_soft_np, 1, 2)
            ema_output_soft_np_accumulated_1 = np.swapaxes(ema_output_soft_np_accumulated_0, 2, 3)
            ema_output_soft_np_accumulated_2 = np.swapaxes(ema_output_soft_np_accumulated_1, 3, 4)
            #print("ema_output_soft_np_accumulated_2", ema_output_soft_np_accumulated_2)
            ema_output_soft_np_accumulated_3 = ema_output_soft_np_accumulated_2.reshape(-1, num_classes)
            ema_output_soft_np_accumulated = np.ascontiguousarray(ema_output_soft_np_accumulated_3)
            masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)
            #print("masks_np_accumulated.shape", masks_np_accumulated.shape)	
            #print("ema_output_soft_np_accumulated.shape", ema_output_soft_np_accumulated.shape)		
            assert masks_np_accumulated.shape[0] == ema_output_soft_np_accumulated.shape[0]
            #print("ema_output_soft_np_accumulated shape", ema_output_soft_np_accumulated.shape)
			
            
            try:
                # For cleanlab v1.0
                # if CL_type in ['both']:
                    # noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, ema_output_soft_np_accumulated, prune_method='both', n_jobs=1)
                # elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                    # noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, ema_output_soft_np_accumulated, prune_method=CL_type, n_jobs=1)
                # For cleanlab v2.0
                if CL_type in ['both']:
                    noise = cleanlab.filter.find_label_issues(masks_np_accumulated, ema_output_soft_np_accumulated, filter_by='both', n_jobs=1)
                elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                    noise = cleanlab.filter.find_label_issues(masks_np_accumulated, ema_output_soft_np_accumulated, filter_by=CL_type, n_jobs=1)

                confident_maps_np = noise.reshape(-1, patch_size[0], patch_size[1], patch_size[2]).astype(np.uint8)
                
                # Correct the LQ label for our focused binary task
                correct_type = args.refine_type
                if correct_type == 'smooth':
                    smooth_arg = 0.8
                    corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                    print('Smoothly correct the noisy label')
                elif correct_type == 'uncertainty_smooth':
                    T = 6
                    _, _, d, w, h = LQ_labeled_volume_batch.shape
                    volume_batch_r = LQ_labeled_volume_batch.repeat(2, 1, 1, 1, 1)
                    stride = volume_batch_r.shape[0] // 2
                    preds0 = torch.zeros([stride * T, 2, d, w, h]).cuda()
                    preds1 = torch.zeros([stride * T, 2, d, w, h]).cuda()
                    preds2 = torch.zeros([stride * T, 2, d, w, h]).cuda()
                    for i in range(T//2):
                        ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                        with torch.no_grad():
                            preds0[2 * stride * i:2 * stride *(i + 1)] = model(ema_inputs)['out']
                            preds1[2 * stride * i:2 * stride *(i + 1)] = ema_model1(ema_inputs)['out']
                            preds2[2 * stride * i:2 * stride *(i + 1)] = ema_model2(ema_inputs)['out']
                    
                    preds0 = torch.softmax(preds0, dim=1)
                    preds1 = torch.softmax(preds1, dim=1)
                    preds2 = torch.softmax(preds2, dim=1)

                    preds_soft = torch.stack([preds0, preds1, preds2], dim=0) 
                    preds_mean = torch.mean(preds_soft, dim=0)

                    preds = preds_mean.reshape(T, stride, 2, d, w, h)
                    preds = preds.float()
                    preds = torch.mean(preds, dim=0)
                    uncertainty = -1.0 * torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
                    uncertainty = uncertainty/math.log(2) # normalize uncertainty to 0 to 1, cuz ln2 is the max value
                    uncertainty_np = uncertainty.cpu().detach().numpy()
                    uncertainty_np_squeeze = np.squeeze(uncertainty_np)
                    smooth_arg = 1 - uncertainty_np_squeeze
                    #print(smooth_arg.shape)
                    corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                    print('Uncertainty-based smoothly correct the noisy label')
                else:
                    corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
                    print('Hard correct the noisy label')
                
                noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(outputs_r_soft.device.index)

                # noisy supervised loss

                loss_ce_weak = ce_loss(outputs_r, noisy_label_batch.long())
                loss_dice_weak = dice_loss(outputs_r_soft, noisy_label_batch.long().unsqueeze(1))
                

                if args.use_entropymin:
                    weak_supervised_loss = 0.5 * (loss_dice_weak + loss_ce_weak) + losses.entropy_loss(outputs_r_soft, C=2)
                else:
                    weak_supervised_loss = 0.5 * (loss_dice_weak + loss_ce_weak)
            
            except Exception as e:
                print('Cannot identify noises')
                if args.use_entropymin:
                    weak_supervised_loss = weak_supervised_loss + losses.entropy_loss(outputs_soft, C=2)
                else:
                    weak_supervised_loss = weak_supervised_loss
                
            # total loss
            t = tau.sigmoid_rampup(epoch_num, 100)
            if args.use_consistency:
                loss = supervised_loss + 0.1 * t * contrastive_loss + consistency_weight * (consistency_loss + args.weak_weight * weak_supervised_loss)
            else:
                loss = supervised_loss + consistency_weight * (args.weak_weight * weak_supervised_loss)
            
            optimizer.zero_grad()
            loss.backward()
            #loss.backward(retain_graph=True)
            optimizer.step()
            update_ema_variables(model, ema_model1, ema_model2, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('info/contrastive_loss',
                              contrastive_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, consistency_loss: %f,consistency_loss1: %f,consistency_loss2: %f, consistency_loss3: %f, weak_supervised_loss: %f, contrastive_loss: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), consistency_loss,consistency_loss1,consistency_loss2,consistency_loss3, weak_supervised_loss, contrastive_loss))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 1000 and iter_num % 50 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.val_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size, stride_xy=18, stride_z=4)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)


                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/home/xsq/xsq/TSS-CL/dwh-result/TLF/model_TLF_70_30_rcps_try/{}_{}/{}".format(
        args.exp, args.clean_labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
