import os
import argparse
import sys
import copy
import torch
from tqdm import tqdm
import open3d as o3d
import numpy as np
from utils import *
from models import *
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from losses import *
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn as nn
import math

def train_one_epoch_seq_action(args, net, train_loader, opt):
    total_loss = 0
    num_examples = 0
    net.train()
    seq_len = train_loader.dataset.mini_clip_len
    loss_items = copy.deepcopy(loss_dict[args.model])

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # use sequence data in order
        iter_loss = 0
        num_examples += args.batch_size
        pc = []
        action_label = None
        for j in range(0, seq_len):
            ## reading data from dataloader and transform their format
            pc1, pc2, ft1, ft2, _, a_label, _, _ = extract_data_info_clip2(data, j)
            action_label = a_label

            batch_size = pc1.size(0)

            pc.append(pc1)

        action_label = torch.squeeze(action_label)
        action_label = action_label[:,0]
        action_label = action_label.long().cuda().contiguous()
        
        pc = torch.stack(pc)
        pc = pc.transpose(0,1)
        pc = pc.transpose(2,3)
        
        score = net(pc)
        L = nn.CrossEntropyLoss()
        loss_a = L(score, action_label)
        opt.zero_grad()
        loss_a.backward()
        opt.step()
        items = {
            'Loss': loss_a.item(),
        }

        total_loss += loss_a.item() * batch_size
        for l in loss_items:
            loss_items[l].append(items[l])

    total_loss = total_loss / num_examples
    for l in loss_items:
        loss_items[l] = np.mean(np.array(loss_items[l]))

    return total_loss, loss_items

def evaluate(args, net, data_loader, textio):
    net.eval()
    num_examples = 0
    acc = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            num_examples += args.batch_size
            pc = []
            action_label = None
            for j in range(0, 5):
                ## reading data from dataloader and transform their format
                pc1, pc2, ft1, ft2, _, a_label, _, _ = extract_data_info_clip2(data, j)
                batch_size = pc1.shape[0]
                action_label = a_label
                    
                pc1 = pc1.detach()
                pc.append(pc1)

            action_label = torch.squeeze(action_label)
            action_label = action_label[:,0]
            action_label = action_label.long().cuda().contiguous()
            
            pc = torch.stack(pc)
            pc = pc.transpose(0,1)
            pc = pc.transpose(2,3)
            
            output = net(pc)

            acc1, acc5 = accuracy(output, action_label, topk=(1, 5))
            acc += acc1 * batch_size

    acc = acc / num_examples
    #textio.cprint(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}'.format(top1=acc1, top5=acc5))

    #textio.cprint(' * Video Acc@1 %f'%total_acc)

    return acc

def train_one_epoch_seq_2(args, net, train_loader, opt, pre_net):
    total_loss = 0
    num_examples = 0
    mode = 'train'
    net.train()
    pre_net.eval()
    loss_items = copy.deepcopy(loss_dict[args.model])
    seq_len = train_loader.dataset.mini_clip_len

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # use sequence data in order
        num_examples += args.batch_size
        pc = []
        flow = []
        action_label = None
        for j in range(0, seq_len):
            ## reading data from dataloader and transform their format
            pc1, pc2, ft1, ft2, _, a_label, _, _ = extract_data_info_clip2(data, j)
            action_label = a_label

            batch_size = pc1.size(0)

            with torch.no_grad():
                if j == 0:
                    pred_f, gfeat = pre_net(pc1, pc2, ft1, ft2, None)
                else:
                    pred_f, gfeat = pre_net(pc1, pc2, ft1, ft2, gfeat)
                
            flow.append(pred_f)
            pc.append(pc1)

        action_label = torch.squeeze(action_label)
        action_label = action_label[:,0]
        action_label = action_label.long().cuda().contiguous()
        
        score = net(pc, flow)
        L = nn.CrossEntropyLoss()
        loss_a = L(score, action_label)
        opt.zero_grad()
        loss_a.backward()
        opt.step()
        
        items = {
            'Loss': loss_a.item(),
        }

        total_loss += loss_a.item() * batch_size
        for l in loss_items:
            loss_items[l].append(items[l])

    total_loss = total_loss / num_examples
    for l in loss_items:
        loss_items[l] = np.mean(np.array(loss_items[l]))

    return total_loss, loss_items

def val_one_epoch_seq2(args, net, pre_net, test_loader, textio):
    net.eval()

    seq_len = test_loader.dataset.mini_clip_len
    batch_size = test_loader.batch_size
    num_examples = 0
    acc = 0
    with torch.no_grad():
        # read sequence data
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            # use sequence data in order
            pc = []
            flow = []
            action_label = None
            num_examples += batch_size
            
            for j in range(0, seq_len):
                ## reading data from dataloader and transform their format
                pc1, pc2, ft1, ft2, _, a_label, _, _ = extract_data_info_clip2(data, j)
                batch_size = pc1.shape[0]
                action_label = a_label

                if j == 0:
                    pred_f, gfeat  = pre_net(pc1, pc2, ft1, ft2, None)
                else:
                    pred_f, gfeat  = pre_net(pc1, pc2, ft1, ft2, gfeat)
                    
                flow.append(pred_f)
                pc.append(pc1)

            action_label = torch.squeeze(action_label)
            action_label = action_label[:,0]
            action_label = action_label.long().cuda().contiguous()
            
            score = net(pc,flow)
            acc1, acc5 = accuracy(score, action_label, topk=(1, 5))
            #acc1 = show_topk(1, score, action_label.unsqueeze(0))
            #acc5 = show_topk(5, score, action_label)
            acc += acc1 * batch_size

        acc = acc / num_examples
        
    return acc

def test_one_epoch_seq2(args, net, pre_net, test_loader, textio):
    net.eval()

    seq_len = test_loader.dataset.mini_clip_len
    batch_size = test_loader.batch_size
    num_examples = 0
    acc = 0
    y_pred = []
    y_true = []
    
    sf_metric = {'epe': 0, 'accs': 0, 'accr': 0}
    epe_xyz = {'x': [], 'y': [], 'z': []}
    
    with torch.no_grad():
        # read sequence data
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            # use sequence data in order
            pc = []
            flow = []
            action_label = None
            num_examples += batch_size
            
            for j in range(0, seq_len):
                ## reading data from dataloader and transform their format
                pc1, pc2, ft1, ft2, gt, a_label, _, _ = extract_data_info_clip2(data, j)
                batch_size = pc1.shape[0]
                action_label = a_label

                if j == 0:
                    pred_f, gfeat  = pre_net(pc1, pc2, ft1, ft2, None)
                else:
                    pred_f, gfeat  = pre_net(pc1, pc2, ft1, ft2, gfeat)
                    
                batch_res = eval_scene_flow(pc1, pred_f.transpose(2, 1).contiguous(), gt, args)
                for metric in sf_metric:
                    sf_metric[metric] += batch_res[metric]

                epe_xyz['x'].append(batch_res['epe_x'])
                epe_xyz['y'].append(batch_res['epe_y'])
                epe_xyz['z'].append(batch_res['epe_z'])
                    
                flow.append(pred_f)
                pc.append(pc1)

            action_label = torch.squeeze(action_label)
            action_label = action_label[0]
            action_label = action_label.long().cuda().contiguous()
            
            score = net(pc,flow)
            acc1, acc5 = accuracy(score, action_label.unsqueeze(0), topk=(1, 5))
            #acc1 = show_topk(1, score, action_label.unsqueeze(0))
            #acc5 = show_topk(5, score, action_label)
            acc += acc1
            
            output = (torch.max(torch.exp(score), 1)[1]).data.cpu()
            y_pred.append(int(output)) # Save Prediction
            labels = action_label.data.cpu()
            y_true.append(int(labels)) # Save Truth
            

        print(acc)
        acc = acc / num_examples
        print(num_examples)
        
        for metric in sf_metric:
            sf_metric[metric] = sf_metric[metric] / num_examples
        
    return acc, y_pred, y_true, sf_metric

def train_one_epoch_seq(args, net, train_loader, opt):
    total_loss = 0
    num_examples = 0
    mode = 'train'
    net.train()
    loss_items = copy.deepcopy(loss_dict[args.model])
    seq_len = train_loader.dataset.mini_clip_len

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # use sequence data in order
        iter_loss = 0
        iter_items = copy.deepcopy(loss_dict[args.model])
        num_examples += args.batch_size
        
        for j in range(0, seq_len):
            ## reading data from dataloader and transform their format
            pc1, pc2, ft1, ft2, flow_label = extract_data_info_clip(data, j)

            batch_size = pc1.size(0)
            opt.zero_grad()

            if args.model in ['mmflow']:
                # forward and loss computation
                if j == 0:
                    pred_f, gfeat = net(pc1, pc2, ft1, ft2, None)
                else:
                    gfeat = gfeat.detach()
                    pred_f, gfeat = net(pc1, pc2, ft1, ft2, gfeat)
                loss, items = mmflowLoss(pred_f.transpose(2, 1), flow_label, pc1.transpose(2, 1))

            loss.backward()
            opt.step()

            iter_loss += loss
            for k in iter_items:
                iter_items[k].append(items[k])

        iter_loss = iter_loss / seq_len
        for l in iter_items:
            loss_items[l].append(np.mean(np.array(iter_items[l])))
        total_loss += iter_loss.item() * batch_size

    total_loss = total_loss / num_examples
    for l in loss_items:
        loss_items[l] = np.mean(np.array(loss_items[l]))

    return total_loss, loss_items


def extract_data_info_clip2(seq_data, idx):
    pc1, pc2, ft1, ft2, gt, action_label, pos_label, skeleton = seq_data
    pc1 = pc1[:, idx].cuda().transpose(2, 1).contiguous()
    pc2 = pc2[:, idx].cuda().transpose(2, 1).contiguous()
    ft1 = ft1[:, idx].cuda().transpose(2, 1).contiguous()
    ft2 = ft2[:, idx].cuda().transpose(2, 1).contiguous()
    gt = gt[:, idx].cuda().transpose(2, 1).contiguous()
    pos_label = pos_label[:, idx].cuda().transpose(2, 1).contiguous().long()
    skeleton = skeleton[:, idx].cuda().transpose(2, 1).float()

    return pc1, pc2, ft1, ft2, gt, action_label, pos_label, skeleton

def extract_data_info_clip(seq_data, idx):
    pc1, pc2, ft1, ft2, gt = seq_data
    pc1 = pc1[:, idx].cuda().transpose(2, 1).contiguous()
    pc2 = pc2[:, idx].cuda().transpose(2, 1).contiguous()
    ft1 = ft1[:, idx].cuda().transpose(2, 1).contiguous()
    ft2 = ft2[:, idx].cuda().transpose(2, 1).contiguous()
    gt = gt[:, idx].cuda().contiguous()
    # pos_label = pos_label[:, idx].cuda().contiguous().long()
    # skeleton = skeleton[:, idx].cuda().transpose(2, 1).float()

    return pc1, pc2, ft1, ft2, gt

def extract_data_info_test(data):
    pc1, pc2, ft1, ft2, gt = data
    pc1 = pc1.cuda().transpose(2, 1).contiguous()
    pc2 = pc2.cuda().transpose(2, 1).contiguous()
    ft1 = ft1.cuda().transpose(2, 1).contiguous()
    ft2 = ft2.cuda().transpose(2, 1).contiguous()
    gt = gt.cuda().contiguous()

    return pc1, pc2, ft1, ft2, gt


def eval_one_epoch_seq(args, net, eval_loader, textio):
    num_pcs = 0

    sf_metric = { 'epe': 0, 'accs': 0, 'accr': 0}

    epe_xyz = {'x': [], 'y': [], 'z': []}

    seq_len = eval_loader.dataset.mini_clip_len
    batch_size = eval_loader.batch_size

    # start point for inference
    # start_point = time.time()

    with torch.no_grad():
        # read sequence data
        for i, data in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            # use sequence data in order
            for j in range(0, seq_len):
                ## reading data from dataloader and transform their format
                pc1, pc2, ft1, ft2, flow_label = extract_data_info_clip(data, j)
                batch_size = pc1.shape[0]

                if args.model in ['mmflow']:
                    if j == 0:
                        pred_f, gfeat  = net(pc1, pc2, ft1, ft2, None)
                    else:
                        pred_f, gfeat  = net(pc1, pc2, ft1, ft2, gfeat)

                batch_res = eval_scene_flow(pc1.transpose(2, 1).contiguous(), pred_f.transpose(2, 1).contiguous(), flow_label, args)
                for metric in sf_metric:
                    sf_metric[metric] += batch_size * batch_res[metric]

                epe_xyz['x'].append(batch_res['epe_x'])
                epe_xyz['y'].append(batch_res['epe_y'])
                epe_xyz['z'].append(batch_res['epe_z'])

                num_pcs += batch_size

    # end point for inference
    # infer_time = time.time() - start_point

    for metric in sf_metric:
        sf_metric[metric] = sf_metric[metric] / num_pcs

    #textio.cprint('###The inference speed is %.3fms per frame###' % (infer_time * 1000 / num_pcs))
    return sf_metric, epe_xyz


def test_one_epoch_seq(args, net, test_loader, textio):
    if not args.model in ['gl_wo', 'icp', 'arfnet_o']:
        net.eval()

    if args.save_res:
        args.save_res_path = 'checkpoints/' + args.exp_name + "/results/"
        num_seq = 0
        clip_info = args.clips_info[num_seq]
        seq_res_path = os.path.join(args.save_res_path, clip_info['clip_name'])
        if not os.path.exists(seq_res_path):
            os.makedirs(seq_res_path)

    num_pcs = 0

    sf_metric = {'epe': 0, 'accs': 0, 'accr': 0}
    epe_xyz = {'x': [], 'y': [], 'z': []}

    gt_trans_all = torch.zeros((len(test_loader), 4, 4)).cuda()
    pre_trans_all = torch.zeros((len(test_loader), 4, 4)).cuda()

    # start point for inference
    #start_point = time.time()

    with torch.no_grad():
        clips_info = test_loader.dataset.clips_info
        clips_name = []
        clips_st_index = []
        # extract clip info
        for i in range(len(clips_info)):
            clips_name.append(clips_info[i]['clip_name'])
            clips_st_index.append(clips_info[i]['index'][0])
        # read data in order
        num_clip = 0
        seq_len = test_loader.dataset.update_len
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):

            ## reading data from dataloader and transform their format
            pc1, pc2, ft1, ft2, gt = extract_data_info_test(data)

            if args.model in ['mmflow_t']:
                # if i==clips_st_index[num_clip]:
                if i == clips_st_index[num_clip] or i % seq_len == 0:
                    pred_f, gfeat = net(pc1, pc2, ft1, ft2, None)
                    if num_clip < (len(clips_name) - 1):
                        num_clip += 1
                else:
                    pred_f, gfeat = net(pc1, pc2, ft1, ft2, gfeat)
                
            if args.model == 'cmflow_bpt':
                if i == clips_st_index[num_clip] or i % seq_len == 0:
                    pred_f, gfeat, trans = net(pc1, pc2, ft1, ft2, None)
                    if num_clip < (len(clips_name) - 1):
                        num_clip += 1
                else:
                    pred_f, gfeat, trans = net(pc1, pc2, ft1, ft2, gfeat)

            if args.save_res:
                res = {
                    'pc1': pc1[0].cpu().numpy().tolist(),
                    'pc2': pc2[0].cpu().numpy().tolist(),
                    'pred_f': pred_f[0].cpu().detach().numpy().tolist(),
                    'gt_f': gt[0].transpose(0, 1).contiguous().cpu().detach().numpy().tolist(),
                }

                if num_pcs < clip_info['index'][1]:
                    res_path = os.path.join(seq_res_path, '{}.json'.format(num_pcs))
                else:
                    num_seq += 1
                    clip_info = args.clips_info[num_seq]
                    seq_res_path = os.path.join(args.save_res_path, clip_info['clip_name'])
                    if not os.path.exists(seq_res_path):
                        os.makedirs(seq_res_path)
                    res_path = os.path.join(seq_res_path, '{}.json'.format(num_pcs))

                ujson.dump(res, open(res_path, "w"))

            ## evaluate the estimated results using ground truth
            batch_res = eval_scene_flow(pc1, pred_f.transpose(2, 1).contiguous(), gt, args)
            for metric in sf_metric:
                sf_metric[metric] += batch_res[metric]

            epe_xyz['x'].append(batch_res['epe_x'])
            epe_xyz['y'].append(batch_res['epe_y'])
            epe_xyz['z'].append(batch_res['epe_z'])

            num_pcs += 1

    # end point for inference
    #infer_time = time.time() - start_point

    for metric in sf_metric:
        sf_metric[metric] = (sf_metric[metric] / num_pcs)

    #textio.cprint('###The inference speed is %.3fms per frame###' % (infer_time * 1000 / num_pcs))
    return sf_metric, epe_xyz