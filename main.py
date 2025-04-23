#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from utils import *
from dataset import *
import numpy as np
import open3d as o3d
from losses import *
from matplotlib import pyplot as plt
from clip_util import *
from sklearn.metrics import confusion_matrix, jaccard_score
import pandas as pd

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def plot_epe_component(epe_xyz, args):

    epe_x = np.array(epe_xyz['x'])
    epe_y = np.array(epe_xyz['y'])
    epe_z = np.array(epe_xyz['z'])
    epe = np.sqrt(epe_x**2 + epe_y**2 + epe_z**2)

    plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(15,5))

    ax1 = ax[0,0]
    data_len = epe_x.shape[0]
    avg = np.mean(epe_x)
    std = np.std(epe_x)
    r1 = avg - std
    r2 = avg + std
    ax1.plot(epe_x, c='m',linestyle='-')
    ax1.plot([0, data_len], [avg, avg], c = 'b', label='Mean')
    ax1.fill_between(np.arange(1, data_len+1, 1), np.tile(r1,data_len), np.tile(r2,data_len), color = 'b', alpha=0.25)
    ax1.set_title('EPE_X (m)')
    ax1.set_ylim(0,1)
    ax1.legend(loc='upper right')

    ax2 = ax[0,1]
    avg = np.mean(epe_y)
    std = np.std(epe_y)
    r1 = avg - std
    r2 = avg + std
    ax2.plot(epe_y, c='m',linestyle='-')
    ax2.plot([0, data_len], [avg, avg], c = 'b', label='Mean')
    ax2.fill_between(np.arange(1, data_len+1, 1), np.tile(r1,data_len), np.tile(r2,data_len), color = 'b', alpha=0.25)
    ax2.set_title('EPE_Y (m)')
    ax2.set_ylim(0,1)
    ax2.legend(loc='upper right')

    ax3 = ax[1,0]
    avg = np.mean(epe_z)
    std = np.std(epe_z)
    r1 = avg - std
    r2 = avg + std
    ax3.plot(epe_z, c='m',linestyle='-')
    ax3.plot([0, data_len], [avg, avg], c = 'b', label='Mean')
    ax3.fill_between(np.arange(1, data_len+1, 1), np.tile(r1,data_len), np.tile(r2,data_len), color = 'b', alpha=0.25)
    ax3.set_title('EPE_Z (m)')
    ax3.set_ylim(0,0.1)
    ax3.legend(loc='upper right')

    ax4 = ax[1,1]
    avg = np.mean(epe)
    std = np.std(epe)
    r1 = avg - std
    r2 = avg + std
    ax4.plot(epe, c='m',linestyle='-')
    ax4.plot([0, data_len], [avg, avg], c = 'b', label='Mean')
    ax4.fill_between(np.arange(1, data_len+1, 1), np.tile(r1,data_len), np.tile(r2,data_len), color = 'b', alpha=0.25)
    ax4.set_title('EPE (m)')
    ax4.set_ylim(0,1)
    ax4.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig('checkpoints/' + args.exp_name + '/' + 'epe_xyz.png', dpi=500)
    print('----save epe components figure----')

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'loss_train'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'loss_train')

def test(args, net, test_loader, textio, pre_net=None):

    if args.model == 'mmflow_a':
        #acc1 , y_pred, y_true, sf_metric= test_one_epoch_seq2(args, net, pre_net, test_loader, textio)
        acc1, sf_metric = val_one_epoch_seq3(args, net, pre_net, test_loader, textio)
        textio.cprint('\tTop{}: {:.2f}%'.format(1, 100 * acc1))
        #textio.cprint('\tTop{}: {:.2f}%'.format(5, 100 * acc5))
        #classes = ('arm', 'bow', 'leg and arm', 'leg', 'wave')
        """
        classes = ('squat', 'bow', 'leg and arm', 'leg', 'wave')

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                     columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('checkpoints/%s/confusion_matrix.png' % args.exp_name,dpi=500)
        """
        
        for metric in sf_metric:
            textio.cprint('###The mean {}: {}###'.format(metric, sf_metric[metric]))
        
    elif args.model in ['pstnet', 'harnet', 'harpointgnn']:
        acc1, y_pred, y_true = test_pst(args, net, test_loader, textio)
        textio.cprint('\tTop{}: {:.2f}%'.format(1, 100 * acc1))
        classes = ('arm', 'bow', 'leg and arm', 'leg', 'wave')

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                     columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('checkpoints/%s/confusion_matrix.png' % args.exp_name,dpi=500)
    
    elif args.model == 'mmflow_p':
        #acc1 , y_pred, y_true, sf_metric= test_one_epoch_seq_p1(args, net, pre_net, test_loader, textio)
        pred_true, sf_metric, parsing_metric= test_one_epoch_p(args, net, pre_net, test_loader, textio)
        for metric in parsing_metric:
            textio.cprint('###The mean {}: {:.2f}%###'.format(metric, 100 * parsing_metric[metric]))
        classes = ('r_arm', 'l_arm', 'r_leg', 'l_leg', 'body')
        
        # Build confusion matrix
        y_true = pred_true['true']
        y_pred = pred_true['pred']
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                     columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('checkpoints/%s/confusion_matrix.png' % args.exp_name,dpi=500)
        
    elif args.model == 'mmflow_bpt':
        sf_metric, epe_xyz = test_one_epoch_seq_track(args, net, test_loader, textio)
        for metric in sf_metric:
            textio.cprint('###The mean {}: {}###'.format(metric, sf_metric[metric]))
        plot_epe_component(epe_xyz,args)
    else:
        if args.dataset == 'ClipDataset':
            sf_metric, epe_xyz = test_one_epoch_seq(args, net, test_loader, textio)
        else:
            sf_metric, epe_xyz = eval_one_epoch(args, net, test_loader, textio)
        ## print scene flow evaluation results
        for metric in sf_metric:
            textio.cprint('###The mean {}: {}###'.format(metric, sf_metric[metric]))

        #plot_epe_component(epe_xyz,args)
        #textio.cprint('###The mean {}: ###'.format(epe_xyz))

    
def test_vis(args, net, test_loader, textio):

    if not args.model in ['gl_wo','icp']:
        net.eval()
    
    args.vis_path_2D='checkpoints/'+args.exp_name+"/test_vis_2d/"
    args.vis_path_3D='checkpoints/'+args.exp_name+"/test_vis_3d_input/"
    args.vis_path_seg='checkpoints/'+args.exp_name+"/test_vis_seg/"
    args.vis_path_seg_pse = 'checkpoints/'+args.exp_name+"/test_vis_seg_pse/"
    args.vis_path_reg='checkpoints/'+args.exp_name+"/test_vis_reg/"

    if not os.path.exists(args.vis_path_2D):
        os.makedirs(args.vis_path_2D)
    if not os.path.exists(args.vis_path_3D):
        os.makedirs(args.vis_path_3D)
    if not os.path.exists(args.vis_path_seg):
        os.makedirs(args.vis_path_seg)
    if not os.path.exists(args.vis_path_seg_pse):
        os.makedirs(args.vis_path_seg_pse)
    if not os.path.exists(args.vis_path_reg):
        os.makedirs(args.vis_path_reg)
   

    sf_metric, seg_metric, pose_metric, gt_trans, pre_trans, _ = eval_one_epoch(args, net, test_loader, textio)
    

def train_action(args, net, train_loader, val_loader, textio, pre_net):
    
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(opt, args.decay_epochs, gamma = args.decay_rate)
    
    opt1 = optim.Adam(pre_net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler1 = StepLR(opt1, args.decay_epochs, gamma = args.decay_rate)

    best_val_res = 0
    train_loss_ls = np.zeros((args.epochs))
    val_score_ls = np.zeros(args.epochs)
    train_items_iter = {
                    'Loss': [],'nnLoss': [],'smoothnessLoss': [],'veloLoss': [],
                    'cycleLoss': [],'curvatureLoss':[],'chamferLoss': [],'L2Loss': [], 'glLoss': [],
                    'egoLoss':[], 'maskLoss': [], 'superviseLoss': [], 'opticalLoss': [], 'L1Loss': [],
                    }
    
    for epoch in range(args.epochs):
        
        textio.cprint('====epoch: %d, learning rate: %f===='%(epoch, opt.param_groups[0]['lr']))

        textio.cprint('==starting training on the training set==')

        if args.model == 'mmflow_a':
            #total_loss, loss_items, test_correct = train_one_epoch_seq_2(args, net, train_loader, opt, pre_net)
            total_loss, loss_items, test_correct = train_one_epoch_seq_4(args, net, train_loader, opt, pre_net, opt1)
        elif args.model == 'mmflow_p':
            #total_loss, loss_items, test_correct = train_one_epoch_p(args, net, train_loader, opt, pre_net, opt1) 
            total_loss, loss_items, test_correct = train_one_epoch_seq_p(args, net, train_loader, opt, pre_net)   
        else:
            total_loss, loss_items, test_correct = train_one_epoch_seq_action(args, net, train_loader, opt)

        train_loss_ls[epoch] = total_loss
        for it in loss_items:
            train_items_iter[it].extend([loss_items[it]])
        textio.cprint('mean train loss: %f'%total_loss)
        textio.cprint("mean train acc : {:.2f}%".format(100.0*test_correct))
        
        textio.cprint('==starting evaluation on the validation set==')
        
        if args.model == 'mmflow_a':
            if args.joint_train:
                acc, sf = val_one_epoch_seq3(args, net, pre_net, val_loader, textio)
            else:    
                acc, sf = val_one_epoch_seq2(args, net, pre_net, val_loader, textio)
        elif args.model == 'mmflow_p':
            #acc, sf = val_one_epoch_seq_p(args, net, pre_net, val_loader, textio)
            acc, sf = val_one_epoch_p(args, net, pre_net, val_loader, textio)
        else:
            acc = evaluate(args, net, val_loader, textio)
        
        val_score_ls[epoch] = acc
        textio.cprint("Val Accuracy {:.2f}%".format(100.0*acc))
        textio.cprint('val EPE is : %f'%sf['epe'])
        if best_val_res <= acc:
            best_val_res = acc
            textio.cprint('best val score till now: %f'%best_val_res)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            if args.joint_train:
                if torch.cuda.device_count() > 1:
                    torch.save(pre_net.module.state_dict(), 'checkpoints/%s/models/modelsf.best.t7' % args.exp_name)
                else:
                    torch.save(pre_net.state_dict(), 'checkpoints/%s/models/modelsf.best.t7' % args.exp_name)
                
        if epoch == args.epochs - 1:
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.latest.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.latest.t7' % args.exp_name)
            
            if args.joint_train:
                if torch.cuda.device_count() > 1:
                    torch.save(pre_net.module.state_dict(), 'checkpoints/%s/models/modelsf.latest.t7' % args.exp_name)
                else:
                    torch.save(pre_net.state_dict(), 'checkpoints/%s/models/modelsf.latest.t7' % args.exp_name)

        scheduler1.step()
        scheduler.step()
        plot_loss_epoch(train_items_iter, args, epoch)
        
    textio.cprint('====best val loss after %d epochs: %f===='%(args.epochs, best_val_res))
    plt.clf()
    plt.plot(train_loss_ls[0:int(args.epochs)], 'b')
    plt.legend(['train_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('checkpoints/%s/loss_train/train_loss.png' % args.exp_name,dpi=500)
    
    plt.clf()
    plt.plot(val_score_ls[0:int(args.epochs)], 'r')
    plt.legend(['val_score'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('checkpoints/%s/val_score.png' % args.exp_name,dpi=500)

    return best_val_res
  
def train(args, net, train_loader, val_loader, textio):
    
    
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(opt, args.decay_epochs, gamma = args.decay_rate)

    best_val_res = np.inf
    train_loss_ls = np.zeros((args.epochs))
    val_score_ls = np.zeros(args.epochs)
    train_items_iter = {
                    'Loss': [],'nnLoss': [],'smoothnessLoss': [],'veloLoss': [],
                    'cycleLoss': [],'curvatureLoss':[],'chamferLoss': [],'L2Loss': [], 'glLoss': [],
                    'egoLoss':[], 'maskLoss': [], 'superviseLoss': [], 'opticalLoss': [], 'L1Loss': [],
                    }
    
    for epoch in range(args.epochs):
        
        textio.cprint('====epoch: %d, learning rate: %f===='%(epoch, opt.param_groups[0]['lr']))

        textio.cprint('==starting training on the training set==')

        if args.dataset == 'ClipDataset':
            total_loss, loss_items = train_one_epoch_seq(args, net, train_loader, opt)
        else:
            total_loss, loss_items = train_one_epoch(args, net, train_loader, opt)

        train_loss_ls[epoch] = total_loss
        for it in loss_items:
            train_items_iter[it].extend([loss_items[it]])
        textio.cprint('mean train loss: %f'%total_loss)

        textio.cprint('==starting evaluation on the validation set==')
        if args.dataset == 'nDataset':
            sf_metric, _ = eval_one_epoch(args, net, val_loader, textio)        
        else:
            sf_metric, _ = eval_one_epoch_seq(args, net, val_loader, textio)
        
        eval_score = sf_metric['epe']
        val_score_ls[epoch] = eval_score
        textio.cprint('mean EPE score: %f'%eval_score)

        if best_val_res >= eval_score:
            best_val_res = eval_score
            textio.cprint('best val score till now: %f'%best_val_res)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        if epoch == args.epochs - 1:
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.latest.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.latest.t7' % args.exp_name)

        scheduler.step()
        # plot_loss_epoch(train_items_iter, args, epoch)

    textio.cprint('====best val loss after %d epochs: %f===='%(args.epochs, best_val_res))
    plt.clf()
    plt.plot(train_loss_ls[0:int(args.epochs)], 'b')
    plt.legend(['train_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('checkpoints/%s/loss_train/train_loss.png' % args.exp_name,dpi=500)
    
    plt.clf()
    plt.plot(val_score_ls[0:int(args.epochs)], 'r')
    plt.legend(['val_score'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('checkpoints/%s/val_score.png' % args.exp_name,dpi=500)

    return best_val_res
   

def main(param_name='train', param_value=False):
    
    args = parse_args_from_yaml("/root/milliflow/configs.yaml")
    
    #args[param_name] = param_value
    #args['exp_name'] = args['model'] + '_' + param_name + '_' + str(param_value)

    # CUDA settings
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    # deterministic results
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    
    # init checkpoint records 
    _init_(args)
    textio = IOStream('/root/milliflow/checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    
    # init dataset and dataloader
    if args.eval:
        test_set = dataset_dict[args.dataset](args=args, root = args.dataset_path, partition=args.eval_split,textio=textio)
        test_loader = DataLoader(test_set,num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=False) 
    else:
        train_set = dataset_dict[args.dataset](args=args, root = args.dataset_path, partition=args.train_set,textio=textio)
        val_set = dataset_dict[args.dataset](args=args, root = args.dataset_path, partition=args.val_split, textio=textio)
        train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, num_workers=args.num_workers, batch_size=args.val_batch_size, shuffle=False, drop_last=False)

    if args.eval:
        args.clips_info = test_set.clips_info

    # init the network (load or from scratch)
    net = init_model(args)
    
    if args.eval:
        best_val_res = None
        if args.vis:
            textio.cprint('==Enable Visulization==')
            test_vis(args, net, test_loader,textio)
        else:
            if args.model in ['mmflow_a', 'mmflow_p']:
                test(args, net, test_loader,textio, net)
            else:
                test(args, net, test_loader,textio)
    else:
        if args.model in ['pstnet', 'mmflow_a', 'harnet', 'harpointgnn', 'mmflow_p']:
            if args.model in ['mmflow_a', 'mmflow_p']:               
                best_val_res = train_action(args, net, train_loader, val_loader, textio, net)
            else:
                best_val_res = train_action(args, net, train_loader, val_loader, textio, pre_net = None)
        else :
            best_val_res = train(args, net, train_loader, val_loader,textio)

    textio.cprint('Max memory alocation: {}MB'.format(torch.cuda.max_memory_allocated(device=0)/1e6)) 
    print('FINISH')

    if best_val_res is not None:
        return best_val_res

if __name__ == '__main__':
    main()