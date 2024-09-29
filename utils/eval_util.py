import numpy as np
import torch
from sklearn.metrics import jaccard_score
#from .odometry_util import *


def get_carterian_res(pc, sensor, args):
    ## measure resolution for r/theta/phi
    if sensor == 'radar':  # LRR30
        r_res = args.radar_res['r_res']  # m
        theta_res = args.radar_res['theta_res']  # radian
        phi_res = args.radar_res['phi_res']  # radian

    if sensor == 'lidar':  # HDL-64E
        r_res = 0.04  # m
        theta_res = 0.4 * np.pi / 180  # radian
        phi_res = 0.08 * np.pi / 180  # radian

    res = np.array([r_res, theta_res, phi_res])
    ## x y z
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    ## from xyz to r/theta/phi (range/elevation/azimuth)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arcsin(z / r)
    phi = np.arctan2(y, x)

    ## compute xyz's gradient about r/theta/phi
    grad_x = np.stack((np.cos(phi) * np.cos(theta), -r * np.sin(theta) * np.cos(phi), -r * np.cos(theta) * np.sin(phi)), axis=2)
    grad_y = np.stack((np.sin(phi) * np.cos(theta), -r * np.sin(phi) * np.sin(theta), r * np.cos(theta) * np.cos(phi)), axis=2)
    grad_z = np.stack((np.sin(theta), r * np.cos(theta), np.zeros((np.size(x, 0), np.size(x, 1)))), axis=2)

    ## measure resolution for xyz (different positions have different resolution)
    x_res = np.sum(abs(grad_x) * res, axis=2)
    y_res = np.sum(abs(grad_y) * res, axis=2)
    z_res = np.sum(abs(grad_z) * res, axis=2)

    xyz_res = np.stack((x_res, y_res, z_res), axis=2)

    return xyz_res

def show_topk(k, result, label):
    
    with torch.no_grad():
        rank = result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    
        return accuracy

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def eval_skeleton(trans, skeleton, skeleton_gt):
    if skeleton_gt == None:
        epe = 0
    else:
        skeleton = skeleton.index_select(2, torch.tensor([2, 4]).cuda())
        h_pc = torch.cat((skeleton,torch.ones((skeleton.size()[0],1,skeleton.size()[2])).cuda()),dim=1)
        pred = torch.matmul(trans,h_pc)[:,:3]
        #epe = np.sqrt(np.sum((pred, skeleton_gt.index_select(2, torch.tensor([0, 2]).cuda())).cpu().numpy() ** 2, 2) + 1e-20)
        error = pred - skeleton_gt.index_select(2, torch.tensor([2, 4]).cuda())
        epe = torch.mean(torch.abs(error))
    
    if torch.isnan(torch.tensor(epe)):
        epe = 0
    return epe

def IOU(pred,target,n_classes = 5):
    ious = []
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        # target_sum = target_inds.sum()
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        ious.append(float(intersection)/float(max(union,1)))
        """
        if union == 0:
            ious.append(float('nan')) # If there is no ground truth，do not include in evaluation
        else:
            ious.append(float(intersection)/float(max(union,1)))
        """
    return ious

def eval_parsing(pc, pred, pos_label, gt, n_classes = 5):
    gt_f = gt-pc
    pos_label = pos_label.squeeze(2).squeeze(0)
    score = pred.data.cpu().numpy().tolist()
    labels = pos_label.data.cpu().numpy().tolist()
    acc = torch.sum(pred == pos_label)

    #MIOU = jaccard_score(score, labels, average='macro')
    #ious = IOU(score, labels)
    ious = IOU(pred, pos_label)
    miou = sum(ious)/n_classes
    
    move_idx = torch.nonzero((gt_f[0]!=0), as_tuple=False)
    move_pred = torch.index_select(pred, dim=0, index = move_idx[:, 0])
    move_poslabel = torch.index_select(pos_label, dim=0, index = move_idx[:, 0])
    move_acc = torch.sum(move_pred == move_poslabel)/len(move_pred)
    parsing_metric = {'oA': acc, 'pred': score, 'true': labels, 'mIoU': miou, 'move_acc': move_acc}
    
    return parsing_metric
    

def eval_scene_flow(pc, pred, labels, args):

    if pred.size()[1] != labels.size()[1]:
        labels = torch.transpose(labels,2, 1)
    pred = pred.detach() 
    #pred += pc   
    
    gt_f = labels-pc
    dis = torch.norm(gt_f, dim=2)
    move_idx = torch.nonzero((dis>0.1), as_tuple=False) #
    #move_idx = torch.nonzero((gt_f[0]>=0.005), as_tuple=False)
    move_pred = torch.index_select(pred, dim=1, index = move_idx[:, 0])
    move_gt = torch.index_select(labels, dim=1, index = move_idx[:, 0])
    move_error = np.sqrt(np.sum((move_pred.cpu().numpy() - move_gt.cpu().numpy()) ** 2, 2) + 1e-20)
    if move_error.size == 0:
        move_epe =0
    else: move_epe = np.mean(move_error)/10
    #move_epe = np.mean(np.sqrt(np.sum((move_pred.cpu().numpy() - move_gt.cpu().numpy()) ** 2, 2) + 1e-20))
    
    #static_epe
    #static_idx = torch.nonzero((gt_f[0]==0), as_tuple=False)
    static_idx = torch.nonzero((dis<=0.1), as_tuple=False)
    static_pred = torch.index_select(pred, dim=1, index = static_idx[:, 0])
    static_gt = torch.index_select(labels, dim=1, index = static_idx[:, 0])
    static_error = np.sqrt(np.sum((static_pred.cpu().numpy() - static_gt.cpu().numpy()) ** 2, 2) + 1e-20)  
    if static_error.size == 0:
        static_epe =0
    else: static_epe = np.mean(static_error)/10
    
    pc = pc.cpu().numpy()
    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()
    error = np.sqrt(np.sum((pred - labels) ** 2, 2) + 1e-20)/10
    error_x = np.abs(pred[0, :, 0] - labels[0, :, 0])
    error_y = np.abs(pred[0, :, 1] - labels[0, :, 1])
    error_z = np.abs(pred[0, :, 2] - labels[0, :, 2])
    gtflow_len = np.sqrt(np.sum(labels * labels, 2) + 1e-20)

    ## compute traditional metric for scene flow
    epe = np.mean(error)
    epe_x = np.mean(error_x)
    epe_y = np.mean(error_y)
    epe_z = np.mean(error_z)

    accs = np.sum(np.logical_or((error <= 0.05), (error / gtflow_len <= 0.05))) / (np.size(pred, 0) * np.size(pred, 1)) #
    accr = np.sum(np.logical_or((error <= 0.10), (error / gtflow_len <= 0.10))) / (np.size(pred, 0) * np.size(pred, 1)) #0.01
    
    accss = np.sum(np.logical_or((error <= 0.025), (error / gtflow_len <= 0.025))) / (np.size(pred, 0) * np.size(pred, 1))
    accrr = np.sum(np.logical_or((error <= 0.05), (error / gtflow_len <= 0.05))) / (np.size(pred, 0) * np.size(pred, 1))

    sf_metric = {'epe': epe, 'move_epe': move_epe, 'static_epe': static_epe, 'accs': accs, 'accr': accr, 'accss': accss, 'accrr': accrr,\
                 'epe_x': epe_x, 'epe_y': epe_y, 'epe_z': epe_z}

    return sf_metric

"""
def eval_trans_RPE(gt_trans, rigid_trans):
    ## Use the RPE to evaluate the prediction
    gt_trans = gt_trans.cpu().numpy()
    rigid_trans = rigid_trans.cpu().detach().numpy()
    error_sf = calculate_rpe_vector(gt_trans, rigid_trans)
    trans_error_sf = calc_rpe_error(error_sf, error_type='translation_part')
    rot_error_sf = calc_rpe_error(error_sf, error_type='rotation_angle_deg')
    pose_metric = {'RTE': np.array(trans_error_sf).mean(), 'RRE': np.array(rot_error_sf).mean()}

    return pose_metric
"""



def eval_motion_seg(pre, gt):
    pre = pre.cpu().detach().numpy()
    gt = gt.cpu().numpy()
    tp = np.logical_and((pre == 1), (gt == 1)).sum()
    tn = np.logical_and((pre == 0), (gt == 0)).sum()
    fp = np.logical_and((pre == 1), (gt == 0)).sum()
    fn = np.logical_and((pre == 0), (gt == 1)).sum()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn + 1e-10)
    miou = 0.5 * (tp / (tp + fp + fn + 1e-10) + tn / (tn + fp + fn + 1e-10))
    seg_metric = {'acc': acc, 'miou': miou, 'sen': sen}

    return seg_metric

def main():
    output = torch.randn(1,3,5)
    move_idx = torch.nonzero((output[0][0]>0), as_tuple=False)
    print(output)
    print(move_idx)
    output = torch.index_select(output, dim=2, index = move_idx.squeeze())
    print(output)

if __name__ == '__main__':
    main()