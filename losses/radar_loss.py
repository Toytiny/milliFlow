import torch
import math

def compute_loss_with_mask(pred_f, gt, mask):
    para = torch.exp(torch.norm(gt, dim=2))
    error = pred_f - gt
    error_norm = torch.norm(error, dim=2)

    masked_error = error_norm * (mask == 0).float()
    masked_para = para * (mask == 0).float()

    loss = torch.sum(masked_error * masked_para) / (torch.sum(mask == 0).float() + 1e-8)
    return loss

def compute_loss(pred_f, gt, mask):
    error = pred_f - gt
    error_norm = torch.norm(error, dim=2)

    masked_error = error_norm * (mask == 0).float()

    loss = torch.sum(masked_error) / (torch.sum(mask == 0).float() + 1e-8)
    return loss


def mmflowLoss(pred_f, gt, pc, mask):

    gt_f = gt.to(torch.float64)-pc.to(torch.float64)
    dis = torch.norm(gt_f, dim=2)
    move_idx = torch.nonzero((dis>0.05), as_tuple=False)
    static_idx = torch.nonzero((dis<=0.05), as_tuple=False)
    
    move_pred = torch.index_select(pred_f, dim=1, index = move_idx[:, 0])
    move_gt = torch.index_select(gt_f, dim=1, index = move_idx[:, 0])
    move_mask = torch.index_select(mask, dim=1, index = move_idx[:, 0])
    
    static_pred = torch.index_select(pred_f, dim=1, index = static_idx[:, 0])
    static_gt = torch.index_select(gt_f, dim=1, index = static_idx[:, 0])
    static_mask = torch.index_select(mask, dim=1, index = static_idx[:, 0])
    
    dyn_loss = compute_loss_with_mask(move_pred, move_gt, move_mask)
    stat_loss = compute_loss_with_mask(static_pred, static_gt, static_mask)
    
    if math.isnan(dyn_loss):
        dyn_loss = 0

    loss = 2*dyn_loss + stat_loss

    items = {
        'Loss': loss.item(),
    }

    return loss, items