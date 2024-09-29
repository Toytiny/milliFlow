import torch
import math

# def compute_loss(est_flow, batch):
#     para = torch.exp(torch.norm(batch, dim=2))
#     true_flow = batch
#     error = est_flow - true_flow
#     loss = torch.mean(torch.norm(error, dim=2) * para)

#     return loss

def compute_loss(est_flow, batch):
    true_flow = batch
    error = est_flow - true_flow
    loss = torch.mean(torch.abs(error))

    return loss

def mmflowLoss(pred_f, gt, pc):

    gt_f = gt.to(torch.float64)-pc.to(torch.float64)
    dis = torch.norm(gt_f, dim=2)
    move_idx = torch.nonzero((dis>0.1), as_tuple=False)
    static_idx = torch.nonzero((dis<=0.1), as_tuple=False)
    
    move_pred = torch.index_select(pred_f, dim=1, index = move_idx[:, 0])
    move_gt = torch.index_select(gt_f, dim=1, index = move_idx[:, 0])
    
    static_pred = torch.index_select(pred_f, dim=1, index = static_idx[:, 0])
    static_gt = torch.index_select(gt_f, dim=1, index = static_idx[:, 0])
    
    dyn_loss = compute_loss(move_pred, move_gt)
    stat_loss = compute_loss(static_pred, static_gt)
    
    if math.isnan(dyn_loss):
        dyn_loss = 0

    loss = 2*dyn_loss + stat_loss

    items = {
        'Loss': loss.item(),
    }

    return loss, items