import torch
from torch.nn import functional as F


def regression_loss(pred, center, openness):
    x, y = center.moveaxis(1, 0)
    
    pred = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    center_loss = F.smooth_l1_loss(pred, center, beta=0.11, reduction='none').sum(1)  # (batch, frames)
    valid_mask = openness.eq(1) & x.gt(0) & x.lt(1) & y.gt(0) & y.lt(1)
    center_loss = torch.where(valid_mask, center_loss, 0).mean()
    
    return center_loss


def tracking_loss(pred, center, openness, gamma=2):
    device = pred.device
    batch_size, _, frames, height, width = pred.shape
    
    x, y = center.moveaxis(1, 0)
    x_ind = (x * width).long().clamp(0, width - 1)  # (batch, frames)
    y_ind = (y * height).long().clamp(0, height - 1)
    x_mod = (x * width) % 1
    y_mod = (y * height) % 1
    center_mod = torch.stack([x_mod, y_mod], dim=1)  # (batch, 2, frames)
    
    pred = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pred_pupil, pred_center_mod = pred[:, 0], pred[:, 1:]
    
    valid_mask = openness.eq(1) & x.gt(0) & x.lt(1) & y.gt(0) & y.lt(1)
    pupil_mask = torch.zeros_like(pred_pupil).bool()  # (batch, frames, height, width)
    
    batch_range = torch.arange(batch_size, device=device).repeat_interleave(frames)
    frames_range = torch.arange(frames, device=device).repeat(batch_size)
    pupil_mask[batch_range, frames_range, y_ind.flatten(), x_ind.flatten()] = 1
    
    # (batch, frames, height, width)
    center_loss = F.smooth_l1_loss(pred_center_mod, center_mod[..., None, None], beta=0.11, reduction='none').sum(1)
    
    focal_loss = torch.where(
        pupil_mask, 
        -1 * (1 - pred_pupil).pow(gamma) * pred_pupil.log() + center_loss, 
        -1 * pred_pupil.pow(gamma) * (1 - pred_pupil).log(), 
    )  # (batch, frames, height, width)
    
    return focal_loss[valid_mask].sum() / valid_mask.sum()


def process_detector_prediction(pred):
    device = pred.device
    batch_size, _, frames, height, width = pred.shape
    
    pred_pupil, pred_x_mod, pred_y_mod = pred.moveaxis(1, 0)
    pred_x_mod = torch.sigmoid(pred_x_mod)
    pred_y_mod = torch.sigmoid(pred_y_mod)
    
    pupil_ind = pred_pupil.flatten(-2, -1).argmax(-1)  # (batch, frames)
    pupil_ind_x = pupil_ind % width
    pupil_ind_y = pupil_ind // width
    
    batch_range = torch.arange(batch_size, device=device).repeat_interleave(frames)
    frames_range = torch.arange(frames, device=device).repeat(batch_size)
    
    pred_x_mod = pred_x_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]
    pred_y_mod = pred_y_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]

    x = (pupil_ind_x + pred_x_mod.view(batch_size, frames)) / width
    y = (pupil_ind_y + pred_y_mod.view(batch_size, frames)) / height
    
    return torch.stack([x, y], dim=1)
    

def p10_acc(pred, center, openness, detector_head=True, 
            height=60, width=80, tolerance=10):
    pred = pred.detach().clone()
    center = center.detach().clone()
    
    if detector_head:
        pred = process_detector_prediction(pred)
    else:
        pred = torch.sigmoid(pred)
    
    pred[:, 0] *= width
    pred[:, 1] *= height
    center[:, 0] *= width
    center[:, 1] *= height
    
    distances = torch.norm(center - pred, dim=1)
    distances_noblinks = distances[openness == 1]

    return (distances < tolerance).sum() / distances.numel(), (distances_noblinks < tolerance).sum() / distances_noblinks.numel(), distances.mean()