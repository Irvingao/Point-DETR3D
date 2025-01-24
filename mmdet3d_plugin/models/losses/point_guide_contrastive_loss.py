# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
import mmcv
from mmdet.core.bbox.match_costs.builder import MATCH_COST
import functools
from mmdet.models.losses.utils import weighted_loss
from mmdet.models.losses import L1Loss

from projects.mmdet3d_plugin.datasets.nuscenes_utils.statistics_data import gtlabels2names, dict_wlh

@LOSSES.register_module()
class PointGuideContrastiveCosLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PointGuideContrastiveCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                img_metas=None,
                pc_range=None,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_contrast = self.loss_weight * contrastive_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        # tensor(0.1998, device='cuda:0', grad_fn=<MulBackward0>)
        return loss_contrast


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def contrastive_cos_loss(pred, target):
    """ Dir cosine similiarity loss
    pred (torch.Tensor): shape [num_samples, num_dir, num_coords]
    target (torch.Tensor): shape [num_samples, num_dir, num_coords]

    """
    if target.numel() == 0:
        return pred.sum() * 0
    
    # assert len(pred.shape) == 4, "input should be [B,C,H,W]"
    # B,C,H,W = pred.size()
    # pred = pred.view(B,C,H*W).permute(0,2,1)
    # target = target.view(B,C,H*W).permute(0,2,1)
    bs, num_feat, dims = pred.shape
    
    loss_func = torch.nn.CosineEmbeddingLoss(reduction='none')
    tgt_param = target.new_ones((bs, num_feat))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0,1), target.flatten(0,1), tgt_param)
    loss = loss.view(bs, num_feat)
    return loss


from mmdet.models.losses.mse_loss import MSELoss, mse_loss
@LOSSES.register_module()
class PointGuideContrastiveNegCosLoss(MSELoss):
    def __init__(self, reduction='mean', loss_weight=1.0, normlize=True):
        super().__init__(reduction, loss_weight)
        self.normlize = normlize
        
    def forward(self,
                pred,
                target,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                img_metas=None,
                pc_range=None,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        bs = pred.size(0)
        
        sample_points = box2bevLoc(gt_bboxes_3d[4:], img_metas, pc_range, 
                                   device_tensor=pred.device)
        
        
        if weight is not None:
            if len(target.shape) > len(weight.shape):
                weight = weight.unsqueeze(-1).repeat(1,1,target.size(-1))
        else:
            weight = [None for i in range(bs)]
        
        losses = 0
        for i, sampling_point in enumerate(sample_points):
            sampling_point = sampling_point.unsqueeze(0)
            select_prd_feat = F.grid_sample(pred[i:i+1], sampling_point)
            select_tgt_feat = F.grid_sample(target[i:i+1], sampling_point).squeeze(-1)
            select_prd_feat = select_prd_feat.squeeze(-1).permute(0,2,1)
            select_tgt_feat = select_tgt_feat.squeeze(-1).permute(0,2,1)
            select_wgt = None
            if weight[i] is not None:
                select_wgt = F.grid_sample(weight[i:i+1], sampling_point).squeeze(-1) 
                select_wgt = select_wgt.squeeze(-1).permute(0,2,1)
            
            if self.normlize:
                select_prd_feat = F.normalize(select_prd_feat, p=2, dim=-1)
                select_tgt_feat = F.normalize(select_tgt_feat, p=2, dim=-1)
            loss = self.loss_weight * mse_loss(
                select_prd_feat, select_tgt_feat, select_wgt, reduction=reduction, avg_factor=avg_factor)
            losses += loss
        losses /= bs
        return losses
    
def box2bevLoc(gt_bboxes_3d, 
               img_metas, 
               pc_range, 
               roi_mode=False,
               pts_roi_size=7,
               pts_roi_scale=1,
               rotate_bev_grid=False,
               device_tensor=None):
    '''
    bevbox2d_corners: (bs, num_box, 5), XYWHR format, value range: [-1,1]
    '''
    bevbox2d_corners = []
    for gt_bbox_3d in gt_bboxes_3d:
        bev2d_coord = gt_bbox_3d.bev        # XYWHR
        # normlize to [-pc,pc] -> [0, 1]
        bev2d_coord[..., 0] = bev2d_coord[..., 0]/(pc_range[3] - pc_range[0]) + 0.5   # X
        bev2d_coord[..., 1] = bev2d_coord[..., 1]/(pc_range[4] - pc_range[1]) + 0.5   # Y
        bev2d_coord[..., 2] = bev2d_coord[..., 2]/(pc_range[3] - pc_range[0])         # W
        bev2d_coord[..., 3] = bev2d_coord[..., 3]/(pc_range[4] - pc_range[1])         # H
        bevbox2d_corners.append(bev2d_coord.to(device_tensor))        # [num_boxes, 5] 
    # bevbox2d_corners = torch.stack(bevbox2d_corners, dim=0
    #                     ).to(device_tensor) # (B, N, 5)
        
    # assert bevbox2d_corners.size(0) == 1
    # bs, num_box, dim = bevbox2d_corners.shape
    # assert dim == 5, 'bevbox2d_corners should be XYWHR format.'
    
    if roi_mode:
        box2d_w = bevbox2d_corners[..., 2]
        box2d_h = bevbox2d_corners[..., 3]
        
        if pts_roi_scale != 1:
            box2d_w *= pts_roi_scale
            box2d_h *= pts_roi_scale
            
        num_points = pts_roi_size
        # 生成网格坐标
        x = torch.linspace(-0.5, 0.5, num_points)
        y = torch.linspace(-0.5, 0.5, num_points)
        grid_x, grid_y = torch.meshgrid(x, y)  # 形状为 [num_points, num_points]
        # 将网格坐标扩展为每个box的采样点坐标
        grid_x = grid_x.view(bs, num_points, num_points).expand(bs, num_box, -1, -1).to(device_tensor)
        grid_y = grid_y.view(bs, num_points, num_points).expand(bs, num_box, -1, -1).to(device_tensor)
        # 根据box的中心坐标和宽高计算采样点坐标
        center_x = bevbox2d_corners[:, :, 0]
        center_y = bevbox2d_corners[:, :, 1]
        
        if rotate_bev_grid:
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.view(bs, num_box, num_points*num_points, 2)
            rot_grid = rotate(grid, bevbox2d_corners[..., -1:])
            rot_grid = rot_grid.unsqueeze(-2).view(bs, num_box, num_points, num_points, 2)
            grid_x = rot_grid[...,0]
            grid_y = rot_grid[...,1]
        
        sample_x = center_x.view(bs, num_box, 1, 1) + grid_x * box2d_w.view(bs, num_box, 1, 1)
        sample_y = center_y.view(bs, num_box, 1, 1) + grid_y * box2d_h.view(bs, num_box, 1, 1)
        # 组合采样点的x和y坐标
        sampling_points = torch.stack([sample_x, sample_y], dim=3)  # 形状为 [num_boxes, num_points, num_points, 2]
        # 将采样点坐标展平成形状为 [num_boxes, num_points*num_points, 2] 的张量
        sampling_points = sampling_points.view(bs, num_box, -1, 2)
    else:
        sampling_points = [bev2d_coord[:,None,:2] for bev2d_coord in bevbox2d_corners] # [torch.Size([31, 1, 2])]

    return sampling_points


@LOSSES.register_module()
class PointGuideWeightedContrastiveNegCosLoss(MSELoss):
    def __init__(self, reduction='mean', loss_weight=1.0, normlize=True):
        super().__init__(reduction, loss_weight)
        self.normlize = normlize
        
    def forward(self,
                pred,
                target,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                img_metas=None,
                pc_range=None,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        bs = pred.size(0)
        num_objs = len(gt_bboxes[0])
        scale_factor = (102.4 / 100)
        feature_map_size = (100, 100)
        heatmap = torch.zeros((100, 100), device=teacher_feat.device)
        gt_bboxes = gt_bboxes[0].tensor
        for k in range(num_objs):
            width = gt_bboxes[k][3]
            length = gt_bboxes[k][4]
            width = width / scale_factor
            length = length / scale_factor
            if width > 0 and length > 0:
                radius = gaussian_radius((length, width), min_overlap=0.1)
            radius = max(2, int(radius))

            x, y, z = gt_bboxes[k][0], gt_bboxes[k][1], gt_bboxes[k][2]
            coor_x = (x - pc_range[0]) / scale_factor
            coor_y = (y - pc_range[1]) / scale_factor
            center = torch.tensor([coor_x, coor_y], dtype=torch.float32,\
                                             device=teacher_feat.device)
            center_int = center.to(torch.int32)
            # throw out not in range objects to avoid out of array
            # area when creating the heatmap
            if not (0 <= center_int[0] < feature_map_size[0]
                    and 0 <= center_int[1] < feature_map_size[1]):
                continue
            draw_heatmap_gaussian(heatmap, center_int, radius)
        
        if self.normlize:
            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)
        
        losses = 0
        for i, sampling_point in enumerate(sample_points):
            sampling_point = sampling_point.unsqueeze(0)
            select_prd_feat = F.grid_sample(pred[i:i+1], sampling_point)
            select_tgt_feat = F.grid_sample(target[i:i+1], sampling_point).squeeze(-1)
            select_tgt_feat = select_tgt_feat.squeeze(-1).permute(0,2,1)
            select_prd_feat = select_prd_feat.squeeze(-1).permute(0,2,1)
            select_wgt = None
            if weight[i] is not None:
                select_wgt = F.grid_sample(weight[i:i+1], sampling_point).squeeze(-1) 
                select_wgt = select_wgt.squeeze(-1).permute(0,2,1)
            loss = self.loss_weight * mse_loss(
                select_prd_feat, select_tgt_feat, select_wgt, reduction=reduction, avg_factor=avg_factor)
            losses += loss
        losses /= bs
        return losses