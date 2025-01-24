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


@LOSSES.register_module()
class ContrastiveCosLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ContrastiveCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
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
class ContrastiveNegCosLoss(MSELoss):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__(reduction, loss_weight)
        
    def forward(self,
                pred,
                target,
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
        if weight is not None:
            if len(target.shape) > len(weight.shape):
                weight = weight.unsqueeze(-1).repeat(1,1,target.size(-1))
            
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss