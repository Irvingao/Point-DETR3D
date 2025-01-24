import mmcv
import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss
# from mmdet.models.losses.smooth_l1_loss import l1_loss


@LOSSES.register_module()
class RTSOL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        tso_rewieght_matrix : [translation, scale, orientation]
    """

    def __init__(self, reduction='mean', loss_weight=1.0, tso_rewieght_matrix=[1.,1.,1.]):
        super(RTSOL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.tso_rewieght_matrix = tso_rewieght_matrix

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
        
        # reweight
        pred = self.reweight(pred)
        target = self.reweight(target)
        
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

    def reweight(self, inputs):
        inputs[:, :3] = inputs[:, :3] * self.tso_rewieght_matrix[0]
        inputs[:, 3:6] = inputs[:, 3:6] * self.tso_rewieght_matrix[1]
        inputs[:, 6:8] = inputs[:, 6:8] * self.tso_rewieght_matrix[2]
        
        return inputs
        

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    
    return loss










from mmdet.core.bbox.match_costs.builder import MATCH_COST

@MATCH_COST.register_module()
class RTSOBBox3DL1Cost(object):
    """BBox3DL1Cost for reweighted translation, scale and orientation.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1., tso_reweight_matrix=[1.,1.,1.]):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight