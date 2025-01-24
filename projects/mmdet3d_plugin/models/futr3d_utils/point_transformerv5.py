import copy
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, build_transformer_layer, build_transformer_layer_sequence
from mmdet.models.utils.transformer import DetrTransformerDecoderLayer, Transformer
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.models.utils.builder import TRANSFORMER
# from projects.mmdet3d_plugin.models.mm_utils.mm_detr import BaseMultiModalTransformerLayerSequence 
from projects.mmdet3d_plugin.models.mm_utils.mm_detrv2 import Point3DMultiModalTransformerV2 

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet3d.core.bbox import LiDARInstance3DBoxes


# Avoid BC-breaking of importing MultiScaleDeformableAttention from this file
try:
    from mmcv.ops.multi_scale_deform_attn import \
        MultiScaleDeformableAttention  # noqa F401
    warnings.warn(
        ImportWarning(
            '``MultiScaleDeformableAttention`` has been moved to '
            '``mmcv.ops.multi_scale_deform_attn``, please change original path '  # noqa E501
            '``from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention`` '  # noqa E501
            'to ``from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention`` '  # noqa E501
        ))

except ImportError:
    warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
                  '``mmcv.ops.multi_scale_deform_attn``, '
                  'You should install ``mmcv-full`` if you need this module. ')
    
    
def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class FUTR3DTransformerDecoderV5(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, 
                 *args, 
                 return_intermediate=False, 
                 **kwargs):
        super(FUTR3DTransformerDecoderV5, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        # self.adp_ = nn.Parameter(torch.Tensor([0.5, 0.25, 0.15, 0.1]), requires_grad=False)  # 设置固定的权重系数, 不用归一化, 直接乘过去

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        
        output = query
        bs = query.size(1)
        # ------------------ V5 修正：默认初始anchor为预测的而不是GT ----------------------
        pseudo_bboxes_flag = True
        pc_range = kwargs['pc_range']
        with torch.no_grad():
            boxes_3d = reg_branches[0](output)[...,:6]
            boxes_3d = boxes_3d.squeeze(1)
            # boxes_3d = denormalize_bbox(pred_bboxes, pc_range)
            # x,y,z直接使用 gt point
            boxes_3d[..., 0:1] = (reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
            boxes_3d[..., 1:2] = (reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
            boxes_3d[..., 2:3] = (reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])
            # w,h,l 使用统计量
            boxes_3d[..., 3:5] = 5  # 宽和长都为5
            boxes_3d[..., 5:6] = 4  # 高为4
            
        kwargs['anchor_bboxes_3d'] = [LiDARInstance3DBoxes(boxes_3d.cpu().numpy(), box_dim=6)]
        
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points

            # add point cloud features here
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                # img_roi_feats=img_roi_feats,
                # roi_pts=roi_pts,
                **kwargs)
            output = output.permute(1, 0, 2)    # (num_query, bs, embed_dims) -> 

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                
                # test时需要用预测结果生成 gt_box_3d
                if pseudo_bboxes_flag:
                    boxes_3d = tmp.detach().squeeze(0)
                    boxes_3d = denormalize_bbox(boxes_3d, pc_range)
                    # 
                    boxes_3d[..., 0:1] = (reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
                    boxes_3d[..., 1:2] = (reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
                    boxes_3d[..., 2:3] = (reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])
                    boxes_3d = boxes_3d[...,:7]
                    kwargs['anchor_bboxes_3d'] = [LiDARInstance3DBoxes(boxes_3d.cpu().numpy())]

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points
    