import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init, constant_init
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from torch.nn.init import normal_

from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils import Transformer


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

@TRANSFORMER.register_module()
class PointFUTR3DTransformer(BaseModule):
    """Implements the DeformableDETR transformer.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 point_init=True,
                 decoder=None,
                 reference_points_aug=False,
                 **kwargs):
        super(PointFUTR3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.point_init = point_init
        self.two_stage_num_proposals = two_stage_num_proposals
        self.reference_points_aug = reference_points_aug
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        if self.point_init:
            pass
        else:
            self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, MultiScaleDeformableAttention):
        #         m.init_weight()
        if not self.point_init:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_pts_feats,
                mlvl_img_feats,
                query_embed,
                point_coord,
                labels,
                pc_range=None,
                reg_branches=None,
                img_metas=None,
                **kwargs):
        """Forward function for `Transformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert query_embed is not None
        if mlvl_pts_feats:
            bs = mlvl_pts_feats[0].size(0)
        else:
            bs = mlvl_img_feats[0].size(0)
        
        if self.point_init:
            query_pos, query = torch.split(query_embed, self.embed_dims, dim=2)
            # reference_points = self.reference_points(query_pos).sigmoid()
            
            positive_num = [point_coord[idx].size()[0] for idx in range(bs)]
            # padding_num = [300 - positive_num[idx] for idx in range(bs)]
            padding_num = [max(positive_num) - positive_num[idx] for idx in range(bs)]
            
            padding_points = point_coord[0].new_zeros((bs,max(padding_num),3))
            all_reference_points = []
            for idx in range(bs):
                reference_points_3d = point_coord[idx]
                # [-pc, pc] -> [0, 1]
                reference_points_3d[..., 0:1] = (reference_points_3d[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
                reference_points_3d[..., 1:2] = (reference_points_3d[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
                reference_points_3d[..., 2:3] = (reference_points_3d[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
                # all_reference_points.append(reference_points_3d[..., 0:2])
                all_reference_points.append(reference_points_3d)    # 3d pts coords
                if padding_num[idx] == 0:
                    continue
                    # ----- 这个地方是用[idx,:padding_num[idx],:]  还是 [idx,300-padding_num[idx]:,:]--------
                
                all_reference_points[idx] = torch.cat(
                    [all_reference_points[idx], padding_points[idx,:padding_num[idx],:]],dim=0)
            
            all_reference_points = torch.stack(all_reference_points, dim=0)
            reference_points = all_reference_points
            init_reference_out = all_reference_points
        else:
            query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos)
            if self.training and self.reference_points_aug:
                reference_points = reference_points + torch.randn_like(reference_points) 
            reference_points = reference_points.sigmoid()
            init_reference_out = reference_points
            
        # decoder
        query = query.permute(1, 0, 2)
    
        query_pos = query_pos.permute(1, 0, 2)
       
        inter_states, inter_references = self.decoder(
            query=query,
            pts_feats=mlvl_pts_feats,
            img_feats=mlvl_img_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            img_metas=img_metas,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out



