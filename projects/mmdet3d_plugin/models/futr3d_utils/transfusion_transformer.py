import copy
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.init import normal_

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init, kaiming_init)
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, build_transformer_layer, build_transformer_layer_sequence
from mmdet.models.utils.transformer import DetrTransformerDecoderLayer, Transformer, BaseTransformerLayer
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.models.utils.builder import TRANSFORMER
# from projects.mmdet3d_plugin.models.mm_utils.mm_detr import BaseMultiModalTransformerLayerSequence 
from projects.mmdet3d_plugin.models.mm_utils.mm_detrv2 import Point3DMultiModalTransformerV2 

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from projects.mmdet3d_plugin.datasets.nuscenes_utils.statistics_data import gtlabels2names, dict_wlh

from projects.mmdet3d_plugin.models.futr3d_utils.transfusion_attn import MultiheadAttention, PositionEmbeddingLearned

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

def denormalize_bbox_vr(normalized_bboxes):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes_vr = torch.cat([rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes_vr = torch.cat([rot], dim=-1)
    return denormalized_bboxes_vr



@TRANSFORMER_LAYER.register_module()
class TransFusionTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 d_model=128, 
                 nhead=8, 
                 dim_feedforward=256,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 dropout=0.1, 
                 activation="relu",
                 self_posembed=None, 
                 cross_posembed=None, 
                 cross_only=True,
                 **kwargs):
        super(TransFusionTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 4
        assert set(operation_order) == set(
            ['cross_attn', 'norm', 'ffn'])

        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = PositionEmbeddingLearned(2, d_model)
        self.cross_posembed = PositionEmbeddingLearned(2, d_model)

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def transfusion_forward(self, query, key, query_pos=None, key_pos=None, attn_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask)[0]
        # TODO: check
        # query (q,h), there might be (k, h) in the query that are non-relevant to the (HW, h)
        # the mask (k, HW) will be all True(-inf), and hence NAN after softmax in attention
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # those NANs will keep and will not propogate to other queries, because the linear and norm ops are all within the query
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        # query = query.permute(1, 2, 0)
        return query
    
    def forward(self,
                query,
                # img_feats,
                # pts_feats,
                # img_pos,
                # pts_pos,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        # key = value = self.transfusion_forward(pts_feats, img_feats, pts_pos, img_pos)

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query




class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)
        # print(f"query: {query.shape}, query_pos_embed: {query_pos_embed.shape}")
        # print(f"key: {key.shape}, key_pos_embed: {key_pos_embed.shape}")
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask)[0]
        # TODO: check
        # query (q,h), there might be (k, h) in the query that are non-relevant to the (HW, h)
        # the mask (k, HW) will be all True(-inf), and hence NAN after softmax in attention
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # those NANs will keep and will not propogate to other queries, because the linear and norm ops are all within the query
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TransFusionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, 
                 *args, 
                 vr_init=False,     # 没用，别开
                 wlh_init=False,    # 有用，理论上来讲初始化的box和统计量相关，更好
                 refine_ref_pts=True,   # 
                 roi_center_refine=False,
                 return_intermediate=False, 
                 **kwargs):
        super(TransFusionTransformerDecoder, self).__init__(*args, **kwargs)
        self.vr_init = vr_init
        self.wlh_init = wlh_init
        self.refine_ref_pts = refine_ref_pts
        self.return_intermediate = return_intermediate
        
        self.gtlabels2names = gtlabels2names
        # [w_mean, w_std, l_mean, l_std, h_mean, h_std]
        self.dict_wlh = dict_wlh
        
        # transfusion
        bias = 'auto'
        num_views = 6
        hidden_channel = 128
        self.out_size_factor_img = 4
        
        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            256,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            256,  # channel of img feature map
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.fc = nn.Sequential(*[nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)])
        # Position Embedding for Cross-Attention, which is re-used during training
        grid_size = [512,512,1]
        out_size_factor = 4
        x_size = grid_size[0] // out_size_factor
        y_size = grid_size[1] // out_size_factor
        self.bev_pos = self.create_2D_grid(x_size, y_size)
        
        # self.query_proj = nn.Linear(hidden_channel*2, hidden_channel)
        # self.query_pos_proj = nn.Linear(hidden_channel*2, hidden_channel)
        
        num_heads = 8
        ffn_channel = 256
        dropout = 0.1
        activation= 'relu'
        self.fusion_decoder = nn.ModuleList()
        # cross-attention only layers for projecting img feature onto BEV
        for i in range(num_views):
            self.fusion_decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_only=True,
                ))
        
        
        
        
        
        # roi center refine
        self.roi_center_refine = roi_center_refine
        if self.roi_center_refine:
            self._init_roi_layers()
        
    def _init_roi_layers(self):
        """Initialize classification branch and regression branch of head."""
        reg_branch = []
        for _ in range(2):
            reg_branch.append(Linear(256, 256))
            reg_branch.append(nn.ReLU())
        box_center_branch = Linear(self.embed_dims, 3)
        self.roi_center_offsets = nn.Sequential(*reg_branch, box_center_branch)
        constant_init(self.roi_center_offsets, 0.)

        for m in self.shared_conv_img.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base
    
    def forward(self,
                query,
                query_pos,
                img_feats,
                pts_feats,
                *args,
                reference_points=None,
                reg_branches=None,
                point_init=True,
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
        # query = self.query_proj(query)
        # query_pos = self.query_pos_proj(query_pos)
        
        output = query
        bs = query.size(1)
        # ------------------ V4 增加gt_box_3d在2d image上的投影框 ----------------------
        # pseudo_bboxes_flag = False
        # 所有情况都应当使用pred
        # if point_init or kwargs['gt_bboxes_3d'] == None:
        # for pred init / only for eval without gt
        pseudo_bboxes_flag = True
        pc_range = kwargs['pc_range']
        
        with torch.no_grad():
            # out_boxes_3d = reg_branches[0](torch.cat([output, query_pos], dim=-1))
            out_boxes_3d = reg_branches[0](output)
        
        out_boxes_3d = out_boxes_3d.squeeze(1)
        boxes_3d = out_boxes_3d[...,:6]
        # boxes_3d = denormalize_bbox(pred_bboxes, pc_range)
        # x,y,z直接使用 gt point
        boxes_3d[..., 0:1] = (reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])   # 此时的ref pts为 weak point 先验
        boxes_3d[..., 1:2] = (reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        boxes_3d[..., 2:3] = (reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])
        # w,h,l 使用统计量
        if self.wlh_init:
            labels = kwargs['labels']
            if isinstance(labels, list):
                labels = labels[0]
            labels = labels.cpu().numpy().tolist()
            if len(labels) == 0:
                labels = [-1]
            boxes_3d[..., 3:4] = reference_points.new_tensor([self.dict_wlh[self.gtlabels2names[label]][0] for label in labels])[:,None]
            boxes_3d[..., 4:5] = reference_points.new_tensor([self.dict_wlh[self.gtlabels2names[label]][2] for label in labels])[:,None]
            boxes_3d[..., 5:6] = reference_points.new_tensor([self.dict_wlh[self.gtlabels2names[label]][4] for label in labels])[:,None]
        else:
            boxes_3d[..., 3:5] = 5  # 宽和长都为5
            boxes_3d[..., 5:6] = 4  # 高为4
        
        box_dim = 6
        if self.vr_init:
            bboxes_3d_vr = denormalize_bbox_vr(out_boxes_3d)
            boxes_3d = torch.cat([boxes_3d, bboxes_3d_vr], dim=-1)
            box_dim = 9
                
        kwargs['gt_bboxes_3d'] = [LiDARInstance3DBoxes(boxes_3d.detach().cpu().numpy(), box_dim=box_dim)]
        
        # Transfusion
        #################################
        # image to BEV
        #################################
        img_inputs = img_feats[0]
        num_views, batch_size, dims, h, w = img_inputs.shape
        img_inputs = img_inputs.view(batch_size*num_views, dims, h, w)
        img_feat = self.shared_conv_img(img_inputs)  # [BS * n_views, C, H, W]

        img_h, img_w, num_channel = img_inputs.shape[-2], img_inputs.shape[-1], img_feat.shape[1]
        raw_img_feat = img_feat.view(batch_size, num_views, num_channel, img_h, img_w).permute(0, 2, 3, 1, 4) # [BS, C, H, n_views, W]
        img_feat = raw_img_feat.reshape(batch_size, num_channel, img_h, img_w * num_views)  # [BS, C, H, n_views*W]
        img_feat_collapsed = img_feat.max(2).values
        img_feat_collapsed = self.fc(img_feat_collapsed).view(batch_size, num_channel, img_w * num_views)   # torch.Size([1, 128, 240])
        img_feat_collapsed_pos = self.create_2D_grid(1, img_feat_collapsed.shape[-1]).to(img_feat.device)
        
        inputs = pts_feats[0]
        lidar_feat = self.shared_conv(inputs)
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)
        bev_feat = lidar_feat_flatten   # torch.Size([1, 128, 65536])
        
        # print(f"bev_feat: {bev_feat.shape}")
        # print(f"bev_pos: {bev_pos.shape}")
        assert len(self.fusion_decoder)==num_views
        for idx_view in range(num_views):
            bev_feat = self.fusion_decoder[idx_view](
                bev_feat, img_feat_collapsed[..., img_w * idx_view:img_w * (idx_view + 1)], 
                bev_pos, img_feat_collapsed_pos[:, img_w * idx_view:img_w * (idx_view + 1)])
        bev_feat = bev_feat.permute(2,0,1)  # torch.Size([16384, 1, 128])
        
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points

            # add point cloud features here
            output = layer(
                output,
                query_pos=query_pos,
                key=bev_feat,
                value=bev_feat,
                # img_feats=img_feat_collapsed,
                # pts_feats=bev_feat,
                # img_pos=img_feat_collapsed_pos,
                # pts_pos=bev_pos,
                *args,
                reference_points=reference_points_input,
                # img_roi_feats=img_roi_feats,
                # roi_pts=roi_pts,
                **kwargs)
            output = output.permute(1, 0, 2)    # (num_query, bs, embed_dims) -> 

            if reg_branches is not None:
                # tmp = reg_branches[lid](torch.cat([output, query_pos.permute(1,0,2)], dim=-1))
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3
                if self.refine_ref_pts:
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
                        
                    kwargs['gt_bboxes_3d'] = [LiDARInstance3DBoxes(boxes_3d.detach().cpu().numpy())]

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points
    
    

@TRANSFORMER.register_module()
class TransFusionTransformer(Point3DMultiModalTransformerV2):
    """Implements the DeformableDETR transformer.
    相较于V2，将接口与`Deformable3DMultiModalDetrTransformerV2`对齐
    
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 query_dims=128,
                 num_cams=6,
                 point_init=True,
                 point_init_ref_pts=True,
                 use_encoder=True,
                 use_LiDAR=True,
                 use_Cam=False,
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 **kwargs):
        super(TransFusionTransformer, self).__init__(**kwargs)
        self.query_dims = query_dims
        self.num_cams = num_cams
        self.point_init = point_init
        self.point_init_ref_pts = point_init_ref_pts
        self.use_encoder = use_encoder
        self.use_LiDAR = use_LiDAR
        self.use_Cam = use_Cam
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.encoder.embed_dims
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                       self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.query_dims, 3)


    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self,
                mlvl_pts_feats,
                mlvl_img_feats,
                mlvl_pts_masks,         # lidar
                mlvl_pts_pos_embeds,    # lidar
                query_embed,
                point_coord,
                labels,
                pc_range,
                reg_branches=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_pts_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_pts_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pts_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


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

        memorys = None
        if self.use_Cam:
            _, bs, c, _h, _w = mlvl_img_feats[0].shape
        else:
            bs, c, _h, _w = mlvl_pts_feats[0].shape
        if self.use_encoder:
            # pc feat flatten
            feat_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            for lvl, (feat, mask, pos_embed) in enumerate(
                    zip(mlvl_pts_feats, mlvl_pts_masks, mlvl_pts_pos_embeds)):
                bs, c, h, w = feat.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                feat = feat.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                feat_flatten.append(feat)
                mask_flatten.append(mask)
            feat_flatten = torch.cat(feat_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=feat_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros(
                (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack(
                [self.get_valid_ratio(m) for m in mlvl_pts_masks], 1)

            # lidar encoder only
            reference_points_enc = \
                self.get_reference_points(spatial_shapes,
                                        valid_ratios,
                                        device=feat.device)

            feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
                1, 0, 2)  # (H*W, bs, embed_dims)
            memory = self.encoder(
                query=feat_flatten,
                key=None,
                value=None,
                query_pos=lvl_pos_embed_flatten,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points_enc,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                **kwargs)
            # memory = feat_flatten

            memory = memory.permute(1, 0, 2)    # [num_q, bs, c] -> [bs, num_q, c]
            bs, l_q, c = memory.shape
            
            # format pts feat to [B, C, H, W]
            memory = memory.permute(0, 2, 1)    # [bs, num_q, c] -> [bs, c, num_q]
            memorys = []
            
            level_start_index_list = level_start_index.tolist()
            level_split_section = [level_start_index_list[i+1]-level_start_index_list[i] \
                                        for i in range(len(level_start_index_list)-1)]
            level_split_section.append(l_q-level_start_index_list[-1])  # 
            mlvl_pts_mem = torch.split(memory, level_split_section, dim=-1)
            for lvl, (mem, spatial_shape) in enumerate(
                    zip(mlvl_pts_mem, spatial_shapes)):
                h, w = spatial_shape
                mem = mem.view(bs, c, h, w)
                memorys.append(mem)
        else:
            memorys = mlvl_pts_feats
        
        if self.point_init:
            query_pos, query = torch.split(query_embed, self.query_dims, dim=2)
            
            if self.point_init_ref_pts:
                reference_points = self.reference_points(query_pos).sigmoid()
                
                positive_num = [point_coord[idx].size()[0] for idx in range(bs)]
                # padding_num = [300 - positive_num[idx] for idx in range(bs)]
                padding_num = [max(positive_num) - positive_num[idx] for idx in range(bs)]
                
                # padding_points = point_coord[0].new_zeros((bs,max(padding_num),3))
                all_reference_points = []
                for idx in range(bs):
                    reference_points_3d = point_coord[idx]
                    if reference_points_3d.size(0) == 0:    # 没有点的情况
                        reference_points_3d = reference_points_3d.new_zeros((1,3))
                    # [-pc, pc] -> [0, 1]
                    reference_points_3d[..., 0:1] = (reference_points_3d[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
                    reference_points_3d[..., 1:2] = (reference_points_3d[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
                    reference_points_3d[..., 2:3] = (reference_points_3d[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
                    # all_reference_points.append(reference_points_3d[..., 0:2])
                    all_reference_points.append(reference_points_3d)    # 3d pts coords
                    # all_reference_points.append(reference_points_3d[..., 0:2])
                    if padding_num[idx] == 0:
                        continue
                        # ----- 这个地方是用[idx,:padding_num[idx],:]  还是 [idx,300-padding_num[idx]:,:]--------
                    all_reference_points[idx] = torch.cat(
                        # [all_reference_points[idx], padding_points[idx,:padding_num[idx],:]],dim=0)
                        [all_reference_points[idx],reference_points[idx,:padding_num[idx],:]],dim=0)
                
                all_reference_points = torch.stack(all_reference_points, dim=0)
            else:
                # learned
                reference_points = self.reference_points(query_pos)
                all_reference_points = reference_points.sigmoid()
            
            
            init_reference_out = all_reference_points
        else:
            query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos)
            all_reference_points = reference_points.sigmoid()
            init_reference_out = all_reference_points
        

        # decoder
        query = query.permute(1, 0, 2)
        # memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            img_feats=mlvl_img_feats,
            pts_feats=memorys,
            query_pos=query_pos,
            reference_points=all_reference_points,
            reg_branches=reg_branches,
            pc_range=pc_range,
            labels=labels,
            point_init=self.point_init,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out
