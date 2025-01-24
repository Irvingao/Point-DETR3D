import math
import copy

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init, constant_init
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from torch.nn.init import normal_

from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils import Transformer

# from projects.mmdet3d_plugin.models.futr3d_utils.imgRoi_self_cross_attn import RoiSelfCrossAttn
# from projects.mmdet3d_plugin.models.futr3d_utils.mm_Roi_attn import MMRoiAttn
from projects.mmdet3d_plugin.models.futr3d_utils.imgRoi_cross_attn import ImgRoiCrossAttn


# 定义旋转函数
def rotate(points, angle):
    """
    points: Tensor with shape [1, N, 2]
    angle: Tensor with shape [1, N, 1]
    """
    s = torch.sin(angle)
    c = torch.cos(angle)
    # 构造旋转矩阵
    rotation_matrix = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)
    # 对坐标应用旋转矩阵
    rotated_points = torch.matmul(points, rotation_matrix)
    return rotated_points

@ATTENTION.register_module()
class MSMMRoiAttn(ImgRoiCrossAttn):
    """使用ImgRoiCrossAttn版本的img ROI sampling + 
          RoiSelfCrossAttnV2 pts ROI sampling
          
        - 相对于V2修改 cross attn 的模式为DETR3D的手写格式，而不是直接调用nn.Attn
    """

    def __init__(self,
                 num_cams=6,
                 num_levels=4,
                 feat_fill_value=1.,
                 img_roi=False,
                 img_self_attn=False,
                 pts_roi=False,
                 pts_self_attn=False,
                 modal_self_attn=False,
                 layer_norm=False,
                 query_out=False,
                 query_attn_type="fusion",
                 **kwargs):
        super().__init__(**kwargs)
        
        self.feat_fill_value = feat_fill_value
        
        
        self.img_roi = img_roi
        self.pts_roi = pts_roi
        self.img_self_attn = img_self_attn
        self.pts_self_attn = pts_self_attn
        self.modal_self_attn = modal_self_attn
        self.layer_norm = layer_norm
        self.query_out = query_out
        
        # img ROI attn
        if self.img_self_attn:
            assert self.roi_attn
            self.img_roi_attn = nn.MultiheadAttention(
                kwargs['embed_dims'], kwargs['num_heads'], kwargs['dropout'])
            self.img_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
        
        # pts ROI attn
        if self.pts_self_attn:
            assert self.roi_attn
            self.pts_roi_attn = nn.MultiheadAttention(
                kwargs['embed_dims'], kwargs['num_heads'], kwargs['dropout'])
            self.pts_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
        
        # multi-modal ROI attn
        if self.modal_self_attn:
            assert self.roi_attn
            self.cs_roi_self_attn = nn.MultiheadAttention(
                kwargs['embed_dims'], kwargs['num_heads'], kwargs['dropout'])
            self.roi_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
            
            self.cs_roi_self_attn2 = nn.MultiheadAttention(
                kwargs['embed_dims'], kwargs['num_heads'], kwargs['dropout'])
            self.roi_layer_norm2 = nn.LayerNorm(kwargs['embed_dims'])
        
        self.query_attn_type = query_attn_type
        
        if self.roi_attn:
            num_points = kwargs['roi_size'] * kwargs['roi_size']
            self.num_cams = num_cams
            self.num_points = num_points
            self.num_levels = num_levels
            # query cross attn
            self.attention_weights = nn.Linear(kwargs['embed_dims'],
                                        num_cams*num_levels*num_points)
            self.output_proj = nn.Linear(kwargs['embed_dims'], kwargs['embed_dims'])
            
            self.pts_attention_weights = nn.Linear(kwargs['embed_dims'],
                                            num_levels*num_points)
            self.pts_output_proj = nn.Linear(kwargs['embed_dims'], kwargs['embed_dims'])
            
            self.weight_dropout = nn.Dropout(kwargs['dropout'])
            
            assert query_attn_type in ['sequence', 'fusion'], \
                "The arguments `query_attn_type` in MMRoiAttn \
                is only supported in ['sequence', 'concat', 'fusion']"
            if query_attn_type == 'sequence':
                pass
            elif query_attn_type == 'fusion':
                fused_embed = self.embed_dims * 2
                self.modality_fusion_layer = nn.Sequential(
                    nn.Linear(fused_embed, self.embed_dims),
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(inplace=False),
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.LayerNorm(self.embed_dims),
                )
    
    def init_weight(self):
        """Default initialization for Parameters of Module."""
        if self.roi_attn:
            constant_init(self.attention_weights, val=0., bias=0.)
            xavier_init(self.output_proj, distribution='uniform', bias=0.)
            constant_init(self.pts_attention_weights, val=0., bias=0.)
            xavier_init(self.pts_output_proj, distribution='uniform', bias=0.)
    
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        
        num_query, bs, dim = query.shape
            
        img_roi_feats = None
        roi_pts = None
        cam_key_mask = None
        
        pts_key = None
        pts_key_pos = None
        if self.roi_attn:
            # ------------------------ img roi feature ----------------
            box2d_corners, box2d_cetners, cam_mask =  self.box3d2img2d(
                            kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                            kwargs['pc_range'], kwargs['reference_points'])
            
            '''
            # 获取无效query的mask，用来补齐对应位置的ROI feature
            drop_cam_mask, drop_idx = self.rand_cam_maskV2(cam_mask)
                
            # 去除中心点没有投在image上的无效点
            valid_box2d_corners, valid_box2d_cetners, vaild_cam_mask = \
                self.drop_invalid_value(drop_idx, 
                    box2d_corners, box2d_cetners, drop_cam_mask)
            
            img_roi_feats, roi_pts = self.msbox2imgRoiV2(
                kwargs['img_feats'], valid_box2d_corners, 
                valid_box2d_cetners, vaild_cam_mask, kwargs['img_metas'])
                
            # hard code for zero GT bugs
            if img_roi_feats[0].size(0) == 0: 
                for i, img_roi_feat in enumerate(img_roi_feats):
                    img_roi_feats[i] = query.new_zeros((query.size(0), 
                    img_roi_feat.size(1), img_roi_feat.size(2)))
                roi_pts = query.new_zeros((query.size(0), 
                    roi_pts.size(1), roi_pts.size(2)))
            else:
                img_roi_feats, roi_pts = self.align_feat_num(
                    query, img_roi_feats, roi_pts, drop_idx)
            '''
            # 修改为取所有cam上的ROI
            img_roi_feats, roi_pts = self.msbox2imgRoiV2(
                kwargs['img_feats'], box2d_corners, 
                box2d_cetners, cam_mask, kwargs['img_metas'])
            
            # hard code for zero GT bugs
            if img_roi_feats[0].size(0) == 0: 
                for i, img_roi_feat in enumerate(img_roi_feats):
                    img_roi_feats[i] = query.new_zeros((query.size(0), 
                    img_roi_feat.size(1), img_roi_feat.size(2)))
                roi_pts = query.new_zeros((query.size(0), 
                    roi_pts.size(1), roi_pts.size(2)))
            # img_roi_feats[0]: torch.Size([36, 49, 256])        √
            # query: torch.Size([36, 1, 256])       √
        
            # ------------------------ pts roi feature ----------------
            bev2d_corners = self.box3d2bev2d(
                kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                kwargs['pc_range'], kwargs['reference_points'])
        
            pts_roi_feats, pts_roi_pts = self.msbox2bevRoi(
                kwargs['pts_feats'], bev2d_corners, 
                kwargs['img_metas'])
            
            # hard code for zero GT bugs
            if pts_roi_feats[0].size(0) == 0: 
                for i, pts_roi_feat in enumerate(pts_roi_feats):
                    pts_roi_feats[i] = query.new_zeros((1, 
                        pts_roi_feat.size(1), pts_roi_feat.size(2)))
            
            
            query = query.transpose(0, 1)
            
            if self.img_self_attn or self.pts_self_attn or self.modal_self_attn:
                
                cam_key_padding_mask = ~cam_mask.repeat(self.num_points*self.num_levels,
                    1, 1).permute(1,2,0).contiguous().view(num_query, -1)
                key = torch.cat(img_roi_feats, dim=1)
                pts_key = torch.cat(pts_roi_feats, dim=1)

                key = key.transpose(0, 1)   
                pts_key = pts_key.transpose(0, 1)
            
                # query: torch.Size([1, 26, 256])
                # key: torch.Size([196, 26, 256])
                # pts_key: torch.Size([196, 26, 256])
                
                # ------------------------ ROI Self Attn ------------------------
                # img ROI 交互
                if self.img_self_attn:
                    res_key = key
                    # 这里将 roi的feat作为query，进行交互，object作为bs维
                    key = self.img_roi_attn(
                        query=key, key=key,             # torch.Size([49, 26, 256])
                        value=key,attn_mask=None,
                        key_padding_mask=cam_key_padding_mask)[0]
                    if self.layer_norm:
                        key = self.img_layer_norm(key + res_key)
                    else:
                        key = key + res_key
                
                # pts ROI 交互
                if self.pts_self_attn:
                    res_pts_key = pts_key
                    # 这里将 roi的feat作为query，进行交互，object作为bs维
                    pts_key = self.pts_roi_attn(
                        query=pts_key, key=pts_key,
                        value=pts_key,attn_mask=None,
                        key_padding_mask=None)[0]
                    if self.layer_norm:
                        pts_key = self.pts_layer_norm(pts_key + res_pts_key)
                    else:
                        pts_key = pts_key + res_pts_key
                
                # 跨模态 ROI 交互
                if self.modal_self_attn:
                    if self.query_modal_first == 'img':
                        res_mm_key = key
                        mm_key = self.cs_roi_self_attn(
                            query=key, key=pts_key,
                            value=pts_key,attn_mask=None,
                            key_padding_mask=None)[0]
                        if self.layer_norm:
                            key = self.roi_layer_norm(mm_key+res_mm_key)
                        else:
                            key = mm_key + res_mm_key
                        res_mm_key = pts_key
                        mm_key = self.cs_roi_self_attn(
                            query=pts_key, key=key,
                            value=key,attn_mask=None,
                            key_padding_mask=cam_key_padding_mask)[0]
                        if self.layer_norm:
                            pts_key = self.roi_layer_norm2(mm_key+res_mm_key)
                        else:
                            pts_key = mm_key + res_mm_key
                    elif self.query_modal_first == 'pts':
                        res_mm_key = pts_key
                        mm_key = self.cs_roi_self_attn(
                            query=pts_key, key=key,
                            value=key,attn_mask=None,
                            key_padding_mask=cam_key_padding_mask)[0]
                        if self.layer_norm:
                            pts_key = self.roi_layer_norm(mm_key+res_mm_key)
                        else:
                            pts_key = mm_key + res_mm_key
                        res_mm_key = key
                        mm_key = self.cs_roi_self_attn(
                            query=key, key=pts_key,
                            value=pts_key,attn_mask=None,
                            key_padding_mask=None)[0]
                        if self.layer_norm:
                            key = self.roi_layer_norm(mm_key+res_mm_key)
                        else:
                            key = mm_key + res_mm_key
                # query: torch.Size([1, 26, 256])
                # key: torch.Size([49, 26, 256])
                # pts_key: torch.Size([49, 26, 256])
                img_output = torch.split(key, [self.num_points 
                    for i in range(self.num_levels)], dim=0)
                pts_output = torch.split(pts_key, [self.num_points 
                    for i in range(self.num_levels)], dim=0)
                
                img_output = torch.stack(img_output, dim=0)
                pts_output = torch.stack(pts_output, dim=0)
                # img_output: torch.Size([4, 49, 26, 256])
                img_output = img_output.permute(3,2,1,0)
                pts_output = pts_output.permute(3,2,1,0)
                # img_output: torch.Size([256, 26, 49, 4])
            else:
                img_output = torch.stack(img_roi_feats, dim=-1).permute(2,0,1,3)
                img_output = img_output.view(self.embed_dims, num_query, 
                    self.num_cams, self.num_points, self.num_levels)
                # img_output: torch.Size([256, 26, 6, 49, 4])
                pts_output = torch.stack(pts_roi_feats, dim=-1).permute(2,0,1,3)
                # pts_output: torch.Size([256, 26, 49, 4])
            
            # ------------------------ ROI Cross Attn ------------------------
            if self.img_roi:
                # (1, num_query, numcams, num_points, num_levels)
                img_attention_weights = self.attention_weights(query).view(
                    1, num_query, self.num_cams, self.num_points, self.num_levels)
                # output (B, C, num_query, num_cam, num_points, len(lvl_feats))
                img_attention_weights = self.weight_dropout(img_attention_weights.sigmoid()) * cam_mask[...,None,None]
                img_output = img_output * img_attention_weights
                # output (emb_dims, num_query)
                img_output = img_output.sum(-1).sum(-1).sum(-1).unsqueeze(0)
                img_output = img_output.permute(2, 0, 1)
                # img_output: torch.Size([26, 1, 256])
                img_output = self.output_proj(img_output)
            if self.pts_roi:
                pts_attention_weights =  self.pts_attention_weights(query).view(
                    1, num_query, self.num_points, self.num_levels)

                pts_attention_weights = self.weight_dropout(pts_attention_weights.sigmoid())
                pts_output = pts_output * pts_attention_weights
                pts_output = pts_output.sum(-1).sum(-1).unsqueeze(0)
                pts_output = pts_output.permute(2, 0, 1)

                pts_output = self.pts_output_proj(pts_output)
                
            if self.img_roi and self.pts_roi:
                out = torch.cat((img_output, pts_output), dim=2).permute(1, 0, 2)
                out = self.modality_fusion_layer(out).permute(1, 0, 2)
            elif self.img_roi:
                out = img_output
            elif self.pts_roi:
                out = pts_output
            
            if self.query_out:
                out = self.attn(
                    query=query,    # (num_query ,batch, embed_dims)
                    key=out,
                    value=out,
                    attn_mask=None,
                    key_padding_mask=None)[0]
        else:
            # query self attention
            out = self.attn(
                    query=query,    # (num_query ,batch, embed_dims)
                    key=query,
                    value=query,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask)[0]        
        
        return identity + self.dropout_layer(self.proj_drop(out))
    
    def msbox2imgRoi(self, mlvl_img_feats, box2d_corners, box2d_cetners, cam_mask, img_metas=None):
        assert box2d_corners.size(0) == 1
        bs, num_cam, num_box, _, _ = box2d_corners.shape
        
        # 根据cam_mask直接选出对应的cam
        sel_box2d_corners = box2d_corners[cam_mask]
        sel_box2d_cetners = box2d_cetners[cam_mask]
        
        # Hard Code： 当没有GT的特殊情况,此时唯一的tensor是给定的
        if sel_box2d_corners.size(0) == 0 and cam_mask.size(2) == 1:
            sel_box2d_corners = torch.full_like(box2d_corners[0,0], 0)
            sel_box2d_cetners = torch.full_like(box2d_cetners[0,0], 0)
        
        # TODO 根据 box 的尺寸 生成7x7的 点 最大值-最小值就是box的w，h
        box2d_coords_x_max = sel_box2d_corners[..., 0].max(-1).values
        box2d_coords_x_min = sel_box2d_corners[..., 0].min(-1).values
        box2d_coords_y_max = sel_box2d_corners[..., 1].max(-1).values
        box2d_coords_y_min = sel_box2d_corners[..., 1].min(-1).values
        # 根据w,h创建grid
        box2d_w = box2d_coords_x_max - box2d_coords_x_min
        box2d_h = box2d_coords_y_max - box2d_coords_y_min
        
        num_points = self.roi_size
        # 生成网格坐标
        x = torch.linspace(-0.5, 0.5, num_points)
        y = torch.linspace(-0.5, 0.5, num_points)
        grid_x, grid_y = torch.meshgrid(x, y)  # 形状为 [num_points, num_points]
        # 将网格坐标扩展为每个box的采样点坐标
        grid_x = grid_x.view(1, num_points, num_points).expand(num_box, -1, -1).to(cam_mask.device)
        grid_y = grid_y.view(1, num_points, num_points).expand(num_box, -1, -1).to(cam_mask.device)
        # 根据box的中心坐标和宽高计算采样点坐标
        center_x = sel_box2d_cetners[:, :, :1]
        center_y = sel_box2d_cetners[:, :, 1:]
        sample_x = center_x.view(num_box, 1, 1) + grid_x * box2d_w.view(num_box, 1, 1)
        sample_y = center_y.view(num_box, 1, 1) + grid_y * box2d_h.view(num_box, 1, 1)
        # 组合采样点的x和y坐标
        sampling_points = torch.stack([sample_x, sample_y], dim=3)  # 形状为 [num_boxes, num_points, num_points, 2]
        
        # 将采样点坐标展平成形状为 [num_boxes, num_points*num_points, 2] 的张量
        sampling_points = sampling_points.view(num_box, -1, 2)
        sel_roi_pts = sampling_points.clone()
        
        sampling_points = sampling_points.unsqueeze(0).unsqueeze(-2)
        sampling_points = sampling_points.repeat(num_cam, 1,1,1,1)
        sampling_points = sampling_points.view(num_cam, num_box*num_points*num_points, 1, 2)
        
        sel_sampled_feats = []
        for lvl, img_feats in enumerate(mlvl_img_feats):
            # TODO 获取完采样点，采样 参考attn
            N, B, C, H, W = img_feats.size()
            img_feats = img_feats.view(B*N, C, H, W)
            sampled_feat = F.grid_sample(img_feats, sampling_points)
            sampled_feat = sampled_feat.view(B, N, C, num_box, num_points*num_points, 1)
            sampled_feat = sampled_feat.squeeze(-1)
            sampled_feat = sampled_feat.permute(0, 1, 3, 2, 4)  # [1, 6, 107, 256, 49]
            # 获取 对应cam对应的ROI区域
            sel_sampled_feat = sampled_feat[cam_mask]
            sel_sampled_feat = sel_sampled_feat.permute(0,2,1)
            sel_sampled_feats.append(sel_sampled_feat)
        
        return sel_sampled_feats, sel_roi_pts
    
    def msbox2imgRoiV2(self, mlvl_img_feats, box2d_corners, box2d_cetners, cam_mask, img_metas=None):
        '''
        不用mask， 六个cam都输出
        '''
        assert box2d_corners.size(0) == 1
        bs, num_cam, num_box, _, _ = box2d_corners.shape
        
        # 根据cam_mask直接选出对应的cam
        # sel_box2d_corners = box2d_corners[cam_mask]
        # sel_box2d_cetners = box2d_cetners[cam_mask]
        sel_box2d_corners = box2d_corners.squeeze(0)
        sel_box2d_cetners = box2d_cetners.squeeze(0)
        
        # Hard Code： 当没有GT的特殊情况,此时唯一的tensor是给定的
        if sel_box2d_corners.size(0) == 0 and cam_mask.size(2) == 1:
            sel_box2d_corners = torch.full_like(box2d_corners[0], 0)
            sel_box2d_cetners = torch.full_like(box2d_cetners[0], 0)
        
        # TODO 根据 box 的尺寸 生成7x7的 点 最大值-最小值就是box的w，h
        box2d_coords_x_max = sel_box2d_corners[..., 0].max(-1).values
        box2d_coords_x_min = sel_box2d_corners[..., 0].min(-1).values
        box2d_coords_y_max = sel_box2d_corners[..., 1].max(-1).values
        box2d_coords_y_min = sel_box2d_corners[..., 1].min(-1).values
        # 根据w,h创建grid
        box2d_w = box2d_coords_x_max - box2d_coords_x_min
        box2d_h = box2d_coords_y_max - box2d_coords_y_min
        
        num_points = self.roi_size
        # 生成网格坐标
        x = torch.linspace(-0.5, 0.5, num_points)
        y = torch.linspace(-0.5, 0.5, num_points)
        grid_x, grid_y = torch.meshgrid(x, y)  # 形状为 [num_points, num_points]
        # 将网格坐标扩展为每个box的采样点坐标
        grid_x = grid_x.view(1, num_points, num_points).expand(num_box, -1, -1).to(cam_mask.device)
        grid_y = grid_y.view(1, num_points, num_points).expand(num_box, -1, -1).to(cam_mask.device)
        # 根据box的中心坐标和宽高计算采样点坐标
        center_x = sel_box2d_cetners[..., :1]
        center_y = sel_box2d_cetners[..., 1:]
        sample_x = center_x.view(num_cam, num_box, 1, 1) + grid_x[None,...] * box2d_w.view(num_cam, num_box, 1, 1)
        sample_y = center_y.view(num_cam, num_box, 1, 1) + grid_y[None,...] * box2d_h.view(num_cam, num_box, 1, 1)
        # 组合采样点的x和y坐标
        sampling_points = torch.stack([sample_x, sample_y], dim=3)  # 形状为 [num_boxes, num_points, num_points, 2]
        
        # 将采样点坐标展平成形状为 [num_boxes, num_points*num_points, 2] 的张量
        sampling_points = sampling_points.view(num_cam, num_box, -1, 2)
        sel_roi_pts = sampling_points.clone()
        
        sampling_points = sampling_points.unsqueeze(-2)
        sampling_points = sampling_points.view(num_cam, num_box*num_points*num_points, 1, 2)
        
        sel_sampled_feats = []
        for lvl, img_feats in enumerate(mlvl_img_feats):
            # TODO 获取完采样点，采样 参考attn
            N, B, C, H, W = img_feats.size()
            img_feats = img_feats.view(B*N, C, H, W)
            sampled_feat = F.grid_sample(img_feats, sampling_points)
            sampled_feat = sampled_feat.view(B, N, C, num_box, num_points*num_points, 1)
            sampled_feat = sampled_feat.squeeze(-1)
            sampled_feat = sampled_feat.permute(0, 1, 3, 2, 4)  # [1, 6, 26, 256, 49]
            '''
            # 获取 对应cam对应的ROI区域
            # sel_sampled_feat = sampled_feat[cam_mask]
            '''
            sampled_feat = sampled_feat.squeeze(0).permute(1,2,0,3).contiguous()
            sampled_feat = sampled_feat.view(num_box, 
                self.embed_dims, num_points*num_points*num_cam)
            sel_sampled_feat = sampled_feat.permute(0,2,1)   # [26, 49, 256]
            sel_sampled_feats.append(sel_sampled_feat)
        
        return sel_sampled_feats, sel_roi_pts
    
    def msbox2bevRoi(self, mlvl_pts_feats, bevbox2d_corners, img_metas=None):
        '''
        bevbox2d_corners: (bs, num_box, 5), XYWHR format, value range: [-1,1]
        '''
        assert bevbox2d_corners.size(0) == 1
        bs, num_box, dim = bevbox2d_corners.shape
        assert dim == 5, 'bevbox2d_corners should be XYWHR format.'
        
        device = mlvl_pts_feats[0].device
        
        '''
        # Hard Code： 当没有GT的特殊情况,此时唯一的tensor是给定的
        if sel_box2d_corners.size(0) == 0 and cam_mask.size(2) == 1:
            sel_box2d_corners = torch.full_like(box2d_corners[0,0], 0)
            sel_box2d_cetners = torch.full_like(box2d_cetners[0,0], 0)
        
        # TODO 根据 box 的尺寸 生成7x7的 点 最大值-最小值就是box的w，h
        '''
        box2d_w = bevbox2d_corners[..., 2]
        box2d_h = bevbox2d_corners[..., 3]
        
        num_points = self.roi_size
        # 生成网格坐标
        x = torch.linspace(-0.5, 0.5, num_points)
        y = torch.linspace(-0.5, 0.5, num_points)
        grid_x, grid_y = torch.meshgrid(x, y)  # 形状为 [num_points, num_points]
        # 将网格坐标扩展为每个box的采样点坐标
        grid_x = grid_x.view(1, num_points, num_points).expand(num_box, -1, -1).to(device)
        grid_y = grid_y.view(1, num_points, num_points).expand(num_box, -1, -1).to(device)
        # 根据box的中心坐标和宽高计算采样点坐标
        center_x = bevbox2d_corners[:, :, 0]
        center_y = bevbox2d_corners[:, :, 1]
        sample_x = center_x.view(num_box, 1, 1) + grid_x * box2d_w.view(num_box, 1, 1)
        sample_y = center_y.view(num_box, 1, 1) + grid_y * box2d_h.view(num_box, 1, 1)
        # 组合采样点的x和y坐标
        sampling_points = torch.stack([sample_x, sample_y], dim=3)  # 形状为 [num_boxes, num_points, num_points, 2]
        # 将采样点坐标展平成形状为 [num_boxes, num_points*num_points, 2] 的张量
        sampling_points = sampling_points.view(num_box, -1, 2)
        
        # 旋转grid中所有点到bev orientation
        sampling_points = rotate(sampling_points, bevbox2d_corners[..., -1:])
        
        sel_roi_pts = sampling_points.clone()
        # sel_roi_pts = sampling_points.clone().permute(1,0,2)
        
        sampling_points = sampling_points.unsqueeze(0).unsqueeze(-2)
        sampling_points = sampling_points.view(1, num_box*num_points*num_points, 1, 2)
        
        sampled_feats = []
        # TODO 获取完采样点，采样 参考attn
        for lvl, pts_feats in enumerate(mlvl_pts_feats):
            B, C, H, W = pts_feats.size()
            sampled_feat = F.grid_sample(pts_feats, sampling_points)
            sampled_feat = sampled_feat.view(B, C, num_box, num_points*num_points, 1)
            sampled_feat = sampled_feat.squeeze(-1)
            sampled_feat = sampled_feat.permute(0, 2, 3, 1)  # torch.Size([1, 26, 49, 256])
            sampled_feat = sampled_feat[0]  # [num_obj, num_pts_feat, dims]
            sampled_feats.append(sampled_feat)
            
        # sampled_feat = sampled_feat.permute(1,0,2) # [num_pts_feat, num_obj, dims]
        
        return sampled_feats, sel_roi_pts
    
    def box3d2bev2d(self, gt_bboxes_3d, img_metas, pc_range, device_tensor):
        
        bev2d_coords = []
        for gt_bbox_3d in gt_bboxes_3d:
            bev2d_coord = gt_bbox_3d.bev        # XYWHR
            bev2d_coords.append(bev2d_coord)        # [num_boxes, 5] 
        bev2d_coords = torch.stack(bev2d_coords, dim=0
                            ).to(device_tensor.device) # (B, N, 5)
        # normlize to [-pc,pc] -> [-1, 1]
        bev2d_coords[..., 0] /= pc_range[3]   # X
        bev2d_coords[..., 1] /= pc_range[4]   # Y
        bev2d_coords[..., 2] /= pc_range[3]   # W
        bev2d_coords[..., 3] /= pc_range[4]   # H
        
        return bev2d_coords
    
    def rand_cam_maskV2(self, cam_mask):
        bs, num_box, num_cam = cam_mask.shape
        # Check 有些点会投在两个image上， 这里随机选取一个image进行投影
        # 获取所有 True 的索引
        true_indices = torch.nonzero(cam_mask)

        # 使用torch.bincount()函数统计每个独特元素的数量
        count = torch.bincount(true_indices[:,1])
        
        # 获取出现两次的重复元素的索引
        repeat_indices = torch.nonzero(count == 2, as_tuple=False) # .squeeze()
        if len(repeat_indices.shape) > 1:
            repeat_indices = repeat_indices.squeeze(-1)
        num_rep = repeat_indices.size(0)
        
        if num_rep > 0:
            # 获取随机选取的 idx
            # true_indices = true_indices.permute(1,0)
            rand_overlap_cam = []
            for repeat_indice in repeat_indices:
                rep_value = true_indices[true_indices[:, 1] == repeat_indice, 2]
                rand_overlap_cam.append(rep_value)
            rand_overlap_cam = torch.stack(rand_overlap_cam, 0)
            
            # 生成随机排列的索引
            rand_indices = torch.randint(0, 2, (10,))

            # 获取随机选择的元素
            sel_overlap_cam = rand_overlap_cam[:,0].clone()
            for idx, (overlap_cam, sel_idx) in enumerate(zip(rand_overlap_cam, rand_indices)):
                sel_overlap_cam[idx] = overlap_cam[sel_idx]
            
            # 把选中的置为False，只保留一个cam
            for repeat_indice, sel_cam in zip(repeat_indices, sel_overlap_cam):
                cam_mask[:, repeat_indice, sel_cam] = False
            
        # 处理全为False的情况，去除该query
        zeros_indices = torch.nonzero(count == 0, as_tuple=False) # .squeeze()
        if len(zeros_indices.shape) > 1:
            zeros_indices = zeros_indices.squeeze(-1)
        # 如果最后的为False，需要单独加上idx
        extral_idx = []
        if cam_mask[:,-1].all() == False:
            for n in range(1, num_box):
                i = num_box - n
                if cam_mask[:, i].any() == True:
                    if len(extral_idx) > 0:
                        break
                else:
                    extral_idx.append(i)
        # 去除 extral_idx 中 已经包含在
        zeros_indices = torch.cat([zeros_indices, zeros_indices.new_tensor(extral_idx[::-1])])
        # zeros_indices = repeat_indices.new_tensor(extral_idx[::-1])
        # 去除重复元素
        zeros_indices, _ = torch.unique(zeros_indices, return_inverse=True)
        
        # 此时的cam_mask都仅有一个相机的，或者没有投在任何一个相机上
        cam_mask = cam_mask.permute(0,2,1)   # [bs, cam, query]
        
        return cam_mask, zeros_indices
    
    def align_feat_num(self, query, img_roi_feats, roi_pts, drop_idx):
        '''
        query: [26, 1, 256]
        img_roi_feats: [26, 49, 256]
        '''
        num_query, bs, dim = query.size()
        if drop_idx.size(0) == 0:
            return img_roi_feats, roi_pts
        
        valid_mask = torch.full([num_query], True)
        valid_mask[drop_idx] = False
        
        aligned_img_roi_feats = []
        for i, img_roi_feat in enumerate(img_roi_feats):
            aligned_img_roi_feat = img_roi_feat.new_full(
                [query.size(0), img_roi_feat.size(1), 
                img_roi_feat.size(2)], self.feat_fill_value)
            # fill
            aligned_img_roi_feat[valid_mask] = img_roi_feat
            aligned_img_roi_feats.append(aligned_img_roi_feat)
        
        
        aligned_roi_pts = roi_pts.new_full(
            [query.size(0), roi_pts.size(1), 
             roi_pts.size(2)], 0.)
        aligned_roi_pts[valid_mask] = roi_pts
        
        return aligned_img_roi_feats, aligned_roi_pts


@ATTENTION.register_module()
class MSMMRoiAttnV2(MSMMRoiAttn):
    """使用ImgRoiCrossAttn版本的img ROI sampling + 
          RoiSelfCrossAttnV2 pts ROI sampling
          
        - 相对于V2修改 cross attn 的模式为DETR3D的手写格式，而不是直接调用nn.Attn
    """

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.img_attn = nn.MultiheadAttention(
            kwargs['embed_dims'], kwargs['num_heads'], kwargs['dropout'])
        self.img_attn_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
        self.pts_attn = nn.MultiheadAttention(
            kwargs['embed_dims'], kwargs['num_heads'], kwargs['dropout'])
        self.pts_attn_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
        
        
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        
        num_query, bs, dim = query.shape
            
        img_roi_feats = None
        roi_pts = None
        cam_key_mask = None
        
        pts_key = None
        pts_key_pos = None
        if self.roi_attn:
            # ------------------------ img roi feature ----------------
            box2d_corners, box2d_cetners, cam_mask =  self.box3d2img2d(
                            kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                            kwargs['pc_range'], kwargs['reference_points'])
            
            # 修改为取所有cam上的ROI
            img_roi_feats, roi_pts = self.msbox2imgRoiV2(
                kwargs['img_feats'], box2d_corners, 
                box2d_cetners, cam_mask, kwargs['img_metas'])
            
            # hard code for zero GT bugs
            if img_roi_feats[0].size(0) == 0: 
                for i, img_roi_feat in enumerate(img_roi_feats):
                    img_roi_feats[i] = query.new_zeros((query.size(0), 
                    img_roi_feat.size(1), img_roi_feat.size(2)))
                roi_pts = query.new_zeros((query.size(0), 
                    roi_pts.size(1), roi_pts.size(2)))
            # img_roi_feats[0]: torch.Size([36, 49, 256])        √
            # query: torch.Size([36, 1, 256])       √
        
            # ------------------------ pts roi feature ----------------
            bev2d_corners = self.box3d2bev2d(
                kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                kwargs['pc_range'], kwargs['reference_points'])
        
            pts_roi_feats, pts_roi_pts = self.msbox2bevRoi(
                kwargs['pts_feats'], bev2d_corners, 
                kwargs['img_metas'])
            
            # hard code for zero GT bugs
            if pts_roi_feats[0].size(0) == 0: 
                for i, pts_roi_feat in enumerate(pts_roi_feats):
                    pts_roi_feats[i] = query.new_zeros((1, 
                        pts_roi_feat.size(1), pts_roi_feat.size(2)))
            
            
            query = query.transpose(0, 1)
            
            if self.img_self_attn or self.pts_self_attn or self.modal_self_attn:
                
                cam_key_padding_mask = ~cam_mask.repeat(self.num_points*self.num_levels,
                    1, 1).permute(1,2,0).contiguous().view(num_query, -1)
                key = torch.cat(img_roi_feats, dim=1)
                pts_key = torch.cat(pts_roi_feats, dim=1)

                key = key.transpose(0, 1)   
                pts_key = pts_key.transpose(0, 1)
            
                # query: torch.Size([1, 26, 256])
                # key: torch.Size([196, 26, 256])
                # pts_key: torch.Size([196, 26, 256])
                
                # ------------------------ ROI Self Attn ------------------------
                # img ROI 交互
                if self.img_self_attn:
                    res_key = key
                    # 这里将 roi的feat作为query，进行交互，object作为bs维
                    key = self.img_roi_attn(
                        query=key, key=key,             # torch.Size([49, 26, 256])
                        value=key,attn_mask=None,
                        key_padding_mask=cam_key_padding_mask)[0]
                    if self.layer_norm:
                        key = self.img_layer_norm(key + res_key)
                    else:
                        key = key + res_key
                
                # pts ROI 交互
                if self.pts_self_attn:
                    res_pts_key = pts_key
                    # 这里将 roi的feat作为query，进行交互，object作为bs维
                    pts_key = self.pts_roi_attn(
                        query=pts_key, key=pts_key,
                        value=pts_key,attn_mask=None,
                        key_padding_mask=None)[0]
                    if self.layer_norm:
                        pts_key = self.pts_layer_norm(pts_key + res_pts_key)
                    else:
                        pts_key = pts_key + res_pts_key
                
                # 跨模态 ROI 交互
                if self.modal_self_attn:
                    if self.query_modal_first == 'img':
                        res_mm_key = key
                        mm_key = self.cs_roi_self_attn(
                            query=key, key=pts_key,
                            value=pts_key,attn_mask=None,
                            key_padding_mask=None)[0]
                        if self.layer_norm:
                            key = self.roi_layer_norm(mm_key+res_mm_key)
                        else:
                            key = mm_key + res_mm_key
                        res_mm_key = pts_key
                        mm_key = self.cs_roi_self_attn(
                            query=pts_key, key=key,
                            value=key,attn_mask=None,
                            key_padding_mask=cam_key_padding_mask)[0]
                        if self.layer_norm:
                            pts_key = self.roi_layer_norm2(mm_key+res_mm_key)
                        else:
                            pts_key = mm_key + res_mm_key
                    elif self.query_modal_first == 'pts':
                        res_mm_key = pts_key
                        mm_key = self.cs_roi_self_attn(
                            query=pts_key, key=key,
                            value=key,attn_mask=None,
                            key_padding_mask=cam_key_padding_mask)[0]
                        if self.layer_norm:
                            pts_key = self.roi_layer_norm(mm_key+res_mm_key)
                        else:
                            pts_key = mm_key + res_mm_key
                        res_mm_key = key
                        mm_key = self.cs_roi_self_attn(
                            query=key, key=pts_key,
                            value=pts_key,attn_mask=None,
                            key_padding_mask=None)[0]
                        if self.layer_norm:
                            key = self.roi_layer_norm(mm_key+res_mm_key)
                        else:
                            key = mm_key + res_mm_key
                # query: torch.Size([1, 26, 256])
                # key: torch.Size([49, 26, 256])
                # pts_key: torch.Size([49, 26, 256])
                img_output = torch.split(key, [self.num_points 
                    for i in range(self.num_levels)], dim=0)
                pts_output = torch.split(pts_key, [self.num_points 
                    for i in range(self.num_levels)], dim=0)
                
                img_output = torch.stack(img_output, dim=0)
                pts_output = torch.stack(pts_output, dim=0)
                # img_output: torch.Size([4, 49, 26, 256])
                img_output = img_output.permute(3,2,1,0)
                pts_output = pts_output.permute(3,2,1,0)
                # img_output: torch.Size([256, 26, 49, 4])
            else:
                img_output = torch.stack(img_roi_feats, dim=-1).permute(2,0,1,3)
                img_output = img_output.view(self.embed_dims, num_query, 
                    self.num_cams, self.num_points, self.num_levels)
                # img_output: torch.Size([256, 26, 6, 49, 4])
                pts_output = torch.stack(pts_roi_feats, dim=-1).permute(2,0,1,3)
                # pts_output: torch.Size([256, 26, 49, 4])
            
            # ------------------------ ROI Cross Attn ------------------------
            if self.img_roi:
                # (1, num_query, numcams, num_points, num_levels)
                img_attention_weights = self.attention_weights(query).view(
                    1, num_query, self.num_cams, self.num_points, self.num_levels)
                # output (C, num_query, num_cam, num_points, len(lvl_feats))
                img_attention_weights = self.weight_dropout(img_attention_weights.sigmoid()) * cam_mask[...,None,None]
                img_output = img_output * img_attention_weights
                
                img_output = img_output.sum(-1).sum(-2) 
                # img_output: torch.Size([256, 26, 49])
                img_output = img_output.permute(2,1,0)
                
                img_output = self.img_attn(
                    query=query,    # (num_query ,batch, embed_dims)
                    key=img_output,
                    value=img_output,
                    attn_mask=None,
                    key_padding_mask=None)[0]
                # img_output: torch.Size([26, 1, 256])
                    
                img_output = self.output_proj(img_output)
            if self.pts_roi:
                pts_attention_weights =  self.pts_attention_weights(query).view(
                    1, num_query, self.num_points, self.num_levels)

                pts_attention_weights = self.weight_dropout(pts_attention_weights.sigmoid())
                pts_output = pts_output * pts_attention_weights
                
                pts_output = pts_output.sum(-1)
                pts_output = pts_output.permute(2,1,0)

                pts_output = self.pts_attn(
                    query=query,    # (num_query ,batch, embed_dims)
                    key=pts_output,
                    value=pts_output,
                    attn_mask=None,
                    key_padding_mask=None)[0]

                pts_output = self.pts_output_proj(pts_output)
                
            if self.img_roi and self.pts_roi:
                out = torch.cat((img_output, pts_output), dim=2)
                out = self.modality_fusion_layer(out)
            elif self.img_roi:
                out = img_output
            elif self.pts_roi:
                out = pts_output
            out = out.permute(1, 0, 2)
        else:
            # query self attention
            out = self.attn(
                    query=query,    # (num_query ,batch, embed_dims)
                    key=query,
                    value=query,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask)[0]
        
        # self.roi_key_pos_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        
        return identity + self.dropout_layer(self.proj_drop(out))