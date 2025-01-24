import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
#                                          TransformerLayerSequence,
#                                          build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
import math
import time

# import transformer based deformable detr
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch

from projects.mmdet3d_plugin.models.futr3d_utils import Deform3DMultiModalCrossAttn

# 定义旋转函数
def rotate(points, angle):
    """
    points: Tensor with shape [1, N, 2]
    angle: Tensor with shape [1, N, 1]
    """
    norm_points = points - 0.5
    s = torch.sin(angle)
    c = torch.cos(angle)
    # 构造旋转矩阵
    rotation_matrix = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)
    # rotation_matrix: torch.Size([8, 2, 2])
    # points: torch.Size([8, 49, 2])
    # 对坐标应用旋转矩阵
    rotated_points = torch.matmul(points, rotation_matrix)
    # rotated_points = points[:,:,None,:] @ rotation_matrix[:,None,:,:]
    # norm_points
    # rotated_points = points[:,:,None,:] @ rotation_matrix[:,None,:,:]
    # points[0,0]
    # rotated_points[0,0]
    # rotation_matrix[0]
    # angle[0,0]
    return rotated_points

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

@ATTENTION.register_module()
class Deform3DRoIWiseMultiModalCrossAttnV2(Deform3DMultiModalCrossAttn):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 img_roi=False,
                 pts_roi=False,
                 img_roi_scale=1,
                 pts_roi_scale=1,
                 roi_size=7,
                 img_roi_size=None,
                 pts_roi_size=None,
                 img_roi_offset=False,
                 pts_roi_offset=False,
                 roi_attn=False,
                 rotate_bev_grid=False,
                 rotate_bev=False,
                 combine_lvl_dim=False,
                 attn_fusion=False,
                 use_pos_encoder=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.img_roi = img_roi
        self.pts_roi = pts_roi
        self.img_roi_scale = img_roi_scale
        self.pts_roi_scale = pts_roi_scale
        self.roi_size = roi_size
        self.roi_attn = roi_attn
        self.img_roi_size = img_roi_size
        self.pts_roi_size = pts_roi_size
        
        self.img_roi_offset = img_roi_offset
        self.pts_roi_offset = pts_roi_offset
        
        self.combine_lvl_dim = combine_lvl_dim
        self.rotate_bev = rotate_bev
        self.rotate_bev_grid = rotate_bev_grid
        self.num_roi_pts = roi_size * roi_size
        
        if self.img_roi_size is None:
            self.img_roi_size = self.roi_size
            self.num_img_roi_pts = self.num_roi_pts
        else:
            self.num_img_roi_pts = img_roi_size * img_roi_size
            
        if self.pts_roi_size is None:
            self.pts_roi_size = self.roi_size
            self.num_pts_roi_pts = self.num_roi_pts
        else:
            self.num_pts_roi_pts = pts_roi_size * pts_roi_size
        
        if self.img_roi:
            self.cam_roi_attention_weights = nn.Linear(self.embed_dims,
                1*1*self.num_img_roi_pts)
                # self.num_heads*self.num_levels*self.num_img_roi_pts)
            self.img_roi_proj = nn.Linear(self.embed_dims*2, self.embed_dims)
            if self.img_roi_offset:
                self.roi_cam_sampling_offsets = nn.Linear(
                    self.embed_dims, self.num_heads * self.num_levels * self.num_img_roi_pts * 2)
            
        if self.pts_roi:
            self.lid_roi_attention_weights = nn.Linear(self.embed_dims,
                1*1*self.num_pts_roi_pts)
            self.pts_roi_proj = nn.Linear(self.embed_dims*2, self.embed_dims)
            if self.pts_roi_offset:
                self.roi_lidar_sampling_offsets = nn.Linear(
                    self.embed_dims, self.num_heads * self.num_levels * self.num_pts_roi_pts * 2)
        
        self.attn_fusion = attn_fusion
        if self.attn_fusion:
            '''1. 
            '''
            self.modality_fusion_attn = nn.MultiheadAttention(self.embed_dims, self.num_heads, dropout=0.0)
            # self.cam_cross_attn = nn.MultiheadAttention(self.embed_dims, self.num_heads, dropout=0.1)
            self.cam_drop = nn.Dropout(0.0)
            # self.lid_cross_attn = nn.MultiheadAttention(self.embed_dims, self.num_heads, dropout=0.1)
            self.lid_drop = nn.Dropout(0.0)
            self.cam_norm = nn.LayerNorm(self.embed_dims)
            self.lid_norm = nn.LayerNorm(self.embed_dims)
            # self.cam_proj = nn.Linear(self.embed_dims, self.embed_dims)
            # self.lid_proj = nn.Linear(self.embed_dims, self.embed_dims)

        self.use_pos_encoder = use_pos_encoder
        
            
            
        self.init_roi_weight()
        
    def init_roi_weight(self):
        
        if self.img_roi:
            constant_init(self.cam_roi_attention_weights, val=0., bias=0.)
            if self.img_roi_offset:
                constant_init(self.roi_cam_sampling_offsets, 0.)
        if self.pts_roi:
            constant_init(self.lid_roi_attention_weights, val=0., bias=0.)
            if self.pts_roi_offset:
                constant_init(self.roi_lidar_sampling_offsets, 0.)
                    
                    
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                level_start_index=None,
                img_feats=None,
                pts_feats=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)
        
        bs, num_query, _ = query.shape

        reference_points_3d = reference_points.clone()
        
        img_metas = kwargs['img_metas']
        pc_range = self.pc_range
        
        if self.use_LiDAR:
            
            # pc feat flatten
            feat_flatten = []
            lid_spatial_shapes = []
            for feat in pts_feats:
                bs, c, h, w = feat.shape
                spatial_shape = (h, w)
                lid_spatial_shapes.append(spatial_shape)
                feat = feat.flatten(2).transpose(1, 2)
                feat_flatten.append(feat)
            feat_flatten = torch.cat(feat_flatten, 1)
            lid_spatial_shapes = torch.as_tensor(
                lid_spatial_shapes, dtype=torch.long, device=feat_flatten.device)
            lid_level_start_index = torch.cat((lid_spatial_shapes.new_zeros(
                (1, )), lid_spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            # value 
            lid_value = feat_flatten

            bs, num_value, _ = lid_value.shape
            assert (lid_spatial_shapes[:, 0] * lid_spatial_shapes[:, 1]).sum() == num_value

            lid_value = self.lid_value_proj(lid_value)
            if key_padding_mask is not None:
                lid_value = lid_value.masked_fill(key_padding_mask[..., None], 0.0)
            lid_value = lid_value.view(bs, num_value, self.num_heads, -1)
            lid_sampling_offsets = self.deform_lidar_sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_lid_points, 3)
            
            lid_attention_weights = self.lid_attention_weights(query).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_lid_points)
            lid_attention_weights = lid_attention_weights.softmax(-1)

            lid_attention_weights = lid_attention_weights.view(bs, num_query,
                                                    self.num_heads,
                                                    self.num_levels,
                                                    self.num_lid_points)
            if self.pts_roi:
                lid_roi_attention_weights = self.lid_roi_attention_weights(query).view(
                    bs, num_query, 1* 1 * self.num_pts_roi_pts)
                lid_roi_attention_weights = lid_roi_attention_weights.repeat(
                    1,1,self.num_heads*self.num_levels) # 复制n份
                lid_roi_attention_weights = lid_roi_attention_weights.softmax(-1)
                lid_roi_attention_weights = lid_roi_attention_weights.view(bs, num_query,
                    self.num_heads, self.num_levels, self.num_pts_roi_pts)
                    # 1, 1, self.num_pts_roi_pts)
            
            # lidar use 2d ref pts to get bev feat
            lid_reference_points = reference_points.clone()
            lid_reference_points = lid_reference_points.unsqueeze(2).repeat(
                                            1,1,self.num_levels,1)
            lid_reference_points = lid_reference_points[..., :2]
            lid_sampling_offsets = lid_sampling_offsets[..., :2]
            
            if lid_reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [lid_spatial_shapes[..., 1], lid_spatial_shapes[..., 0]], -1)
                lid_sampling_locations = lid_reference_points[:, :, None, :, None, :] \
                    + lid_sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
            else:
                raise ValueError(
                    f'Last dim of reference_points must be'
                    f' 2, but get {reference_points.shape[-1]} instead.')
            
            if self.pts_roi:
                lid_roi_sampling_locations = self.box2bevLoc(kwargs['gt_bboxes_3d'], 
                    img_metas, pc_range, lid_sampling_locations.device)
                # lid_roi_sampling_locations: torch.Size([26, 49, 2])
                lid_roi_sampling_locations = lid_roi_sampling_locations.unsqueeze(1)\
                    .unsqueeze(1).unsqueeze(0).repeat(1,1,self.num_heads,self.num_levels,1,1)
                if self.pts_roi_offset:
                    # roi offset
                    lid_roi_sampling_offsets = self.roi_lidar_sampling_offsets(query).view(
                        bs, num_query, self.num_heads, self.num_levels, self.num_pts_roi_pts, 2)
                    # normalize
                    lid_roi_sampling_locations = lid_roi_sampling_locations + lid_roi_sampling_offsets \
                        / offset_normalizer[None, None, None, :, None, :]
                    
                lid_sampling_locations = torch.cat([lid_sampling_locations, lid_roi_sampling_locations], dim=-2)
                
                lid_attention_weights = torch.cat([lid_attention_weights, lid_roi_attention_weights], dim=-1)
                
            
            if torch.cuda.is_available() and lid_value.is_cuda:
                # lid_sampling_locations: torch.Size([1, 26, 8, 4, 53, 2])
                # lid_attention_weights: torch.Size([1, 26, 8, 4, 53])
                lid_output = MultiScaleDeformableAttnFunction.apply(
                    lid_value, lid_spatial_shapes, lid_level_start_index, lid_sampling_locations,
                    lid_attention_weights, self.im2col_step)
                # if self.pts_roi:
                    # lid_roi_output = MultiScaleDeformableAttnFunction.apply(
                        # lid_value, lid_spatial_shapes, lid_level_start_index, lid_roi_sampling_locations,
                        # lid_roi_attention_weights, self.im2col_step)
                    # lid_output = self.pts_roi_proj(torch.cat([lid_output, lid_roi_output], dim=-1))
            else:
                lid_output = multi_scale_deformable_attn_pytorch(
                    lid_value, lid_spatial_shapes, lid_sampling_locations, lid_attention_weights)

            lid_output = self.lid_output_proj(lid_output)

        if self.use_Cam:
            src_flattens = []
            cam_spatial_shapes = []
            for i in range(len(img_feats)):
                n, bs, c, h, w = img_feats[i].shape
                cam_spatial_shapes.append((h, w))
                flatten_feat = img_feats[i].view(bs * n, c, h, w).flatten(2).transpose(1, 2)
                src_flattens.append(flatten_feat)
            cam_value_flatten = torch.cat(src_flattens, 1)
            cam_spatial_shapes = torch.as_tensor(cam_spatial_shapes, dtype=torch.long, device=flatten_feat.device)
            cam_level_start_index = torch.cat((cam_spatial_shapes.new_zeros((1, )), cam_spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            query_cam = query.repeat(self.num_cams, 1, 1)
            cam_value_flatten = self.cam_value_proj(cam_value_flatten)
            _, num_value, _ = cam_value_flatten.size()
            cam_value_flatten = cam_value_flatten.view(bs * self.num_cams, num_value, self.num_heads, -1)
            
            cam_attention_weights = self.cam_attention_weights(query_cam).view(
                    bs * self.num_cams, num_query, self.num_heads, self.num_levels, self.num_cam_points)
            
            if self.img_roi:
                cam_roi_attention_weights = self.cam_roi_attention_weights(query_cam).view(
                    bs * self.num_cams, num_query, self.num_img_roi_pts)

            cam_fusion_weights = self.cam_fusion_weights(query).view(
                bs, self.num_cams, num_query, 1)
        
            # prepare for deformable attention
            lidar2img = []
            for img_meta in img_metas:
                lidar2img.append(img_meta['lidar2img'])
            lidar2img = np.asarray(lidar2img)
            lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
            lidar2img_ori = lidar2img.clone()
            reference_points = reference_points.clone()
            reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
            reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
            reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

            # add offset before projecting them onto 2d plane
            cam_sampling_offsets = self.deform_cam_sampling_offsets(query).view(
                bs, num_query, self.num_heads, 1, self.num_cam_points, 3).repeat(1, 1, 1, self.num_levels, 1, 1)
            
            # V12 normalize_offsetV2 
            cam_sampling_offsets = self.normalize_offsetV2(cam_sampling_offsets)
            reference_points = reference_points.view(bs, num_query, 1, 1, 1, 3) + cam_sampling_offsets
            
            
            reference_points = reference_points.view(bs, num_query * self.num_heads * self.num_levels * self.num_cam_points, 3)
            # reference_points (B, num_queries, 4)
            reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
            B, num_query_fake = reference_points.size()[:2]
            num_cam = lidar2img.size(1)
            reference_points = reference_points.view(B, 1, num_query_fake, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
            lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query_fake, 1, 1)
            reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
            eps = 1e-5
            mask = (reference_points_cam[..., 2:3] > eps)
            reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
                reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
            reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
            reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
            mask = (mask & (reference_points_cam[..., 0:1] > 0.) 
                        & (reference_points_cam[..., 0:1] < 1.0) 
                        & (reference_points_cam[..., 1:2] > 0.) 
                        & (reference_points_cam[..., 1:2] < 1.0))
            nan_mask = torch.isnan(mask)
            mask[nan_mask] = 0.

            reference_points_cam = reference_points_cam.view(B * self.num_cams, 
                                                             num_query, 
                                                             self.num_heads, 
                                                             self.num_levels, 
                                                             self.num_cam_points, 2)
            # reference_points_cam: torch.Size([6, 26, 8, 4, 4, 2])
            # mask: torch.Size([6, 26, 8, 16])
            mask = mask.view(bs * self.num_cams, num_query, self.num_heads, self.num_levels, self.num_cam_points)
            
            # filter outrange pts and query
            cam_attention_weights = cam_attention_weights.softmax(-1) * mask
            
            if self.img_roi:
                reference_points_cam_roi, roi_mask = self.box2imgRoiLoc(kwargs['gt_bboxes_3d'], mask, lidar2img_ori, img_metas, pc_range)
                
                reference_points_cam_roi = reference_points_cam_roi.unsqueeze(2).unsqueeze(2).repeat(
                    1,1,self.num_heads, self.num_levels, 1,1)
                
                if self.img_roi_offset:
                    # sigmoid normalize
                    cam_roi_sampling_offsets = self.roi_cam_sampling_offsets(query).sigmoid().view(
                        bs, num_query, self.num_heads, self.num_levels, self.num_img_roi_pts, 2)
                    reference_points_cam_roi = reference_points_cam_roi + cam_roi_sampling_offsets
                    
                
                reference_points_cam = torch.cat([reference_points_cam, reference_points_cam_roi], dim=-2)
                
                roi_mask = roi_mask # .unsqueeze(-1).repeat(1,1,1, 1,self.num_img_roi_pts)
                # roi_mask: torch.Size([6, 26, 1, 1])
                
                # roi_mask = roi_mask.unsqueeze(-1).repeat(1,1,self.num_heads, self.num_levels,self.num_img_roi_pts)
                # reference_points_cam_roi: torch.Size([6, 26, 8, 4, 49, 2])
                
                # mask = torch.cat([mask, roi_mask], dim=-1)
                # roi_mask = roi_mask.view(bs * self.num_cams, num_query, 1, 
                #     1 * self.num_img_roi_pts)
                
                # mask: torch.Size([6, 26, 8, 212])
                cam_roi_attention_weights = cam_roi_attention_weights.repeat(
                    1, 1, self.num_heads*self.num_levels)
                cam_roi_attention_weights = cam_roi_attention_weights.softmax(-1)
                cam_roi_attention_weights = cam_roi_attention_weights.view(self.num_cams, num_query, 
                    self.num_heads, self.num_levels, self.num_img_roi_pts)
                # cam_roi_attention_weights: torch.Size([6, 26, 8, 4, 49])
                cam_roi_attention_weights = cam_roi_attention_weights * roi_mask[..., None]
                
                cam_attention_weights = cam_attention_weights.view(self.num_cams, num_query, 
                    self.num_heads, self.num_levels, self.num_cam_points)                    
                
                cam_attention_weights = torch.cat([cam_attention_weights, cam_roi_attention_weights], dim=-1)
                
                if self.combine_lvl_dim:
                    '''保持和原先的一致
                    # cam_attention_weights: torch.Size([6, 26, 8, 16])
                    # reference_points_cam: torch.Size([6, 26, 8, 4, 4, 2])
                    '''
                    num_pts = cam_attention_weights.size(-1)
                    # reference_points_cam = reference_points_cam.view(self.num_cams, num_query, 
                        # self.num_heads, self.num_levels*num_pts, 2)
                    cam_attention_weights = cam_attention_weights.view(self.num_cams, num_query, 
                        self.num_heads, self.num_levels*num_pts)
                    # cam_attention_weights: torch.Size([6, 26, 8, 212])
                    # reference_points_cam: torch.Size([6, 20, 8, 4, 53, 2])
                    
                
            if torch.cuda.is_available() and cam_value_flatten.is_cuda:
                # reference_points_cam: torch.Size([6, 26, 8, 4, 53, 2])
                # cam_attention_weights: torch.Size([6, 26, 8, 4, 53])
                cam_output = MultiScaleDeformableAttnFunction.apply(
                    cam_value_flatten, cam_spatial_shapes, cam_level_start_index, reference_points_cam,
                    cam_attention_weights, self.im2col_step)
                # if self.img_roi:
                #     cam_roi_output = MultiScaleDeformableAttnFunction.apply(
                #         cam_value_flatten, cam_spatial_shapes, cam_level_start_index, reference_points_cam_roi,
                #         cam_roi_attention_weights, self.im2col_step)
                #     cam_output = self.img_roi_proj(torch.cat([cam_output, cam_roi_output], dim=-1))
            else:
                # WON'T REACH HERE
                print("Won't Reach Here")
                raise ValueError(f"assert {torch.cuda.is_available()} and cam_value_flatten.is_cuda is {cam_value_flatten.is_cuda}")
                # cam_output = multi_scale_deformable_attn_pytorch(
                    # cam_value_flatten, cam_spatial_shapes, sampling_locations, cam_attention_weights)

            cam_fusion_weights = cam_fusion_weights.sigmoid()
            # cam_attention_weights = cam_attention_weights.permute(0, 2, 3, 1)
            cam_output = cam_output.view(bs, self.num_cams, num_query, -1)
                
            cam_output = cam_output * cam_fusion_weights
            cam_output = cam_output.sum(1)
                
            cam_output = self.cam_output_proj(cam_output)
            # [bs, num_q, 256]
        if self.use_Cam and self.use_LiDAR:
            if self.attn_fusion:
                '''
                '''
                # 1. #####################################################
                output = torch.cat((cam_output, lid_output), dim=0)
                output = self.modality_fusion_attn(
                                query=output,
                                key=output,
                                value=output,
                                attn_mask=None,
                                key_padding_mask=None)[0]
                cam_out, lid_out = torch.split(output, [1,1], dim=0)
                # cam_out = self.cam_proj(cam_out)
                cam_output = cam_output + self.cam_drop(cam_out)
                cam_output = self.cam_norm(cam_output)
                # lid_out = self.lid_proj(lid_out)
                lid_output = lid_output + self.lid_drop(lid_out)
                lid_output = self.lid_norm(lid_output)
                # #####################################################
            output = torch.cat((cam_output, lid_output), dim=2).permute(1, 0, 2)
            output = self.modality_fusion_layer(output) 
        elif self.use_Cam:
            output = cam_output.permute(1, 0, 2)
        elif self.use_LiDAR:
            output = lid_output.permute(1, 0, 2)
        # (num_query, bs ,embed_dims)
        if self.use_pos_encoder:
            pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
            return self.dropout(output) + inp_residual + pos_feat
        else:
            return self.dropout(output) + inp_residual
        # (num_query, bs ,embed_dims)
    
    def box2bevLoc(self, gt_bboxes_3d, img_metas, pc_range, device_tensor=None):
        '''
        bevbox2d_corners: (bs, num_box, 5), XYWHR format, value range: [-1,1]
        '''
        bevbox2d_corners = []
        for gt_bbox_3d in gt_bboxes_3d:
            bev2d_coord = gt_bbox_3d.bev        # XYWHR
            bevbox2d_corners.append(bev2d_coord)        # [num_boxes, 5] 
        bevbox2d_corners = torch.stack(bevbox2d_corners, dim=0
                            ).to(device_tensor) # (B, N, 5)
        # normlize to [-pc,pc] -> [0, 1]
        bevbox2d_corners[..., 0] = bevbox2d_corners[..., 0]/(pc_range[3] - pc_range[0]) + 0.5   # X
        bevbox2d_corners[..., 1] = bevbox2d_corners[..., 1]/(pc_range[4] - pc_range[1]) + 0.5   # Y
        bevbox2d_corners[..., 2] = bevbox2d_corners[..., 2]/(pc_range[3] - pc_range[0])         # W
        bevbox2d_corners[..., 3] = bevbox2d_corners[..., 3]/(pc_range[4] - pc_range[1])         # H
        
        assert bevbox2d_corners.size(0) == 1
        bs, num_box, dim = bevbox2d_corners.shape
        assert dim == 5, 'bevbox2d_corners should be XYWHR format.'
        
        '''
        # Hard Code： 当没有GT的特殊情况,此时唯一的tensor是给定的
        if sel_box2d_corners.size(0) == 0 and cam_mask.size(2) == 1:
            sel_box2d_corners = torch.full_like(box2d_corners[0,0], 0)
            sel_box2d_cetners = torch.full_like(box2d_cetners[0,0], 0)
        
        # TODO 根据 box 的尺寸 生成7x7的 点 最大值-最小值就是box的w，h
        '''
        box2d_w = bevbox2d_corners[..., 2]
        box2d_h = bevbox2d_corners[..., 3]
        
        if self.pts_roi_scale != 1:
            box2d_w *= self.pts_roi_scale
            box2d_h *= self.pts_roi_scale
            
        num_points = self.pts_roi_size
        # 生成网格坐标
        x = torch.linspace(-0.5, 0.5, num_points)
        y = torch.linspace(-0.5, 0.5, num_points)
        grid_x, grid_y = torch.meshgrid(x, y)  # 形状为 [num_points, num_points]
        # 将网格坐标扩展为每个box的采样点坐标
        grid_x = grid_x.view(1, num_points, num_points).expand(num_box, -1, -1).to(device_tensor)
        grid_y = grid_y.view(1, num_points, num_points).expand(num_box, -1, -1).to(device_tensor)
        # 根据box的中心坐标和宽高计算采样点坐标
        center_x = bevbox2d_corners[:, :, 0]
        center_y = bevbox2d_corners[:, :, 1]
        
        if self.rotate_bev_grid:
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.view(num_box, self.num_pts_roi_pts, 2)
            rot_grid = rotate(grid, bevbox2d_corners[..., -1:])
            rot_grid = rot_grid.unsqueeze(-2).view(num_box, num_points, num_points, 2)
            grid_x = rot_grid[...,0]
            grid_y = rot_grid[...,1]
        
        sample_x = center_x.view(num_box, 1, 1) + grid_x * box2d_w.view(num_box, 1, 1)
        sample_y = center_y.view(num_box, 1, 1) + grid_y * box2d_h.view(num_box, 1, 1)
        # 组合采样点的x和y坐标
        sampling_points = torch.stack([sample_x, sample_y], dim=3)  # 形状为 [num_boxes, num_points, num_points, 2]
        # 将采样点坐标展平成形状为 [num_boxes, num_points*num_points, 2] 的张量
        sampling_points = sampling_points.view(num_box, -1, 2)
        
        if self.rotate_bev:
            # 旋转grid中所有点到bev orientation
            sampling_points = rotate(sampling_points, bevbox2d_corners[..., -1:])

        return sampling_points
    
    
    def box2imgRoiLoc(self, gt_bboxes_3d, mask, lidar2img, img_metas, pc_range):
        '''
        不用mask， 六个cam都输出
        '''
        device_tensor = mask.device
        box3d_coords = []
        box3d_centers = []
        # box3d_num = []    # 如果bs > 1则 对齐num_box 组成bs投影
        for gt_bbox_3d in gt_bboxes_3d:
            box3d_coord = gt_bbox_3d.corners
            box3d_center = gt_bbox_3d.gravity_center
            # box3d_num.append(box3d_coord.size(0))
            # box3d_coord = torch.cat([box3d_coord, ])
            box3d_coords.append(box3d_coord)     # [num_boxes, 8, 3]
            box3d_centers.append(box3d_center)
        box3d_coords = torch.stack(box3d_coords, dim=0
                            ).to(device_tensor) # (B, N, 4, 4)
        box3d_centers = torch.stack(box3d_centers, dim=0
                            ).to(device_tensor)
        # concat 8 corners and 1 center
        box3d_coords = torch.cat([box3d_coords, box3d_centers.unsqueeze(-2)], 
                                 dim=-2)
        
        bs, num_box, num_corner, coords = box3d_coords.shape

        box3d_coords = box3d_coords.view(bs, num_box*num_corner, coords)
        # (bs, num_pts, 3) -> (bs, num_pts, 4)
        box3d_coords = torch.cat((box3d_coords, torch.ones_like(box3d_coords[..., :1])), -1)
        
        num_cam = lidar2img.size(1)
        box3d_coords = box3d_coords.unsqueeze(1).repeat(1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(bs, num_cam, 1, 4, 4).repeat(1, 1, num_box*num_corner, 1, 1)
        # lidar coords -> cam coords
        cam_box3d_coords = torch.matmul(lidar2img, box3d_coords).squeeze(-1)
        eps = 1e-5
        roi_mask = (cam_box3d_coords[..., 2:3] > eps)
        cam_box3d_coords = cam_box3d_coords[..., 0:2] / torch.max(
            cam_box3d_coords[..., 2:3], torch.ones_like(cam_box3d_coords[..., 2:3])*eps)
        # normlize to [0,1]
        cam_box3d_coords[..., 0] /= img_metas[0]['img_shape'][0][1]
        cam_box3d_coords[..., 1] /= img_metas[0]['img_shape'][0][0]
        roi_mask = (roi_mask & (cam_box3d_coords[..., 0:1] > 0.) 
                    & (cam_box3d_coords[..., 0:1] < 1.0) 
                    & (cam_box3d_coords[..., 1:2] > 0.) 
                    & (cam_box3d_coords[..., 1:2] < 1.0))
        nan_mask = torch.isnan(roi_mask)
        roi_mask[nan_mask] = 0.
        
        # roi_mask: torch.Size([1, 6, 234, 1])
        # cam_box3d_coords: torch.Size([1, 6, 234, 2])
        
        roi_mask = roi_mask.view(bs*num_cam, num_box, num_corner, 1)    # [..., -1, :]        
        cam_box3d_coords = cam_box3d_coords.view(bs*num_cam, num_box, num_corner, 2)   # 只取中心点
        
        roi_mask = roi_mask[...,-1:,:]    # .repeat(1,1,self.num_heads, self.num_img_roi_pts) # 
        box2d_corners_coords, box2d_cetners = torch.split(cam_box3d_coords, [8,1], dim=-2)
        
        num_cam, num_box, num_corner, coords_2d = box2d_corners_coords.shape

        sel_box2d_corners = box2d_corners_coords
        sel_box2d_cetners = box2d_cetners
        # Hard Code： 当没有GT的特殊情况,此时唯一的tensor是给定的
        if sel_box2d_corners.size(0) == 0 and cam_mask.size(2) == 1:
            sel_box2d_corners = torch.full_like(box2d_corners, 0)
            sel_box2d_cetners = torch.full_like(box2d_cetners, 0)
        
        # TODO 根据 box 的尺寸 生成7x7的 点 最大值-最小值就是box的w，h
        box2d_coords_x_max = sel_box2d_corners[..., 0].max(-1).values
        box2d_coords_x_min = sel_box2d_corners[..., 0].min(-1).values
        box2d_coords_y_max = sel_box2d_corners[..., 1].max(-1).values
        box2d_coords_y_min = sel_box2d_corners[..., 1].min(-1).values
        # 根据w,h创建grid
        box2d_w = box2d_coords_x_max - box2d_coords_x_min
        box2d_h = box2d_coords_y_max - box2d_coords_y_min
        
        if self.img_roi_scale != 1:
            box2d_w *= self.img_roi_scale
            box2d_h *= self.img_roi_scale
        
        num_points = self.img_roi_size
        # 生成网格坐标
        x = torch.linspace(-0.5, 0.5, num_points)
        y = torch.linspace(-0.5, 0.5, num_points)
        grid_x, grid_y = torch.meshgrid(x, y)  # 形状为 [num_points, num_points]
        # 将网格坐标扩展为每个box的采样点坐标
        grid_x = grid_x.view(1, num_points, num_points).expand(num_box, -1, -1).to(device_tensor)
        grid_y = grid_y.view(1, num_points, num_points).expand(num_box, -1, -1).to(device_tensor)
        # 根据box的中心坐标和宽高计算采样点坐标
        center_x = sel_box2d_cetners[..., :1]
        center_y = sel_box2d_cetners[..., 1:]
        sample_x = center_x.view(num_cam, num_box, 1, 1) + grid_x[None,...] * box2d_w.view(num_cam, num_box, 1, 1)
        sample_y = center_y.view(num_cam, num_box, 1, 1) + grid_y[None,...] * box2d_h.view(num_cam, num_box, 1, 1)
        # 组合采样点的x和y坐标
        sampling_points = torch.stack([sample_x, sample_y], dim=3)  # 形状为 [num_cam, num_boxes, num_points, num_points, 2]
        
        # 将采样点坐标展平成形状为 [num_boxes, num_points*num_points, 2] 的张量
        sampling_points = sampling_points.view(num_cam, num_box, -1, 2)
        # sel_roi_pts = sampling_points.clone()
        
        # sampling_points = sampling_points.view(num_cam, num_box*num_points*num_points, 1, 2)
        
        
        return sampling_points, roi_mask
    
