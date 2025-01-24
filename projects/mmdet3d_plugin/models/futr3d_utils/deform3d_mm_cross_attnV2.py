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

from projects.mmdet3d_plugin.models.futr3d_utils.deform3d_mm_cross_attn import Deform3DMultiModalCrossAttn

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
class Deform3DMultiModalCrossAttnV2(Deform3DMultiModalCrossAttn):
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
                 lid_attn_type="V2",
                 cam_attn_type="V2",
                 union_deform_pts=False,
                 weight_dropout=0.,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.lid_attn_type = lid_attn_type
        self.cam_attn_type = cam_attn_type
        self.union_deform_pts = union_deform_pts
        
        assert self.lid_attn_type in ["V1", "V2"], ""
        assert self.cam_attn_type in ["V1", "V2"], ""
        
        self.weight_dropout = nn.Dropout(weight_dropout)
        
        if self.cam_attn_type == "V2" and self.use_Cam:
            num_cam = 6
            self.cam_attention_weights = nn.Linear(self.embed_dims,
                num_cam*self.num_heads*self.num_levels*self.num_cam_points)
            

        
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
        
        if self.use_LiDAR:
            
            if self.lid_attn_type == "V1":
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
                if torch.cuda.is_available() and lid_value.is_cuda:
                    lid_output = MultiScaleDeformableAttnFunction.apply(
                        lid_value, lid_spatial_shapes, lid_level_start_index, lid_sampling_locations,
                        lid_attention_weights, self.im2col_step)
                else:
                    lid_output = multi_scale_deformable_attn_pytorch(
                        lid_value, lid_spatial_shapes, lid_sampling_locations, lid_attention_weights)

                lid_output = self.lid_output_proj(lid_output)
                
            elif self.lid_attn_type == "V2":
                lid_spatial_shapes = []
                for feat in pts_feats:
                    bs, c, h, w = feat.shape
                    spatial_shape = (h, w)
                    lid_spatial_shapes.append(spatial_shape)
                lid_spatial_shapes = torch.as_tensor(
                    lid_spatial_shapes, dtype=torch.long, device=feat.device)
                
                lid_sampling_offsets = self.deform_lidar_sampling_offsets(query).view(
                    bs, num_query, self.num_heads, self.num_levels, self.num_lid_points, 3)
                # lidar use 2d ref pts to get bev feat
                lid_reference_points = reference_points.clone()
                lid_reference_points = lid_reference_points.unsqueeze(2)
                lid_sampling_offsets = lid_sampling_offsets[..., :2]
                if lid_reference_points.shape[-1] == 3:
                    offset_normalizer = torch.stack(
                        [lid_spatial_shapes[..., 1], lid_spatial_shapes[..., 0]], -1)
                    lid_sampling_locations = lid_reference_points[:, :, None, :, None, :2] \
                        + lid_sampling_offsets \
                        / offset_normalizer[None, None, None, :, None, :]
                else:
                    raise ValueError(
                        f'Last dim of reference_points must be'
                        f'2 or 3, but get {reference_points.shape[-1]} instead.')
                # lid_sampling_locations： torch.Size([1, 26, 8, 4, 16, 3]) [bs, num_q, num_head, num_lvl, num_pts, 2]
                
                lid_sampling_locations = lid_sampling_locations.permute(0,1,2,4,3,5).contiguous()    # [bs, num_q, num_head, num_pts, num_lvl, 2]
                lid_sampling_locations = lid_sampling_locations.view(bs, 
                    num_query, self.num_heads*self.num_lid_points, self.num_levels, 2)
                
                lid_feats = self.feature_sampling_lidar(pts_feats, lid_sampling_locations, self.pc_range, num_query)    # torch.Size([1, 256, 13312, 4])
                lid_feats = lid_feats.view(bs, self.embed_dims, num_query, 
                    self.num_heads, self.num_lid_points, self.num_levels)
                
                # cost too much memo
                # lid_feats = self.lid_value_proj(lid_feats.permute(0,5,2,3,4,1)).permute(0,5,2,3,4,1)
                
                lid_feats = lid_feats.permute(0,1,2,3,5,4)  # 和lid_attention_weights对齐
                
                lid_attention_weights = self.lid_attention_weights(query).view(
                    bs, num_query, self.num_heads, self.num_levels, self.num_lid_points)
                lid_attention_weights = self.weight_dropout(lid_attention_weights.sigmoid())

                lid_output = lid_feats * lid_attention_weights
                lid_output = lid_output.sum(-1).sum(-1).sum(-1)
                
                lid_output = lid_output.permute(0,2,1)
                
                lid_output = self.lid_output_proj(lid_output)   # torch.Size([1, 26, 256])
        
        if self.use_Cam:
            if self.cam_attn_type == "V1":
            
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
                    bs * self.num_cams, num_query, self.num_heads, self.num_levels * self.num_cam_points)

                cam_fusion_weights = self.cam_fusion_weights(query).view(
                    bs, self.num_cams, num_query, 1)
            
                # prepare for deformable attention
                lidar2img = []; img_metas = kwargs['img_metas']; pc_range = self.pc_range
                for img_meta in img_metas:
                    lidar2img.append(img_meta['lidar2img'])
                lidar2img = np.asarray(lidar2img)
                lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
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

                
                mask = mask.view(bs * self.num_cams, num_query, self.num_heads, self.num_levels * self.num_cam_points)
                # filter outrange pts and query
                cam_attention_weights = cam_attention_weights.softmax(-1) * mask

                if torch.cuda.is_available() and cam_value_flatten.is_cuda:
                    cam_output = MultiScaleDeformableAttnFunction.apply(
                        cam_value_flatten, cam_spatial_shapes, cam_level_start_index, reference_points_cam,
                        cam_attention_weights, self.im2col_step)
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
                
            elif self.cam_attn_type == "V2":
        
                lidar2img = []; img_metas = kwargs['img_metas']; pc_range = self.pc_range
                for img_meta in img_metas:
                    lidar2img.append(img_meta['lidar2img'])
                lidar2img = np.asarray(lidar2img)
                lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
                if self.union_deform_pts:
                    cam_sampling_locations_xy = lid_sampling_locations
                    if cam_sampling_locations_xy.shape[-1] == 2:
                        # z-axis normalize by pc_range
                        offset_z_normalizer = pc_range[5] - pc_range[2]
                        cam_sampling_locations_z = lid_reference_points[:, :, None, :, None, 2:3] \
                            + lid_sampling_offsets[..., 2:3] / offset_z_normalizer
                        cam_sampling_locations = torch.cat([cam_sampling_locations_xy, 
                                                            cam_sampling_locations_z], dim=-1)
                        # lid_sampling_locations： torch.Size([1, 26, 8, 4, 16, 3]) [bs, num_q, num_head, num_lvl, num_pts, 2]
                    # [0,1] -> [-pc, pc]
                    cam_sampling_locations[..., 0:1] = cam_sampling_locations[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
                    cam_sampling_locations[..., 1:2] = cam_sampling_locations[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
                    cam_sampling_locations[..., 2:3] = cam_sampling_locations[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
                        
                else:
                    # [0,1] -> [-pc, pc]
                    # prepare for deformable attention
                    reference_points = reference_points.clone()
                    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
                    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
                    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
                    
                    # add offset before projecting them onto 2d plane
                    cam_sampling_offsets = self.deform_cam_sampling_offsets(query).view(
                        bs, num_query, self.num_heads, 1, self.num_cam_points, 3).repeat(1, 1, 1, self.num_levels, 1, 1)
                    # V12 normalize_offsetV2 
                    cam_sampling_offsets = self.normalize_offsetV2(cam_sampling_offsets)
                    cam_sampling_locations = reference_points.view(bs, num_query, 1, 1, 1, 3) + cam_sampling_offsets
                    # torch.Size([1, 26, 8, 4, 16, 3])
                
                cam_sampling_locations = cam_sampling_locations.permute(0,1,2,4,3,5).contiguous()    # [bs, num_q, num_head, num_pts, num_lvl, 2]
                cam_sampling_locations = cam_sampling_locations.view(bs, 
                    num_query, self.num_heads*self.num_cam_points, self.num_levels, 3)
                    
                cam_feats, cam_mask = self.feature_sampling_cam(img_feats, cam_sampling_locations, lidar2img, img_metas)
                # cam_feats torch.Size([1, 6, 256, 3328, 4])
                
                # cam_feats = self.cam_value_proj(cam_feats.permute(0,1,4,3,2)).permute(0,1,4,3,2)
                
                cam_feats = cam_feats.view(bs, self.num_cams, self.embed_dims, num_query, 
                                    self.num_heads, self.num_cam_points, self.num_levels)
                cam_mask = cam_mask.permute(0,1,3,2).view(bs, self.num_cams, 1, num_query, 
                                    self.num_heads, self.num_cam_points, self.num_levels)
                # cam_feats torch.Size([1, 6, 256, 26, 8, 16, 4])
                # cam_mask  torch.Size([1, 6,  1,  26, 8, 16, 4])
                
                # cam_attn_type
                cam_attention_weights = self.cam_attention_weights(query).view(
                    bs, num_query, self.num_cams, self.num_heads, self.num_cam_points, self.num_levels) # torch.Size([1, 26, 6, 8, 16, 4])
                cam_attention_weights = cam_attention_weights.unsqueeze(2)
                cam_attention_weights = cam_attention_weights.permute(0,3,2,1,4,5,6) 
                # cam_attention_weights torch.Size([1, 6, 1, 26, 8, 16, 4])
                
                # cam_fusion_weights = self.cam_fusion_weights(query)
                # cam_fusion_weights = cam_fusion_weights.unsqueeze(2).permute(0,3,2,1)   # torch.Size([1, 6, 1, 26])
                # cam_fusion_weights = cam_fusion_weights.sigmoid()
                
                cam_attention_weights = self.weight_dropout(cam_attention_weights.sigmoid()) * cam_mask
                cam_output = cam_feats * cam_attention_weights

                cam_output = cam_output.sum(-1).sum(-1).sum(-1)
                # sum cam
                cam_output = cam_output.sum(1)
                
                cam_output = cam_output.permute(0, 2, 1)    # [bs, num_q, dim]
                
        if self.use_Cam and self.use_LiDAR:
            output = torch.cat((cam_output, lid_output), dim=2).permute(1, 0, 2)
            output = self.modality_fusion_layer(output) 
        elif self.use_Cam:
            output = cam_output.permute(1, 0, 2)
        elif self.use_LiDAR:
            output = lid_output.permute(1, 0, 2)
        # (num_query, bs ,embed_dims)
        
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat
        # (num_query, bs ,embed_dims)
    
    
    def feature_sampling_lidar(self, mlvl_feats, reference_points, pc_range, num_query):
        
        assert reference_points.size(-1) == 2, ""
        reference_points_rel = reference_points.clone()
        # reference_points_rel = reference_points_rel[...,:2]
        bs, num_q, _, num_lvl, _ = reference_points_rel.shape
        reference_points_rel = reference_points_rel.view(bs, 
            num_query*self.num_heads*self.num_lid_points, self.num_levels, 2)
        num_points = reference_points_rel.size(1)

        # [0, 1] -> [-pc, pc]
        reference_points_rel[..., 0:1] = reference_points_rel[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points_rel[..., 1:2] = reference_points_rel[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        # reference_points_rel[..., 2:3] = reference_points_rel[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
        # reference_points_rel (B, num_sampled_pts, 3)

        # [-pc, pc] -> [-1, 1]
        reference_points_rel[..., 0] = reference_points_rel[..., 0] / pc_range[3]
        reference_points_rel[..., 1] = reference_points_rel[..., 1] / pc_range[4]

        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            if len(feat.size()) == 4:
                feat = feat.unsqueeze(1)
            B, N, C, H, W = feat.size()
            feat = feat.view(B*N, C, H, W)
            reference_points_rel_lvl = reference_points_rel[..., lvl:lvl+1, :]
            # reference_points_rel_lvl = reference_points_rel.view(B*N, num_points, 1, 2)
            sampled_feat = F.grid_sample(feat, reference_points_rel_lvl)    # torch.Size([1, 256, 13312, 1])
            sampled_feat = sampled_feat.squeeze(-1) 
            sampled_feats.append(sampled_feat)
        sampled_feats = torch.stack(sampled_feats, -1)  # torch.Size([1, 256, 13312, 4])
        sampled_feats = torch.nan_to_num(sampled_feats)
        return sampled_feats
    
    
    def feature_sampling_cam(self, mlvl_feats, reference_points, lidar2img, img_metas):
        '''
        reference_points: [-pc, pc]
        
        '''
        assert reference_points.size(-1) == 3, ""
        reference_points = reference_points.clone()
        B, num_query, _, num_lvl, _ = reference_points.shape
        reference_points = reference_points.view(B, 
            num_query*self.num_heads*self.num_cam_points, self.num_levels, 3)   # (B, num_pts, 4, 3)
        num_pts = reference_points.size(1)
        num_cam = lidar2img.size(1)
        
        reference_points = reference_points.view(B, 
            num_query*self.num_heads*self.num_cam_points*self.num_levels, 3)   # (B, num_pts*num_lvl, 3)
        num_lvl_pts = reference_points.size(1)
        
        # ref_point change to (B, num_cam, num_lvl_pts, 4, 1)
        reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., :1])], dim=-1)
        reference_points = reference_points.view(B, 1, num_lvl_pts, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
        # lidar2img chaneg to (B, num_cam, num_lvl_pts, 4, 4)
        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_lvl_pts, 1, 1)
        reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
        eps = 1e-5
        mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        # [0,1] -> [-1,1] 因为grid_sample需要grid输入为[-1,1]
        reference_points_cam = (reference_points_cam - 0.5) * 2 
        mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
        # mask = (mask & (reference_points_cam[..., 0:1] > 0.) 
        #             & (reference_points_cam[..., 0:1] < 1.0) 
        #             & (reference_points_cam[..., 1:2] > 0.) 
        #             & (reference_points_cam[..., 1:2] < 1.0))
        # mask shape (B, 6, num_lvl_pts, 1)
        mask = torch.nan_to_num(mask)

        reference_points_cam = reference_points_cam.view(B * self.num_cams, 
            num_query*self.num_heads*self.num_cam_points, self.num_levels, 2)

        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            N, B, C, H, W = feat.size()
            feat = feat.view(B*N, C, H, W)
            
            reference_points_cam_lvl = reference_points_cam[..., lvl:lvl+1, :]
            # reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2) 
            # sample_feat shape (B*N, C, num_query/10, 10)
            sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)    # torch.Size([6, 256, 3328, 1])
            sampled_feat = sampled_feat.squeeze(-1) 
            sampled_feats.append(sampled_feat)
        sampled_feats = torch.stack(sampled_feats, -1)
        sampled_feats = torch.nan_to_num(sampled_feats) # torch.Size([6, 256, 3328, 4])
        sampled_feats = sampled_feats.view(B, num_cam, self.embed_dims, num_pts, self.num_levels)
        
        return sampled_feats, mask