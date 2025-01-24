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
class Deform3DMultiModalCrossAttn(BaseModule):
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
                 use_LiDAR=True,
                 use_Cam=False,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_lid_points=4,
                 num_cam_points=4,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 offset_range=[-5,-5,-1,5,5,2],  # add offset生成的范围约束
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Deform3DMultiModalCrossAttn, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.offset_range = offset_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_lid_points = num_lid_points
        self.num_cam_points = num_cam_points
        self.num_cams = num_cams
        self.use_LiDAR = use_LiDAR
        self.use_Cam = use_Cam
        self.fused_embed = 0
        if self.use_Cam:
            self.cam_value_proj = nn.Linear(embed_dims, embed_dims) # √
            # TODO cam这里不同level用的相同offset，检查必要性
            self.deform_cam_sampling_offsets = nn.Linear(
            embed_dims, num_heads * 1 * num_cam_points * 3) # √
            self.cam_attention_weights = nn.Linear(embed_dims,
                                           num_heads*num_levels*num_cam_points)
            self.cam_output_proj = nn.Linear(embed_dims, embed_dims)
            self.cam_fusion_weights = nn.Linear(embed_dims,
                                            num_cams)    # √
            self.fused_embed += embed_dims
        
        if self.use_LiDAR:
            self.lid_value_proj = nn.Linear(embed_dims, embed_dims) # √
            # TODO lidar这里不同level用的不同offset，检查必要性
            self.deform_lidar_sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_lid_points * 3)
            
            self.lid_attention_weights = nn.Linear(embed_dims,
                                            num_heads * num_levels * self.num_lid_points)
            self.lid_output_proj = nn.Linear(embed_dims, embed_dims)
            self.fused_embed += embed_dims

        if self.fused_embed > embed_dims:
            self.modality_fusion_layer = nn.Sequential(
                nn.Linear(self.fused_embed, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=False),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        if self.use_Cam:
            constant_init(self.cam_fusion_weights, val=0., bias=0.)
            xavier_init(self.cam_value_proj, distribution='uniform', bias=0.)

            # deform_cam_sampling_offsets
            constant_init(self.deform_cam_sampling_offsets, 0.)
            thetas = torch.arange(
                self.num_heads,
                dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin(), thetas.cos()], -1)
            grid_init = (grid_init /
                        grid_init.abs().max(-1, keepdim=True)[0]).view(
                            self.num_heads, 1, 1,
                            3).repeat(1, 1, self.num_cam_points, 1)
            for i in range(self.num_cam_points):
                grid_init[:, :, i, :] *= i + 1
            self.deform_cam_sampling_offsets.bias.data = grid_init.view(-1)
            
            constant_init(self.cam_attention_weights, val=0., bias=0.)
            xavier_init(self.cam_output_proj, distribution='uniform', bias=0.)

        if self.use_LiDAR:
            xavier_init(self.lid_value_proj, distribution='uniform', bias=0.)
            
            # deform_lidar_sampling_offsets
            constant_init(self.deform_lidar_sampling_offsets, 0.)
            thetas = torch.arange(
                self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin(), thetas.cos()], -1)
            grid_init = (grid_init /
                        grid_init.abs().max(-1, keepdim=True)[0]).view(
                            self.num_heads, 1, 1,
                            3).repeat(1, self.num_levels, self.num_lid_points, 1)
            for i in range(self.num_lid_points):
                grid_init[:, :, i, :] *= i + 1
            self.deform_lidar_sampling_offsets.bias.data = grid_init.view(-1)

            constant_init(self.lid_attention_weights, val=0., bias=0.)
            xavier_init(self.lid_output_proj, distribution='uniform', bias=0.)

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
    
    def normalize_offsetV2(self, sampling_offsets):
        
        offset_normalizer = torch.stack(
            [sampling_offsets.new_tensor(self.offset_range[3]-self.offset_range[0]), 
                sampling_offsets.new_tensor(self.offset_range[4]-self.offset_range[1]), 
                sampling_offsets.new_tensor(self.offset_range[5]-self.offset_range[2])], -1)
        offset_edge = torch.stack(
            [sampling_offsets.new_tensor(self.offset_range[0]), 
                sampling_offsets.new_tensor(self.offset_range[1]), 
                sampling_offsets.new_tensor(self.offset_range[2])], -1)
        pc_normalizer = torch.stack(
            [sampling_offsets.new_tensor(self.pc_range[3]-self.pc_range[0]), 
                sampling_offsets.new_tensor(self.pc_range[4]-self.pc_range[1]), 
                sampling_offsets.new_tensor(self.pc_range[5]-self.pc_range[2])], -1)
        
        sampling_offsets = sampling_offsets.sigmoid()   # 先归一化到[0,1]
        # normlize offest 
        # reference_points为 [0,1]，因此需要将offset也归一化到相同的尺度 pc_range
        sampling_offsets = (sampling_offsets - 0.5) * offset_normalizer
        sampling_offsets = sampling_offsets / pc_normalizer
        '''
        # [0,1]放大到[-5,5]
        sampling_offsets[..., 0:1] = (sampling_offsets[..., 0:1]-0.5)*(self.offset_range[3] - self.offset_range[0])
        sampling_offsets[..., 1:2] = (sampling_offsets[..., 1:2]-0.5)*(self.offset_range[4] - self.offset_range[1])
        sampling_offsets[..., 2:3] = (sampling_offsets[..., 2:3]-0.5)*(self.offset_range[5] - self.offset_range[2])
        # [-5,5]放缩到[0,51.2*2]尺度
        sampling_offsets[..., 0:1] = sampling_offsets[..., 0:1]/(self.pc_range[3] - self.pc_range[0]) # + self.pc_range[0]
        sampling_offsets[..., 1:2] = sampling_offsets[..., 1:2]/(self.pc_range[4] - self.pc_range[1]) # + self.pc_range[1]
        sampling_offsets[..., 2:3] = sampling_offsets[..., 2:3]/(self.pc_range[5] - self.pc_range[2]) # + self.pc_range[2]
        '''
        return sampling_offsets