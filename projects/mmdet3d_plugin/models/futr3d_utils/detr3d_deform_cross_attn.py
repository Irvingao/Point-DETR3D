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
class Detr3DDeformCamCrossAttn(BaseModule):
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
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DDeformCamCrossAttn, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

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
        self.num_points = num_points
        self.num_cams = num_cams
        if self.num_points == 1:
            self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)
        else:
            self.attention_weights = nn.Linear(embed_dims,
                                           num_heads*num_levels*num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.cam_attention_weights = nn.Linear(embed_dims,
                                           num_cams)    # √
        
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first
        
        self.deform_cam_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 3) # √

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.cam_attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        
        constant_init(self.deform_cam_sampling_offsets, 0.)
        ### initilize offset bias for DCN
        # print("NEED TO initialize DCN offset")
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin(), thetas.cos()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         3).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.deform_cam_sampling_offsets.bias.data = grid_init.view(-1)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                level_start_index=None,
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
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
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
        # value 
        value = kwargs['img_feats']
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        if self.num_points == 1:
            attention_weights = self.attention_weights(query).view(
                bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        else:
            attention_weights = self.attention_weights(query)
            attention_weights = attention_weights.view(bs, num_query, 
                                    self.num_heads, self.num_levels, self.num_points)
            attention_weights = attention_weights.repeat(self.num_cams,1,1,1,1)
            
            # attention_weights = self.attention_weights(query_cam).view(
                # bs * self.num_cams, num_query, self.num_heads, self.num_levels * self.num_points)
            # mask = mask.view(bs * self.num_cams, num_query, self.num_heads, self.num_levels * self.num_points)
        
        reference_points_3d = reference_points.clone()
        reference_points = self.get_deform_reference_points_3d(query, reference_points)
        
        _, output, mask = self.feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        # attention_weights: [1, 6, 107, 8, 4, 4]
        # output: [1, 6, 256, 107, 8, 4, 4]
        output = output * attention_weights[:,:,:,None,...]
        output = output.sum(-1).sum(-1).sum(-1)
        
        # cam fusion
        cam_attention_weights = self.cam_attention_weights(query).view(
            bs, self.num_cams, num_query, 1)
        cam_attention_weights = cam_attention_weights.sigmoid()
        
        output = output * cam_attention_weights
        output = output.sum(1)
        # -------
        
        output = self.output_proj(output)
        output = output.permute(1, 0, 2)
        
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat

    def get_deform_reference_points_3d(self, query, reference_points):
        bs, num_query, _ = query.size()
        
        # add offset before projecting them onto 2d plane
        sampling_offsets = self.deform_cam_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 3) # [1, 107, 8, 4, 4, 3]
        
        # repeat ref pts in `num_levels` dims,  -> [bs, num_query, num_levels, 3]
        reference_points = reference_points.unsqueeze(2).repeat(1,1,self.num_levels,1)
        
        reference_points = reference_points.view(bs, num_query, 1, self.num_levels, 1, 3)
        
        # use only pts
        # sampling_locations = sampling_locations.repeat(1,1,self.num_heads,1,self.num_points,1)
        # directly add pts and offset
        sampling_locations = reference_points + sampling_offsets
        
        sampling_locations = sampling_locations.view(
            bs, num_query*self.num_heads*self.num_levels*self.num_points, 3)
        
        return sampling_locations

    def feature_sampling(self, mlvl_feats, reference_points, pc_range, img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
        reference_points = reference_points.clone()
        reference_points_3d = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
        # reference_points (B, num_queries, 4)
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        B, num_query = reference_points.size()[:2]
        num_cam = lidar2img.size(1)
        reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
        reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
        eps = 1e-5
        mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        reference_points_cam = (reference_points_cam - 0.5) * 2
        mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                    & (reference_points_cam[..., 0:1] < 1.0) 
                    & (reference_points_cam[..., 1:2] > -1.0) 
                    & (reference_points_cam[..., 1:2] < 1.0))
        if self.num_points == 1:
            mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
        else:
            # mask [1, 6, 13696, 1]
            num_q = num_query // self.num_heads // self.num_levels // self.num_points
            mask = mask.view(B, num_cam, num_q, self.num_heads, self.num_levels, self.num_points)
            # [1, 6, 107, 8, 4, 4]
        mask = torch.nan_to_num(mask)
        sampled_feats = []
        # reference_points_cam: [1, 6, 13696, 2]
        # 
        for lvl, feat in enumerate(mlvl_feats):
            # feat = feat.permute(1,0,2,3,4)  # (N, B, C, H, W) -> (B, N, C, H, W)
            # B, N, C, H, W = feat.size()
            N, B, C, H, W = feat.size()
            feat = feat.view(B*N, C, H, W)
            if self.num_points == 1:
                # use same reference point in every levels
                reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
            else:
                
                reference_points_cam_lvl = reference_points_cam.view(
                    B*N, -1, self.num_heads, self.num_levels, self.num_points, 2)
                # get specific index level ref pts
                num_q = reference_points_cam_lvl.size(1)
                num_query = num_q * self.num_heads * self.num_points
                reference_points_cam_lvl = reference_points_cam_lvl[:,:,:,lvl,:,:].contiguous()
                reference_points_cam_lvl = reference_points_cam_lvl.view(B*N, num_query, 1, 2)
            
            sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
            sampled_feat = sampled_feat.view(B, N, C, num_query, 1)
            if self.num_points == 1:
                sampled_feat = sampled_feat.permute(0, 2, 3, 1, 4)
            else:
                sampled_feat = sampled_feat.view(B, N, C, -1, self.num_heads, self.num_points)
                # sampled_feat = sampled_feat.permute(0, 1, 3, 2)
            sampled_feats.append(sampled_feat)
        if self.num_points == 1:
            sampled_feats = torch.stack(sampled_feats, -1)
            sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
        else:
            # put level into -2 dim position
            sampled_feats = torch.stack(sampled_feats, -2)
            
            sampled_feats = sampled_feats.permute(0,1,3,2,4,5,6)
            # attention_weights: [1, 6, 107, 8, 4, 4]
        return reference_points_3d, sampled_feats, mask
        # reference_points_3d: [1, 13696, 3]
        # sampled_feats: [1, 256, 3424, 6, 1, 4]
        # [1, 1, 13696, 6, 1, 1]
        
        # sampled_feats: [1, 6, 107, 256, 8, 4, 4]