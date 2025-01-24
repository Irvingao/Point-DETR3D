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
class Deform3DCamCrossAttnV7(BaseModule):
    """An attention module used in Detr3d. 
        相对于V6，先去除offset，只用reference point复制n份，且使用 cam_attention_weights 来对多个cam的output 进行加权
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
                 batch_first=False,
                 fix_offset=False,
                 depth_encode=False,
                 uncertainty_fusion=False):
        super(Deform3DCamCrossAttnV7, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fix_offset = fix_offset
        self.depth_encode = depth_encode
        self.uncertainty_fusion = uncertainty_fusion

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
        self.multi_cam_attention_weights = nn.Linear(embed_dims,
                                           num_cams)    # √
        if self.uncertainty_fusion:
            self.uncertainty_weights = nn.Linear(embed_dims, 1)
        self.output_proj = nn.Linear(embed_dims, embed_dims)    # √
      
        self.position_encoder = nn.Sequential(
            nn.Linear(4 if self.depth_encode else 3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        # NOTE Deform Weights
        # TODO check if need to differentiate offset in different levels
        self.deform_cam_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 3) # √
        
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * self.num_points)    # √
        self.value_proj = nn.Linear(embed_dims, embed_dims) # √

        self.init_weight()

        if self.fix_offset:
            self.deform_cam_sampling_offsets.weight.requires_grad = False
            self.deform_cam_sampling_offsets.bias.requires_grad = False

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.multi_cam_attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

        constant_init(self.deform_cam_sampling_offsets, 0.)
        ### TODO initilize offset bias for DCN
        # print("NEED TO initialize DCN offset")
        # time.sleep(1)
            
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
        
        if self.uncertainty_fusion:
            constant_init(self.uncertainty_weights, val=0., bias=0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)

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
        # pc feat flatten
        feat_flatten = []
        spatial_shapes = []
        for feat in kwargs['img_feats']:
            feat = feat.permute(1,0,2,3,4)  # (N, B, C, H, W) -> (B, N, C, H, W)
            B, N, C, H, W = feat.size()
            # bs, n, c, h, w = feat.shape
            feat = feat.view(B*N, C, H, W)
            spatial_shape = (H, W)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        # value 
        key = feat_flatten
        key_padding_mask = kwargs['key_padding_mask']
        
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
        
        bs, num_query, _ = query.shape
        bsn, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        
        # cam fusion 
        # cam_attention_weights = self.multi_cam_attention_weights(query).view(
        #     bs, self.num_cams, num_query, 1)
        # c = self.multi_cam_attention_weights(query).view(
            # bs, self.num_cams, num_query, 1)
        # V5 change to: 这里直接用view和先变换维度后的结果并不相同
        # a = self.multi_cam_attention_weights(query).view(bs, self.num_cams, num_query)
        # b = self.multi_cam_attention_weights(query).permute(0,2,1)
        cam_attention_weights = self.multi_cam_attention_weights(query) # [1, 107, 6]
        cam_attention_weights = cam_attention_weights.permute(0,2,1).unsqueeze(-1)  # [1, 6, 107, 1]

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bsn, num_value, self.num_heads, -1)
        # offsets
        sampling_offsets = self.deform_cam_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 3)
        
        # repeat ref pts in `num_levels` dims,  -> [bs, num_query, num_levels, 3]
        reference_points = reference_points.unsqueeze(2).repeat(
                                        1,1,self.num_levels,1)
        
        # if reference_points.shape[-1] == 2:
        #     offset_normalizer = torch.stack(
        #         [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
        #     sampling_locations = reference_points[:, :, None, :, None, :] \
        #         + sampling_offsets \
        #         / offset_normalizer[None, None, None, :, None, :]
        if reference_points.shape[-1] == 3:   # 3d ref pts and offsets
            '''
            offset_normalizer = torch.stack(
                [sampling_offsets.new_tensor(self.pc_range[3]-self.pc_range[0]), 
                 sampling_offsets.new_tensor(self.pc_range[4]-self.pc_range[1]), 
                 sampling_offsets.new_tensor(self.pc_range[5]-self.pc_range[2])], -1)
            offset_edge = torch.stack(
                [sampling_offsets.new_tensor(self.pc_range[0]), 
                 sampling_offsets.new_tensor(self.pc_range[1]), 
                 sampling_offsets.new_tensor(self.pc_range[2])], -1)
            # reference_points为 [0,1]，因此需要将offset也归一化到[0,1]
            # offset = [ [-50,50] - (-50) ] / 100 = [0,1]
            sampling_offsets = (sampling_offsets - offset_edge)/offset_normalizer
                # / offset_normalizer[None, None, None, None, None, :]
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets
            '''
            # sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets
            
            # V7: 直接把 reference points 复制 num_pts份
            sampling_locations = reference_points.view(bs, num_query, 1, self.num_levels, 1, 3)
            sampling_locations = sampling_locations.repeat(1,1,self.num_heads,1,self.num_points,1)
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
            
        # get cam ref pts
        sampling_locations = sampling_locations.view(
            bs, num_query*self.num_heads*self.num_levels*self.num_points, 3)
        cam_sampling_locations, mask = self.get_cam_sampling_locations(
                            sampling_locations, kwargs['img_metas'])
        
        cam_sampling_locations = cam_sampling_locations.view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 3)
        mask = mask.view(bsn, num_query, self.num_heads, self.num_levels, self.num_points)
        
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        
        # repeat
        cam_sampling_locations = cam_sampling_locations.repeat(bsn,1,1,1,1,1)
        attention_weights = attention_weights.repeat(bsn,1,1,1,1)
        
        # filter outrange ref pts
        attention_weights = attention_weights * mask      
        
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, cam_sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, cam_sampling_locations, attention_weights)   # [bsn, num_q, 256]

        # fusion multi cam info into single query
        cam_attention_weights = cam_attention_weights.sigmoid()
        output = output.view(bs, self.num_cams, num_query, -1)
        # V6: 去除 cam_attention_weights
        # output = output * cam_attention_weights
        output = output.sum(1)
        
        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + inp_residual
    
    def get_cam_sampling_locations(self, reference_points, img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
        reference_points = reference_points.clone()
        reference_points_3d = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1]*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2]*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3]*(self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
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
                    & (reference_points_cam[..., 1:2] < 1.0))   # [1, 6, 13696, 1]
        
        # mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
        mask = torch.nan_to_num(mask)
        
        
        
        return reference_points_3d, mask



'''
def get_reference_points(reference_points, pc_range, img_metas):
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
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    # 在 weakly sup中，点肯定都是在image上的，因此mask应当全都有效。
    
    
    return reference_points_3d, mask
    
'''
    