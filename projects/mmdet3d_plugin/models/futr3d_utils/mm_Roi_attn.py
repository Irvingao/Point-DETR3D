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
class MMRoiAttn(ImgRoiCrossAttn):
    """使用ImgRoiCrossAttn版本的img ROI sampling + 
          RoiSelfCrossAttnV2 pts ROI sampling
          
        - 目前不支持 多模态的ROI Self Attn，因为img_roi采样没有对齐obejct个数，存在bug；
        - 目前只支持多模态sequence模式的 和query交互，concat也是因为img_roi没对齐问题。
    """

    def __init__(self,
                 img_level=0,
                 img_roi=False,
                 img_self_attn=False,
                 pts_level=0,
                 pts_roi=False,
                 pts_self_attn=False,
                 modal_self_attn=False,
                 query_attn_type="sequence",
                 query_modal_first="pts",
                 transpose_pts=False,
                 img_norm="V1",
                 **kwargs):
        super().__init__(**kwargs)
        
        assert self.batch_first == False
        
        self.img_level = img_level
        self.pts_level = pts_level
        self.img_roi = img_roi        
        self.pts_roi = pts_roi
        self.img_self_attn = img_self_attn
        self.pts_self_attn = pts_self_attn
        self.modal_self_attn = modal_self_attn
        
        # assert self.modal_self_attn == False
        
        self.query_attn_type = query_attn_type
        self.query_modal_first = query_modal_first
        self.transpose_pts = transpose_pts
        self.img_norm = img_norm
        
        assert query_attn_type in ['sequence', 'concat', 'fusion'], \
            "The arguments `query_attn_type` in MMRoiAttn \
            is only supported in ['sequence', 'concat', 'fusion']"
        if query_attn_type == 'concat':
                raise ValueError("`query_attn_type` is not \
                    supported as 'concat'")
        elif query_attn_type == 'fusion':
            assert self.img_roi and self.pts_roi, ""
            fused_embed = self.embed_dims * 2
            self.modality_fusion_layer = nn.Sequential(
                nn.Linear(fused_embed, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=False),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )
        
        assert query_modal_first in ['img', 'pts'], \
            "The arguments `query_modal_first` in MMRoiAttn \
            is only supported in ['img', 'pts']"
        
        assert img_norm in ['V1', 'V2'], \
            "The arguments `img_norm` in MMRoiAttn \
            is only supported in ['V1', 'V2']"
        
        # 在V3 中， roi_attn 弃用
        if self.roi_attn:
            self.roi_attn = False
            warnings.warn(
                'The arguments `roi_attn` in MMRoiAttn '
                'has been deprecated, now you can separately '
                'set `img_roi`(img modality), pts_roi(lidar '
                'modality), ', DeprecationWarning)

        # img ROI attn
        if self.img_self_attn:
            assert self.img_roi
            self.img_roi_attn = nn.MultiheadAttention(
                kwargs['embed_dims'], kwargs['num_heads'],
                kwargs['dropout'])
            self.img_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
        
        # pts ROI attn
        if self.pts_self_attn:
            assert self.pts_roi
            self.pts_roi_attn = nn.MultiheadAttention(
                kwargs['embed_dims'], kwargs['num_heads'],
                kwargs['dropout'])
            self.pts_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
        
        # multi-modal ROI attn
        if self.modal_self_attn:
            assert self.img_roi and self.pts_roi, "The arguments \
                `modal_self_attn` only turn on when  `img_roi` \
                and `pts_roi` are turning on as `True`."
        
            self.cs_roi_self_attn = nn.MultiheadAttention(
                kwargs['embed_dims'], kwargs['num_heads'],
                kwargs['dropout'])
            self.roi_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
        
        if self.img_roi and self.pts_roi and \
                self.query_attn_type in ['sequence', 'fusion']:
            self.attn_2 = nn.MultiheadAttention(
                kwargs['embed_dims'], kwargs['num_heads'],
                kwargs['dropout'])
        
        if self.key_pos_enable:
            # extra pts_position
            self.roi_pts_key_pos_encoder = nn.Sequential(
                        nn.Linear(2, self.embed_dims),
                        nn.LayerNorm(self.embed_dims),
                        nn.ReLU(inplace=False),
                        nn.Linear(self.embed_dims, self.embed_dims),
                        nn.LayerNorm(self.embed_dims),
                    )
        
        
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
        img_roi_feats = None
        roi_pts = None
        cam_key_mask = None
        # ------------------------ img roi feature ----------------
        if self.img_roi:
            if self.img_norm == "V1":
                box2d_corners, box2d_cetners, cam_mask =  self.box3d2img2d(
                                kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                                kwargs['pc_range'], kwargs['reference_points'])
            elif self.img_norm == "V2":
                box2d_corners, box2d_cetners, cam_mask =  self.box3d2img2dV2(
                                kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                                kwargs['pc_range'], kwargs['reference_points'])
            
            cam_mask, drop_idx = self.rand_cam_mask(cam_mask)
            # 去除中心点没有投在image上的无效点
            box2d_corners, box2d_cetners, cam_mask = \
                self.drop_invalid_value(drop_idx, 
                    box2d_corners, box2d_cetners, cam_mask)
            
            # only use single scale of img feats
            img_roi_feats, roi_pts = self.box2imgRoi(
                kwargs['img_feats'][self.img_level], box2d_corners, 
                box2d_cetners, cam_mask, kwargs['img_metas'])
            
            # hard code for zero GT bugs
            if img_roi_feats.size(0) == 0: 
                img_roi_feats = query.new_zeros((1, img_roi_feats.size(1), img_roi_feats.size(2)))
            
            # 如果有 img_roi_feats,则使用，没有的话退化成self attn
            key = img_roi_feats
            value = key
            if self.key_pos_enable and roi_pts is not None:
                key_pos = self.roi_key_pos_encoder(roi_pts)
        
        pts_key = None
        pts_key_pos = None
        # ------------------------ pts roi feature ----------------
        if self.pts_roi:
            bev2d_corners = self.box3d2bev2d(
                kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                kwargs['pc_range'], kwargs['reference_points'])
        
            pts_roi_feats, pts_roi_pts = self.box2bevRoi(
                kwargs['pts_feats'][self.pts_level], bev2d_corners, 
                kwargs['img_metas'])
            # hard code for zero GT bugs
            if pts_roi_feats.size(0) == 0: 
                pts_roi_feats = query.new_zeros((1, 
                            pts_roi_feats.size(1), 
                            pts_roi_feats.size(2)))
            
            pts_key = pts_roi_feats
            if self.key_pos_enable and pts_roi_pts is not None:
                pts_key_pos = self.roi_pts_key_pos_encoder(pts_roi_pts)
        
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            if key_pos.size(1) == key.size(1):
                key = key + key_pos
        if pts_key_pos is not None:
            if key_pos.size(1) == key.size(1):
                pts_key = pts_key + pts_key_pos
        
        # ------------------------ ROI Self Attn ------------------------
        # img ROI 交互
        if self.img_self_attn:
            key = key.transpose(0, 1)
            res_key = key
            # 这里将 roi的feat作为query，进行交互，object作为bs维
            # [roi_dim, obj_dim, dim]
            key = self.img_roi_attn(
                query=key, key=key,             # torch.Size([49, 26, 256])
                value=key,attn_mask=None,
                key_padding_mask=None)[0]
            key = self.img_layer_norm(key + res_key)
            key = key.transpose(0, 1)
        
        # pts ROI 交互
        if self.pts_self_attn:
            pts_key = pts_key.transpose(0, 1)
            res_pts_key = pts_key
            # 这里将 roi的feat作为query，进行交互，object作为bs维
            # [roi_dim, obj_dim, dim]
            pts_key = self.pts_roi_attn(
                query=pts_key, key=pts_key,
                value=pts_key,attn_mask=None,
                key_padding_mask=None)[0]
            pts_key = self.pts_layer_norm(pts_key + res_pts_key)
            pts_key = pts_key.transpose(0, 1)
        
        # 3种功能：
        # 1. 跨模态交互， 得到key；
        # 2. 
        
        # 跨模态 ROI 交互
        if self.modal_self_attn:
            if self.query_modal_first == 'img':
                res_mm_key = key
                mm_key = self.cs_roi_self_attn(
                    query=key, key=pts_key,
                    value=pts_key,attn_mask=None,
                    key_padding_mask=None)[0]
            if self.query_modal_first == 'pts':
                res_mm_key = pts_key
                mm_key = self.cs_roi_self_attn(
                    query=pts_key, key=key,
                    value=key,attn_mask=None,
                    key_padding_mask=None)[0]
            key = self.roi_layer_norm(mm_key+res_mm_key)
        # 不跨模态交互，直接拼接 / 
        else:
            # 退化为self attn
            if not self.img_roi and not self.pts_roi:
                key = query
            elif self.img_roi:
                pass
            elif self.pts_roi:
                key = pts_key
            else:
                # key 直接concat 多个 ROI
                # ['sequence', 'concat']
                if self.query_attn_type == 'concat':
                    raise ValueError("`query_attn_type` is not \
                        supported as 'sequence'")
                    key = torch.cat([pts_key, key], dim=0)
                elif self.query_attn_type == 'sequence':
                    pass
        
        # if self.img_roi:
            # key = key.transpose(0, 1)   # 如果有img roi ，需要变换成 torch.Size([26, 49, 256]) 形状
        
        if self.pts_roi:
            query = query.transpose(0, 1)
            # key = key.transpose(0, 1)
            # query [1, obj_dim, dim]
            # key [roi_dim, obj_dim, dim]
            # query (num_query ,batch, embed_dims) 
            # key   (num_query ,batch, embed_dims)

        # img query cross attn:
        # query=query,    # torch.Size([26, 1, 256])
        # key=key,        # torch.Size([26, 49, 256])
        
        # ------------------------ ROI Cross Attn ------------------------
        if self.img_roi and self.pts_roi:
            if self.transpose_pts:
                pts_key = pts_key.transpose(0, 1)   # torch.Size([49, 26, 256])
            if self.query_attn_type == 'concat':
                raise ValueError("not be here.")
                out = self.attn(
                    query=query,    # (num_query ,batch, embed_dims)
                    key=key,
                    value=key,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask)[0]
            elif self.query_attn_type == 'sequence':
                out = query
                # key = key.transpose(0, 1)   # TODO: 这里的img_roi 采用object作为bs维度会报错，所以先跑通
                if self.query_modal_first == 'img':
                    out = out.transpose(0, 1)
                    out = self.attn(
                            query=out,    # (num_query ,batch, embed_dims)
                            key=key,
                            value=key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
                    out = out.transpose(0, 1)
                    out = self.attn_2(
                            query=out,    # (num_query ,batch, embed_dims)
                            key=pts_key,
                            value=pts_key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
                elif self.query_modal_first == 'pts':
                    out = self.attn(
                            query=out,    # (num_query ,batch, embed_dims)
                            key=pts_key,
                            value=pts_key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
                    # img将bs维作为第一维，num_obj作为0维
                    out = out.transpose(0, 1)
                    out = self.attn_2(
                            query=out,    # (num_query ,batch, embed_dims)
                            key=key,
                            value=key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
                    out = out.transpose(0, 1)
                else:
                    raise ValueError("not be here.")
            elif self.query_attn_type == 'fusion':
                pts_query = self.attn(
                            query=query,    # (num_query ,batch, embed_dims)
                            key=pts_key,
                            value=pts_key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
                query = query.transpose(0, 1)
                img_query = self.attn_2(
                            query=query,    # (num_query ,batch, embed_dims)
                            key=key,
                            value=key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
                img_query = img_query.transpose(0, 1)
                fusion_query = torch.cat([pts_query, img_query], dim=-1)
                out = self.modality_fusion_layer(fusion_query)
        else:
            # query self attention
            out = self.attn(
                    query=query,    # (num_query ,batch, embed_dims)
                    key=key,
                    value=key,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask)[0]
            
            
        if self.pts_roi:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))
    
    
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
        
    def box2bevRoi(self, pts_feats, bevbox2d_corners, img_metas=None):
        '''
        bevbox2d_corners: (bs, num_box, 5), XYWHR format, value range: [-1,1]
        '''
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
        
        num_points = self.roi_size
        # 生成网格坐标
        x = torch.linspace(-0.5, 0.5, num_points)
        y = torch.linspace(-0.5, 0.5, num_points)
        grid_x, grid_y = torch.meshgrid(x, y)  # 形状为 [num_points, num_points]
        # 将网格坐标扩展为每个box的采样点坐标
        grid_x = grid_x.view(1, num_points, num_points).expand(num_box, -1, -1).to(pts_feats.device)
        grid_y = grid_y.view(1, num_points, num_points).expand(num_box, -1, -1).to(pts_feats.device)
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
        
        # TODO 获取完采样点，采样 参考attn
        B, C, H, W = pts_feats.size()
        sampled_feat = F.grid_sample(pts_feats, sampling_points)
        sampled_feat = sampled_feat.view(B, C, num_box, num_points*num_points, 1)
        sampled_feat = sampled_feat.squeeze(-1)
        sampled_feat = sampled_feat.permute(0, 2, 3, 1)  # torch.Size([1, 26, 49, 256])
        
        assert sampled_feat.size(0) == 1, 'only support bs 1.'
        sampled_feat = sampled_feat[0]  # [num_obj, num_pts_feat, dims]
        # sampled_feat = sampled_feat.permute(1,0,2) # [num_pts_feat, num_obj, dims]
        
        return sampled_feat, sel_roi_pts
    
    def box3d2img2dV2(self, gt_bboxes_3d, img_metas, pc_range, device_tensor):
        
        assert len(gt_bboxes_3d) == 1, f"batch size == 1 is supported, but currently {len(gt_bboxes_3d)}"
        
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = device_tensor.new_tensor(lidar2img) # (B, N, 4, 4)
        
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
                            ).to(device_tensor.device) # (B, N, 4, 4)
        box3d_centers = torch.stack(box3d_centers, dim=0
                            ).to(device_tensor.device)
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
        
        # get selcet camera mask
        cam_box3d_coords = cam_box3d_coords.view(bs, num_cam, num_box, num_corner, 4)
        sel_cam_mask = (cam_box3d_coords[..., -1:, 2:3] > eps)
        
        # normlize depth
        cam_box3d_coords = cam_box3d_coords[..., 0:2] / torch.max(
            cam_box3d_coords[..., 2:3], torch.ones_like(cam_box3d_coords[..., 2:3])*eps)
        
        # normlize to [0,1]
        cam_box3d_coords[..., 0] /= img_metas[0]['img_shape'][0][1]
        cam_box3d_coords[..., 1] /= img_metas[0]['img_shape'][0][0]
        # cam_box3d_coords = (cam_box3d_coords - 0.5) * 2 # [-1, 1]
        
        # selcet cam
        box2d_corners_coords, box2d_cetners = torch.split(cam_box3d_coords, [8,1], dim=-2)
        
        sel_cam_mask = (sel_cam_mask & (box2d_cetners[..., 0:1] > 0.) 
                    & (box2d_cetners[..., 0:1] < 1.0) 
                    & (box2d_cetners[..., 1:2] > 0.) 
                    & (box2d_cetners[..., 1:2] < 1.0))
        # sel_cam_mask = (sel_cam_mask & (box2d_cetners[..., 0:1] > -1.0)
        #             & (box2d_cetners[..., 0:1] < 1.0)
        #             & (box2d_cetners[..., 1:2] > -1.0)
        #             & (box2d_cetners[..., 1:2] < 1.0))
        sel_cam_mask = sel_cam_mask.squeeze(-1).squeeze(-1)
        # 获取中心点在6个cam上的哪个的mask
        sel_cam_mask = sel_cam_mask.permute(0,2,1)  # (bs, num_boxes, 6)
        
        return box2d_corners_coords, box2d_cetners, sel_cam_mask
