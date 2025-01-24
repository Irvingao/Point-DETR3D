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
from projects.mmdet3d_plugin.models.futr3d_utils.mm_Roi_attn import MMRoiAttn


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
class MMRoiAttnV2(MMRoiAttn):
    """使用ImgRoiCrossAttn版本的img ROI sampling + 
          RoiSelfCrossAttnV2 pts ROI sampling
          
        - 目前不支持 多模态的ROI Self Attn，因为img_roi采样没有对齐obejct个数，存在bug；
        - 目前只支持多模态sequence模式的 和query交互，concat也是因为img_roi没对齐问题。
    """

    def __init__(self,
                 query_align=True,
                 feat_fill_value=1.,
                 **kwargs):
        super().__init__(**kwargs)
        self.query_align = query_align
        self.feat_fill_value = feat_fill_value
        
        assert self.query_align
        assert self.img_norm == "V1"
        
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
            
        img_roi_feats = None
        roi_pts = None
        cam_key_mask = None
        # ------------------------ img roi feature ----------------
        if self.img_roi:
            box2d_corners, box2d_cetners, cam_mask =  self.box3d2img2d(
                            kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                            kwargs['pc_range'], kwargs['reference_points'])

            if self.query_align:
                # 获取无效query的mask，用来补齐对应位置的ROI feature
                drop_cam_mask, drop_idx = self.rand_cam_maskV2(cam_mask)
            else:
                drop_cam_mask, drop_idx = self.rand_cam_mask(cam_mask)
                
            # 去除中心点没有投在image上的无效点
            valid_box2d_corners, valid_box2d_cetners, vaild_cam_mask = \
                self.drop_invalid_value(drop_idx, 
                    box2d_corners, box2d_cetners, drop_cam_mask)
            
            # only use single scale of img feats
            img_roi_feats, roi_pts = self.box2imgRoi(
                kwargs['img_feats'][0], valid_box2d_corners, 
                valid_box2d_cetners, vaild_cam_mask, kwargs['img_metas'])
            
            if self.query_align:
                # hard code for zero GT bugs
                if img_roi_feats.size(0) == 0: 
                    img_roi_feats = query.new_zeros((query.size(0), 
                        img_roi_feats.size(1), img_roi_feats.size(2)))
                    roi_pts = query.new_zeros((query.size(0), 
                        roi_pts.size(1), roi_pts.size(2)))
                else:
                    img_roi_feats, roi_pts = self.align_feat_num(
                        query, img_roi_feats, roi_pts, drop_idx)
            
            # 如果有 img_roi_feats,则使用，没有的话退化成self attn
            key = img_roi_feats
            value = key
            if self.key_pos_enable and roi_pts is not None:
                if self.query_align:
                    key_pos = self.roi_key_pos_encoder(roi_pts)
                else:
                    key_pos = self.roi_key_pos_encoder(roi_pts)
            # key: torch.Size([36, 49, 256])
            # query: torch.Size([36, 1, 256])
        
        pts_key = None
        pts_key_pos = None
        # ------------------------ pts roi feature ----------------
        if self.pts_roi:
            bev2d_corners = self.box3d2bev2d(
                kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                kwargs['pc_range'], kwargs['reference_points'])
        
            # if self.drop_query:
                # 去除无效的query
                # bev2d_corners = self.drop_invalid_bev_center(bev2d_corners, drop_idx)
        
            pts_roi_feats, pts_roi_pts = self.box2bevRoi(
                kwargs['pts_feats'][0], bev2d_corners, 
                kwargs['img_metas'])
            
            
            # hard code for zero GT bugs
            if pts_roi_feats.size(0) == 0: 
                pts_roi_feats = query.new_zeros((1, 
                            pts_roi_feats.size(1), 
                            pts_roi_feats.size(2)))
            
            pts_key = pts_roi_feats
            if self.key_pos_enable and pts_roi_pts is not None:
                pts_key_pos = self.roi_pts_key_pos_encoder(pts_roi_pts)
        
        
        if key_pos is not None:
            if key_pos.size(1) == key.size(1):
                key = key + key_pos
        if pts_key_pos is not None:
            if key_pos.size(1) == key.size(1):
                pts_key = pts_key + pts_key_pos
        # query: torch.Size([36, 1, 256])
        # key: torch.Size([36, 49, 256])
        # pts_key: torch.Size([36, 49, 256])
        
        if self.img_roi:
            key = key.transpose(0, 1)   # 如果有img roi ，需要变换成 torch.Size([26, 49, 256]) 形状
        if self.pts_roi:
            pts_key = pts_key.transpose(0, 1)
        
        # ------------------------ ROI Self Attn ------------------------
        # img ROI 交互
        if self.img_self_attn:
            res_key = key
            # 这里将 roi的feat作为query，进行交互，object作为bs维
            # [roi_dim(49), obj_dim, dim]
            key = self.img_roi_attn(
                query=key, key=key,             # torch.Size([49, 26, 256])
                value=key,attn_mask=None,
                key_padding_mask=None)[0]
            key = self.img_layer_norm(key + res_key)
        
        # pts ROI 交互
        if self.pts_self_attn:
            res_pts_key = pts_key
            # 这里将 roi的feat作为query，进行交互，object作为bs维
            # [roi_dim, obj_dim, dim]
            pts_key = self.pts_roi_attn(
                query=pts_key, key=pts_key,
                value=pts_key,attn_mask=None,
                key_padding_mask=None)[0]
            pts_key = self.pts_layer_norm(pts_key + res_pts_key)
        
        # 跨模态 ROI 交互
        if self.modal_self_attn:
            key = key.transpose(0, 1)
            pts_key = pts_key.transpose(0, 1)
            if self.query_modal_first == 'img':
                res_mm_key = key
                mm_key = self.cs_roi_self_attn(
                    query=key, key=pts_key,
                    value=pts_key,attn_mask=None,
                    key_padding_mask=None)[0]
                key = self.roi_layer_norm(mm_key+res_mm_key)
            if self.query_modal_first == 'pts':
                res_mm_key = pts_key
                mm_key = self.cs_roi_self_attn(
                    query=pts_key, key=key,
                    value=key,attn_mask=None,
                    key_padding_mask=None)[0]
                pts_key = self.roi_layer_norm(mm_key+res_mm_key)
            key = key.transpose(0, 1)
            pts_key = pts_key.transpose(0, 1)
        # 不跨模态交互，直接拼接 / 
        else:
            # 退化为self attn
            if not self.img_roi and not self.pts_roi:
                key = query
            
        if self.pts_roi:
            query = query.transpose(0, 1)

        # query: torch.Size([1, 26, 256])
        # key: torch.Size([49, 26, 256])
        # pts_key: torch.Size([49, 26, 256])
        
        # ------------------------ ROI Cross Attn ------------------------
        if self.img_roi and self.pts_roi:
            if self.query_attn_type == 'concat':
                key = torch.cat([key, pts_key], dim=0)
                out = self.attn(
                    query=query,    # (num_query ,batch, embed_dims)
                    key=key,
                    value=key,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask)[0]
                
            elif self.query_attn_type == 'sequence':
                out = query
                if self.query_modal_first == 'img':
                    out = self.attn(
                            query=out,    # (num_query ,batch, embed_dims)
                            key=key,
                            value=key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
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
                    out = self.attn_2(
                            query=out,    # (num_query ,batch, embed_dims)
                            key=key,
                            value=key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
                else:
                    raise ValueError("not be here.")
            elif self.query_attn_type == 'fusion':
                pts_query = self.attn(
                            query=query,    # (num_query ,batch, embed_dims)
                            key=pts_key,
                            value=pts_key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
                img_query = self.attn_2(
                            query=query,    # (num_query ,batch, embed_dims)
                            key=key,
                            value=key,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)[0]
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
        
        aligned_img_roi_feats = img_roi_feats.new_full(
            [query.size(0), img_roi_feats.size(1), 
             img_roi_feats.size(2)], self.feat_fill_value)
        # fill
        aligned_img_roi_feats[valid_mask] = img_roi_feats
        
        aligned_roi_pts = roi_pts.new_full(
            [query.size(0), roi_pts.size(1), 
             roi_pts.size(2)], 0.)
        aligned_roi_pts[valid_mask] = roi_pts
        
        return aligned_img_roi_feats, aligned_roi_pts

