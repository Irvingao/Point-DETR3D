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
class RoiSelfCrossAttn(ImgRoiCrossAttn):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 roi_self_interact=False,
                 roi_cross_interact=False,
                 img_roi=True,
                 pts_roi=False,
                 **kwargs):
        super().__init__(
            **kwargs)
        
        
        self.roi_self_interact = roi_self_interact
        self.roi_cross_interact = roi_cross_interact
        
        self.img_roi = img_roi
        self.pts_roi = pts_roi
        if roi_cross_interact:
            assert img_roi and pts_roi, '`roi_cross_interact` only \
                enable when two modality ROI feature existing.'
        
        self.img_roi_attn = None
        self.pts_roi_attn = None
        if roi_self_interact:
            if img_roi:
                self.img_roi_attn = nn.MultiheadAttention(
                    kwargs['embed_dims'], kwargs['num_heads'],
                    kwargs['dropout'])
                    # **kwargs)
                self.img_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
            if pts_roi:
                self.pts_roi_attn = nn.MultiheadAttention(
                    # **kwargs)
                    kwargs['embed_dims'], kwargs['num_heads'],
                    kwargs['dropout'])
                    # kwargs['dropout'], **kwargs)
                self.pts_layer_norm = nn.LayerNorm(kwargs['embed_dims'])
        if roi_cross_interact:
            self.cs_roi_self_attn = nn.MultiheadAttention(
                    # **kwargs)
                    kwargs['embed_dims'], kwargs['num_heads'],
                    kwargs['dropout'])
                # kwargs['attn_drop'], **kwargs)
            self.roi_layer_norm = nn.LayerNorm(kwargs['embed_dims'])

        
        
        
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
        # ------------------ gt_box_3d在2d image上的投影框 ----------------------
        img_roi_feats = None
        roi_pts = None
        if 'gt_bboxes_3d' in kwargs and self.roi_attn:
            box2d_corners, box2d_cetners, cam_mask =  self.box3d2img2d(
                            kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                            kwargs['pc_range'], kwargs['reference_points'])
            
            cam_mask, drop_idx = self.rand_cam_mask(cam_mask)
            # 去除中心点没有投在image上的无效点
            box2d_corners, box2d_cetners, cam_mask = \
                self.drop_invalid_value(drop_idx, 
                    box2d_corners, box2d_cetners, cam_mask)
            
            # only use single scale of img feats
            img_roi_feats, roi_pts = self.box2imgRoi(
                kwargs['img_feats'][0], box2d_corners, 
                box2d_cetners, cam_mask, kwargs['img_metas'])

            # hard code for zero GT bugs
            if img_roi_feats.size(0) == 0: 
                img_roi_feats = query.new_zeros((1, img_roi_feats.size(1), img_roi_feats.size(2)))
            
            # 如果有 img_roi_feats,则使用，没有的话退化成self attn
            key = img_roi_feats
            value = key
            if self.key_pos_enable and roi_pts is not None:
                key_pos = self.roi_key_pos_encoder(roi_pts)
            
            '''
            # -------------------------- vis -----------------------------
            # from mmdet3d.core.visualizer.image_vis import plot_rect3d_on_img, draw_lidar_bbox3d_on_img
            from projects.mmdet3d_plugin.core.visualizer.image_vis import visualize_bevbox2d_corners, draw_lidar_bbox3d_on_img
            import cv2
            img = cv2.imread(kwargs['img_metas'][0]['filename'][0]) # front 
            box = kwargs['gt_bboxes_3d'][0]
            lidar2img_rt = kwargs['img_metas'][0]['lidar2img'][0]
            plot_img = draw_lidar_bbox3d_on_img(box,
                                                raw_img=img,
                                                lidar2img_rt=lidar2img_rt,
                                                img_metas=None,
                                                color=(0, 255, 0),
                                                thickness=1)
            cv2.imwrite("plot_img.jpg", plot_img)
            # ------------------------------------------------------------
            '''
        '''
        # 如果有 img_roi_feats,则使用，没有的话退化成self attn
        if 'img_roi_feats' in kwargs:
            key = kwargs['img_roi_feats']
            value = key
            roi_pts = kwargs['roi_pts']
            if self.key_pos_enable and roi_pts is not None:
                key_pos = self.roi_key_pos_encoder(roi_pts)
        '''
            
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            if key_pos.size(1) == key.size(1):
                key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        # query (num_query ,batch, embed_dims) 
        # key   (num_query ,batch, embed_dims)
        
        if self.roi_self_interact:
            if self.img_roi:
                key = key.permute(1,0,2)
                # input should be [num_q,bs,dims]
                key = self.img_roi_attn(
                    query=key, key=key,
                    value=key,attn_mask=None,
                    key_padding_mask=None)[0]
                key = key.permute(1,0,2)
        
        out = self.attn(
            query=query,    # torch.Size([4, 1, 256])
            key=key,        # torch.Size([1, 49, 256])
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def rand_cam_mask(self, cam_mask):
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
        zeros_indices = torch.cat([zeros_indices, zeros_indices.new_tensor(extral_idx[::-1])])
        
        # 此时的cam_mask都仅有一个相机的，或者没有投在任何一个相机上
        cam_mask = cam_mask.permute(0,2,1)
        
        return cam_mask, zeros_indices
    
    def drop_invalid_value(self, drop_idx, box2d_corners, box2d_cetners, cam_mask):
        if drop_idx.size(0) == 0:
            return box2d_corners, box2d_cetners, cam_mask
        valid_mask = torch.full_like(cam_mask[:,0], True)
        valid_mask[:,drop_idx] = False
        # filter
        box2d_corners = box2d_corners.permute(0,2,1,3,4)
        vaild_box2d_corners = box2d_corners[valid_mask]
        box2d_cetners = box2d_cetners.permute(0,2,1,3,4)
        vaild_box2d_cetners = box2d_cetners[valid_mask]
        cam_mask = cam_mask.permute(0,2,1)
        vaild_cam_mask = cam_mask[valid_mask]
        
        vaild_box2d_corners = vaild_box2d_corners.permute(1,0,2,3).unsqueeze(0)
        vaild_box2d_cetners = vaild_box2d_cetners.permute(1,0,2,3).unsqueeze(0)
        vaild_cam_mask = vaild_cam_mask.permute(1,0).unsqueeze(0)
        
        return vaild_box2d_corners, vaild_box2d_cetners, vaild_cam_mask
        
    
    def box2imgRoi(self, img_feats, box2d_corners, box2d_cetners, cam_mask, img_metas=None):
        assert box2d_corners.size(0) == 1
        bs, num_cam, num_box, _, _ = box2d_corners.shape
        
        
        # 根据cam_mask直接选出对应的cam
        sel_box2d_corners = box2d_corners[cam_mask]
        sel_box2d_cetners = box2d_cetners[cam_mask]
        '''
        # -------------------------- vis -----------------------------
        # 检查cam_mask后的center point
        from projects.mmdet3d_plugin.core.visualizer.image_vis import draw_pts_on_img, draw_lidar_bbox3d_on_img
        import cv2
        img = cv2.imread(img_metas[0]['filename'][0]) # front 
        # 归一化时需要使用pad后的 还原到img size
        sel_box2d = sel_box2d_cetners  # 取bs1
        sel_box2d = sel_box2d/2 +0.5
        sel_box2d = sel_box2d.reshape(-1, 2)
        sel_box2d[..., 0] *= img_metas[0]['img_shape'][0][1]
        sel_box2d[..., 1] *= img_metas[0]['img_shape'][0][0]
        plot_pts_img = draw_pts_on_img(img, sel_box2d, thickness=2, color=[0,255,0])
        cv2.imwrite("cam_sel_cent2d_img.jpg", plot_pts_img)
        # 2.
        new_img = cv2.imread(img_metas[0]['filename'][0]) # front 
        box2d_cet = box2d_cetners  # 取bs1
        box2d_cet = box2d_cet/2 +0.5
        box2d_cet = box2d_cet.reshape(-1, 2)
        box2d_cet[..., 0] *= img_metas[0]['img_shape'][0][1]
        box2d_cet[..., 1] *= img_metas[0]['img_shape'][0][0]
        new_plot_pts_img = draw_pts_on_img(new_img, box2d_cet, thickness=2, color=[0,255,0])
        cv2.imwrite("cam_cent2d_img.jpg", new_plot_pts_img)
        # -------------------------- vis -----------------------------
        '''
        
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
            
        '''
        # -------------------------- vis -----------------------------
        from projects.mmdet3d_plugin.core.visualizer.image_vis import draw_pts_on_img, draw_lidar_bbox3d_on_img
        import cv2
        img = cv2.imread(img_metas[0]['filename'][0]) # front 
        # 需要按照resize的尺度变换，而不是pad后的shape
        # r_h, r_w,_ = img.shape
        # f = img_metas[0]['img_resize_scale']
        # img = cv2.resize(img, (int(r_w*f), int(r_h*f)))
        imgfov_pts_2d = sampling_points.view(-1, 2)
        
        # 归一化时需要使用pad后的 还原到img size
        imgfov_pts_2d = imgfov_pts_2d / 2 + 0.5
        p_h, p_w,_ = img_metas[0]['img_shape'][0]
        imgfov_pts_2d[:,0] *= p_w
        imgfov_pts_2d[:,1] *= p_h
        plot_pts_img = draw_pts_on_img(img, imgfov_pts_2d, thickness=2, color=[255,255,0])
        cv2.imwrite("plot_pts_img.jpg", plot_pts_img)
        # -------------------------- vis -----------------------------
        '''
        
        # 将采样点坐标展平成形状为 [num_boxes, num_points*num_points, 2] 的张量
        sampling_points = sampling_points.view(num_box, -1, 2)
        sel_roi_pts = sampling_points.clone()
        
        sampling_points = sampling_points.unsqueeze(0).unsqueeze(-2)
        sampling_points = sampling_points.repeat(num_cam, 1,1,1,1)
        sampling_points = sampling_points.view(num_cam, num_box*num_points*num_points, 1, 2)
        
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
        
        return sel_sampled_feat, sel_roi_pts
    
    def box3d2img2d(self, gt_bboxes_3d, img_metas, pc_range, device_tensor):
        
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
        '''
        # -------------------------- vis -----------------------------
        from projects.mmdet3d_plugin.core.visualizer.image_vis import draw_pts_on_img, draw_lidar_bbox3d_on_img
        import cv2
        img = cv2.imread(img_metas[0]['filename'][0]) # front 
        cam_coords = cam_box3d_coords[0,0]  # 取bs1
        cam_coords = cam_coords.reshape(-1, 2)
        plot_pts_img = draw_pts_on_img(img, cam_coords, thickness=2, color=[0,255,0])
        cv2.imwrite("cam_box2d_img.jpg", plot_pts_img) # (√) 投影无误
        # -------------------------- vis -----------------------------
        '''
        
        # normlize to [0,1]
        cam_box3d_coords[..., 0] /= img_metas[0]['img_shape'][0][1]
        cam_box3d_coords[..., 1] /= img_metas[0]['img_shape'][0][0]
        # [-1, 1]
        cam_box3d_coords = (cam_box3d_coords - 0.5) * 2 
        
        # selcet cam
        box2d_corners_coords, box2d_cetners = torch.split(cam_box3d_coords, [8,1], dim=-2)
        
        '''
        # -------------------------- vis -----------------------------
        from projects.mmdet3d_plugin.core.visualizer.image_vis import draw_pts_on_img, draw_lidar_bbox3d_on_img
        import cv2
        new_img = cv2.imread(img_metas[0]['filename'][0]) # front 
        box2d_cet = box2d_cetners[0,0]  # 取bs1
        box2d_cet = box2d_cet/2 +0.5
        box2d_cet = box2d_cet.reshape(-1, 2)
        box2d_cet[..., 0] *= img_metas[0]['img_shape'][0][1]
        box2d_cet[..., 1] *= img_metas[0]['img_shape'][0][0]
        new_plot_pts_img = draw_pts_on_img(new_img, box2d_cet, thickness=2, color=[0,255,0])
        cv2.imwrite("cam_box2d_cet2_img.jpg", new_plot_pts_img) # (√) 投影无误
        # -------------------------- vis -----------------------------
        '''
        
        # sel_cam_mask = (sel_cam_mask & (box2d_cetners[..., 0:1] > 0.) 
        #             & (box2d_cetners[..., 0:1] < 1.0) 
        #             & (box2d_cetners[..., 1:2] > 0.) 
        #             & (box2d_cetners[..., 1:2] < 1.0))
        sel_cam_mask = (sel_cam_mask & (box2d_cetners[..., 0:1] > -1.0)
                    & (box2d_cetners[..., 0:1] < 1.0)
                    & (box2d_cetners[..., 1:2] > -1.0)
                    & (box2d_cetners[..., 1:2] < 1.0))
        sel_cam_mask = sel_cam_mask.squeeze(-1).squeeze(-1)
        # 获取中心点在6个cam上的哪个的mask
        sel_cam_mask = sel_cam_mask.permute(0,2,1)  # (bs, num_boxes, 6)
        
        return box2d_corners_coords, box2d_cetners, sel_cam_mask

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
        
        sel_roi_pts = sampling_points.clone().permute(1,0,2)
        
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
        sampled_feat = sampled_feat.permute(1,0,2) # [num_pts_feat, num_obj, dims]
        
        return sampled_feat, sel_roi_pts


@ATTENTION.register_module()
class RoiSelfCrossAttnV2(RoiSelfCrossAttn):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 force_mask=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.force_mask = force_mask
        
        if self.key_pos_enable:
            # pts
            self.roi_pts_key_pos_encoder = nn.Sequential(
                        nn.Linear(2, self.embed_dims),
                        nn.LayerNorm(self.embed_dims),
                        nn.ReLU(inplace=False),
                        nn.Linear(self.embed_dims, self.embed_dims),
                        nn.LayerNorm(self.embed_dims),
                    )
        
        self.batch_first = False
        assert self.batch_first == False

        
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
        if self.roi_attn:
            # ------------------------ img roi feature ----------------
            if self.img_roi:
                box2d_corners, box2d_cetners, cam_mask =  self.box3d2img2d(
                                # kwargs['gt_bboxes_3d'], kwargs['img_metas'],  # CAUTION: visulize only!!!!
                                kwargs['anchor_bboxes_3d'], kwargs['img_metas'],
                                kwargs['pc_range'], kwargs['reference_points'])

                # only use single scale of img feats
                img_roi_feats, roi_pts, cam_key_mask = self.box2imgRoi(
                    kwargs['img_feats'][0], box2d_corners, 
                    box2d_cetners, cam_mask, kwargs['img_metas'])

                # hard code for zero GT bugs
                if img_roi_feats.size(0) == 0: 
                    img_roi_feats = query.new_zeros((1, 
                                img_roi_feats.size(1), 
                                img_roi_feats.size(2)))
                
                key = img_roi_feats
                value = key
                if self.key_pos_enable and roi_pts is not None:
                    key_pos = self.roi_key_pos_encoder(roi_pts)
            
            # ------------------------ pts roi feature ----------------
            pts_key = None
            pts_key_pos = None
            if self.pts_roi:
                bev2d_corners = self.box3d2bev2d(
                    kwargs['anchor_bboxes_3d'], kwargs['img_metas'],
                    kwargs['pc_range'], kwargs['reference_points'])
            
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
                
                '''
                # -------------------------- vis -----------------------------
                from projects.mmdet3d_plugin.core.visualizer.image_vis import visualize_bevbox2d_corners, draw_lidar_bbox3d_on_img
                import cv2
                img = cv2.imread(kwargs['img_metas'][0]['filename'][0]) # front 
                box = kwargs['gt_bboxes_3d'][0]
                lidar2img_rt = kwargs['img_metas'][0]['lidar2img'][0]
                plot_img = draw_lidar_bbox3d_on_img(box,
                                                    raw_img=img,
                                                    lidar2img_rt=lidar2img_rt,
                                                    img_metas=None,
                                                    color=(0, 255, 0),
                                                    thickness=1)
                cv2.imwrite("plot_gt_lidar_img.jpg", plot_img)
                pred_box = kwargs['anchor_bboxes_3d'][0]
                pred_plot_img = draw_lidar_bbox3d_on_img(pred_box,
                                                    raw_img=img,
                                                    lidar2img_rt=lidar2img_rt,
                                                    img_metas=None,
                                                    color=(0, 255, 0),
                                                    thickness=1)
                cv2.imwrite("plot_pred_lidar_img.jpg", pred_plot_img)
                gt_bev2d_corners = self.box3d2bev2d(
                    kwargs['gt_bboxes_3d'], kwargs['img_metas'],
                    kwargs['pc_range'], kwargs['reference_points'])
                # ------------------------------------------------------------
                '''

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
        
        query = query.permute(1,0,2)
        # query [1, obj_dim, dim]
        # key [roi_dim, obj_dim, dim]
        # query (num_query ,batch, embed_dims) 
        # key   (num_query ,batch, embed_dims)
        
        key_int_mask = None
        if cam_key_mask is not None:
            
            key_valid_mask = cam_key_mask.to(torch.uint8)
            # key_mask中value==1的位置变成inf,最终attn接近0
            key_mask = ~cam_key_mask
            key_padding_mask = key_mask.to(torch.uint8)
            key_padding_mask = key_padding_mask.transpose(0, 1)
            if self.force_mask:
            # 让没有投在image上值都为0
                key = key_valid_mask[..., None] * key 
                key_padding_mask = None
        
        if self.roi_self_interact:
            if self.img_roi:
                res_key = key
                # 这里将 roi的feat作为query，进行交互，object作为bs维
                # [roi_dim, obj_dim, dim]
                key = self.img_roi_attn(
                    query=key, key=key,
                    value=key,attn_mask=None,
                    key_padding_mask=None)[0]
                key = self.img_layer_norm(key + res_key)
                
            if self.pts_roi:
                res_pts_key = pts_key
                # 这里将 roi的feat作为query，进行交互，object作为bs维
                # [roi_dim, obj_dim, dim]
                pts_key = self.pts_roi_attn(
                    query=pts_key, key=pts_key,
                    value=pts_key,attn_mask=None,
                    key_padding_mask=None)[0]
                pts_key = self.pts_layer_norm(pts_key + res_pts_key)
                
            if self.roi_cross_interact:
                
                key = self.cs_roi_self_attn(
                    query=pts_key, key=key,
                    value=key,attn_mask=None,
                    key_padding_mask=None)[0]
                    # key_padding_mask=key_padding_mask)[0]
                key = self.roi_layer_norm(pts_key+key)
                # 
                key_padding_mask = None
            else:
                if self.img_roi and self.pts_roi:
                    key = torch.cat([pts_key, key], dim=0)
                    key_padding_mask = None
                elif self.pts_roi:
                    key = pts_key
                    key_padding_mask = None
        else:
            if self.img_roi and self.pts_roi:
                key = torch.cat([pts_key, key], dim=0)
                key_padding_mask = None
                # raise ValueError("Here is not allowed.")
            elif self.pts_roi:
                key = pts_key
                key_padding_mask = None

        out = self.attn(
            query=query,    # (num_query ,batch, embed_dims)
            key=key,
            value=key,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        
        out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))
    
    def box2imgRoi(self, img_feats, box2d_corners, box2d_cetners, cam_mask, img_metas=None):
        assert box2d_corners.size(0) == 1
        bs, num_cam, num_box, _1, _2 = box2d_corners.shape
        
        sel_box2d_corners = box2d_corners
        sel_box2d_cetners = box2d_cetners
        # Hard Code： 当没有GT的特殊情况,此时唯一的tensor是给定的
        if num_box == 0:
            sel_box2d_corners = torch.zeros((bs, num_cam, num_box, 8, 2), 
                dtype=box2d_corners.dtype, device=box2d_corners.device)
            sel_box2d_cetners = torch.zeros((bs, num_cam, num_box, 1, 2), 
                dtype=box2d_cetners.dtype, device=box2d_cetners.device)
        
        '''
        # -------------------------- vis -----------------------------
        # 检查cam_mask后的center point
        from projects.mmdet3d_plugin.core.visualizer.image_vis import draw_pts_on_img, draw_lidar_bbox3d_on_img
        import cv2
        img = cv2.imread(img_metas[0]['filename'][0]) # front 
        # 归一化时需要使用pad后的 还原到img size
        sel_box2d = sel_box2d_cetners[0,0]  # 取bs1
        sel_box2d = sel_box2d/2 +0.5
        sel_box2d = sel_box2d.reshape(-1, 2)
        sel_box2d[..., 0] *= img_metas[0]['img_shape'][0][1]
        sel_box2d[..., 1] *= img_metas[0]['img_shape'][0][0]
        plot_pts_img = draw_pts_on_img(img, sel_box2d, thickness=2, color=[0,255,0])
        cv2.imwrite("cam_sel_cent2d_img.jpg", plot_pts_img) # (√) 投影无误
        # 2.
        new_img = cv2.imread(img_metas[0]['filename'][0]) # front 
        box2d_cet = box2d_cetners[0,0]  # 取bs1
        box2d_cet = box2d_cet/2 +0.5
        box2d_cet = box2d_cet.reshape(-1, 2)
        box2d_cet[..., 0] *= img_metas[0]['img_shape'][0][1]
        box2d_cet[..., 1] *= img_metas[0]['img_shape'][0][0]
        new_plot_pts_img = draw_pts_on_img(new_img, box2d_cet, thickness=2, color=[0,255,0])
        cv2.imwrite("cam_cent2d_img.jpg", new_plot_pts_img) # (√) 投影无误
        # -------------------------- vis -----------------------------
        '''
        
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
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bs, num_cam, 1,1,1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bs, num_cam, 1,1,1)    # torch.Size([1, 6, 107, 1, 1])
        # 根据box的中心坐标和宽高计算采样点坐标
        center_x = sel_box2d_cetners[..., :1]
        center_y = sel_box2d_cetners[..., 1:]
        sample_x = center_x.view(bs, num_cam, num_box, 1, 1) + grid_x * box2d_w.view(bs, num_cam, num_box, 1, 1)
        sample_y = center_y.view(bs, num_cam, num_box, 1, 1) + grid_y * box2d_h.view(bs, num_cam, num_box, 1, 1)
        # 组合采样点的x和y坐标
        sampling_points = torch.stack([sample_x, sample_y], dim=-1)  # [bs, num_cam, num_boxes, num_points, num_points, 2]
            
        # 将采样点坐标展平成形状为 [bs, num_cam, num_boxes, num_points*num_points, 2] 的张量
        sampling_points = sampling_points.view(bs, num_cam, num_box, -1, 2)
        sel_roi_pts = sampling_points.clone()
        
        sampling_points = sampling_points.view(bs*num_cam, num_box, -1, 2)
        
        '''
        # -------------------------- vis -----------------------------
        from projects.mmdet3d_plugin.core.visualizer.image_vis import draw_pts_on_img, draw_lidar_bbox3d_on_img
        import cv2
        lnew_img = cv2.imread(img_metas[0]['filename'][0]) # front 
        box2d_cet = sampling_points[0]  # 取bs1
        box2d_cet = box2d_cet/2 +0.5
        box2d_cet = box2d_cet.reshape(-1, 2)
        box2d_cet[..., 0] *= img_metas[0]['img_shape'][0][1]
        box2d_cet[..., 1] *= img_metas[0]['img_shape'][0][0]
        lnew_plot_pts_img = draw_pts_on_img(lnew_img, box2d_cet, thickness=2, color=[0,255,0])
        cv2.imwrite("cam_box2d_cet2_img.jpg", lnew_plot_pts_img) # (√) 投影无误
        # -------------------------- vis -----------------------------
        '''
        
        # TODO 获取完采样点，采样 参考attn
        N, B, C, H, W = img_feats.size()
        img_feats = img_feats.view(B*N, C, H, W)
        sampled_feat = F.grid_sample(img_feats, sampling_points)
        sampled_feat = sampled_feat.view(B, N, C, num_box, num_points*num_points, 1)
        sampled_feat = sampled_feat.squeeze(-1)
        sampled_feat = sampled_feat.permute(0, 1, 3, 2, 4)  # [1, 6, 107, 256, 49]
        
        sampled_feat = sampled_feat.squeeze(0)
        sampled_feat = sampled_feat.permute(0,3,1,2).contiguous()
        sampled_feat = sampled_feat.view(N*num_points*num_points, num_box, self.embed_dims)
        
        nan_mask = torch.isnan(cam_mask)
        cam_mask[nan_mask] = 0.
        cam_mask = cam_mask.squeeze(0)
        cam_mask = cam_mask.unsqueeze(-1).repeat(1, 1, num_points*num_points)
        cam_mask = cam_mask.permute(0,2,1).contiguous()
        cam_mask = cam_mask.view(N*num_points*num_points, num_box)
        
        sel_roi_pts = sel_roi_pts.squeeze(0)
        sel_roi_pts = sel_roi_pts.permute(0,2,1,3).contiguous()
        sel_roi_pts = sel_roi_pts.view(N*num_points*num_points, num_box, 2)
        
        return sampled_feat, sel_roi_pts, cam_mask  # [num_feat, num_obj, dims]