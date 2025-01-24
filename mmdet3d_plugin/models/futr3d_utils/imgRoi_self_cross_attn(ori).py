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
                #  embed_dims,
                #  num_heads,
                #  roi_attn=False, 
                #  roi_size=7,
                #  key_pos_enable=False,
                 roi_self_interact=False,
                 roi_cross_interact=False,
                 img_roi=True,
                 pts_roi=False,
                #  dropout=0.,
                 **kwargs):
        # super(ImgRoiCrossAttn, self).__init__(
        super().__init__(
            # embed_dims=embed_dims,num_heads=num_heads,
            # roi_attn=roi_attn, roi_size=roi_size, 
            # key_pos_enable=key_pos_enable,
            # dropout=dropout, 
            **kwargs)
        
        # self.embed_dims = embed_dims
        # self.num_heads = num_heads
        
        self.roi_self_interact = roi_self_interact
        self.roi_cross_interact = roi_cross_interact
        self.img_roi = img_roi
        self.pts_roi = pts_roi
        
        self.img_roi_attn = None
        self.pts_roi_attn = None
        if roi_self_interact:
            if img_roi:
                self.img_roi_attn = nn.MultiheadAttention(
                    kwargs['embed_dims'], kwargs['num_heads'],
                    kwargs['dropout'])
                    # **kwargs)
            if pts_roi:
                self.pts_roi_attn = nn.MultiheadAttention(
                    # **kwargs)
                    kwargs['embed_dims'], kwargs['num_heads'],
                    kwargs['dropout'])
                    # kwargs['dropout'], **kwargs)
        if roi_cross_interact:
            self.cs_roi_self_attn = nn.MultiheadAttention(
                    # **kwargs)
                    kwargs['embed_dims'], kwargs['num_heads'],
                    kwargs['dropout'])
                # kwargs['attn_drop'], **kwargs)

        
        
        
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
            if query.size(0) == 1 and img_roi_feats.size(0) == 0: 
                img_roi_feats = query.new_zeros((1, img_roi_feats.size(1), img_roi_feats.size(2)))
            
            # 如果有 img_roi_feats,则使用，没有的话退化成self attn
            key = img_roi_feats
            value = key
            if self.key_pos_enable and roi_pts is not None:
                key_pos = self.roi_key_pos_encoder(roi_pts)
            
            '''
            # -------------------------- vis -----------------------------
            # from mmdet3d.core.visualizer.image_vis import plot_rect3d_on_img, draw_lidar_bbox3d_on_img
            import cv2
            img = cv2.imread(kwargs['img_metas'][0]['filename'][0]) # front 
            r_h, r_w,_ = kwargs['img_metas'][0]['img_shape'][0]
            img = cv2.resize(img, (r_w, r_h))
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
            query=query,    # (num_query ,batch, embed_dims)
            key=key,
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
        img = cv2.imread(img_metas[0]['filename'][0]) # front 
        r_h, r_w,_ = img_metas[0]['img_shape'][0]
        img = cv2.resize(img, (r_w, r_h))
        imgfov_pts_2d = sampling_points.view(-1, 2)
        # 
        imgfov_pts_2d = imgfov_pts_2d / 2 + 0.5
        imgfov_pts_2d[:,0] *= r_w
        imgfov_pts_2d[:,1] *= r_h
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
        
        # normlize to [0,1]
        cam_box3d_coords[..., 0] /= img_metas[0]['img_shape'][0][1]
        cam_box3d_coords[..., 1] /= img_metas[0]['img_shape'][0][0]
        cam_box3d_coords = (cam_box3d_coords - 0.5) * 2 # [-1, 1]
        
        # selcet cam
        box2d_corners_coords, box2d_cetners = torch.split(cam_box3d_coords, [8,1], dim=-2)
        
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

        
import cv2
def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    h,w,c = img.shape
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for idx, corner in enumerate(corners):
            corners[idx][0] = w if corner[0] > w else corner[0]
            corners[idx][0] = 0 if corner[0] < 0 else corner[0]
            corners[idx][1] = w if corner[1] > h else corner[1]
            corners[idx][1] = 0 if corner[1] < 0 else corner[1]
            
                
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)

def draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

def draw_pts_on_img(img, imgfov_pts_2d, thickness=3, color=[0,255,0]):
    if isinstance(imgfov_pts_2d, torch.Tensor):
        imgfov_pts_2d = imgfov_pts_2d.detach().cpu().numpy()
    
    h,w,c = img.shape
    for i in range(imgfov_pts_2d.shape[0]):
        if imgfov_pts_2d[i, 0] > w or imgfov_pts_2d[i, 0] < 0:
            continue
        if imgfov_pts_2d[i, 1] > h or imgfov_pts_2d[i, 1] < 0:
            continue
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=thickness,
            color=color,
            thickness=thickness,
        )
    return img
