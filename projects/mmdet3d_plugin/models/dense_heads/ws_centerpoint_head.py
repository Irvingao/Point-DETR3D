# Copyright (c) OpenMMLab. All rights reserved.
import copy
from copy import deepcopy

import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply

from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead, DCNSeparateHead

from projects.mmdet3d_plugin.datasets.nuscenes_utils.statistics_data import dict_wlh, gtlabels2names

@HEADS.register_module()
class WSCenterHead(BaseModule):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 point_guide=False,
                 point_size_factor=None,
                 obj_contrast=False,
                 feat_contrast=False,
                 loss_obj_contrast=None,
                 loss_feat_contrast=None,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(WSCenterHead, self).__init__(init_cfg=init_cfg)

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        
        # add point guide mask
        self.point_guide = point_guide
        self.point_size_factor = point_size_factor
        if self.point_size_factor is None:
            self.point_size_factor = 1
        # add contrastive loss
        self.obj_contrast = obj_contrast
        self.feat_contrast = feat_contrast
        self.loss_obj_contrast = None
        self.loss_feat_contrast = None
        if obj_contrast and loss_obj_contrast is not None:
            self.loss_obj_contrast = build_loss(loss_obj_contrast)
        if feat_contrast and loss_feat_contrast is not None:
            self.loss_feat_contrast = build_loss(loss_feat_contrast)
                
        self.pc_range = self.train_cfg['point_cloud_range']
        
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(builder.build_head(separate_head))

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats)
    
    def forward_singleV2(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []
        # obj_feats = []

        x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        ret_dicts.append(x)
        # obj_feats.append(x)
        
        return ret_dicts
        # return (ret_dicts, obj_feats)

    def forwardV2(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_singleV2, feats)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0)
                    ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def lossV2(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, pts_feats, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        
        preds_dicts, obj_feats = preds_dicts[:-1], preds_dicts[-1]
        
        # pts_feats = pts_feats[0]
        aligned_pts_feats, aligned_gt_bboxes_3d = self.align_aug_data(\
                        pts_feats[0], gt_bboxes_3d, kwargs['img_metas'])
        aligned_pts_feats = aligned_pts_feats.permute(0,2,3,1).contiguous() 
        aligned_pts_feats = aligned_pts_feats.view(aligned_pts_feats.size(0), -1, aligned_pts_feats.size(3))
        
        obj_feats = obj_feats[0].permute(0,2,3,1).contiguous()                  # torch.Size([8, 128, 128, 64])
        obj_feats = obj_feats.view(obj_feats.size(0), -1, obj_feats.size(3))    # torch.Size([8, 16384, 64])
        
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()   # torch.Size([8, 128, 128, 10])
            pred = pred.view(pred.size(0), -1, pred.size(3))                    # torch.Size([8, 16384, 10])
            pred = self._gather_feat(pred, ind)                                 # torch.Size([8, 500, 10])
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            '''
            # ------------------- add obj contrastive loss-----------------
            box_feat = self._gather_feat(obj_feats, ind)        # torch.Size([8, 500, 64])
            box_feat_v1, box_feat_v2 = torch.split(box_feat, box_feat.size(0) // 2)
            '''
        # ------------------- add dense feat contrastive loss-----------------
        if self.feat_contrast:
            pts_feats_v1, pts_feats_v2 = torch.split(
                aligned_pts_feats, aligned_pts_feats.size(0) // 2)
            num_feat = aligned_pts_feats.size(1)
            loss_feat_contrast = self.loss_feat_contrast(
                pts_feats_v1, pts_feats_v2,
                avg_factor=max(num_feat, 1))
            loss_dict[f'loss_feat_contrast'] = loss_feat_contrast
            
        return loss_dict
    
    @force_fp32(apply_to=('preds_dicts'))
    def lossV3(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, pts_feats, **kwargs):
        """Loss function for CenterHead.
        与ceterpointV2对应使用，输入的为align过的feat、pred和gt
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        
        preds_dicts_1, obj_feats_1 = preds_dicts[0][:-1], preds_dicts[0][-1]
        preds_dicts_2, obj_feats_2 = preds_dicts[1][:-1], preds_dicts[1][-1]
        obj_feats_1 = obj_feats_1[0].permute(0,2,3,1).contiguous()                  # torch.Size([4, 128, 128, 64])
        obj_feats_1 = obj_feats_1.view(obj_feats_1.size(0), -1, obj_feats_1.size(3))    # torch.Size([4, 16384, 64])
        obj_feats_2 = obj_feats_2[0].permute(0,2,3,1).contiguous()                  
        obj_feats_2 = obj_feats_2.view(obj_feats_2.size(0), -1, obj_feats_2.size(3)) 
        
        pts_feats_1, pts_feats_2 = pts_feats
        pts_feats_1 = pts_feats_1.permute(0,2,3,1).contiguous() 
        pts_feats_1 = pts_feats_1.view(pts_feats_1.size(0), -1, pts_feats_1.size(3))
        pts_feats_2 = pts_feats_2.permute(0,2,3,1).contiguous() 
        pts_feats_2 = pts_feats_2.view(pts_feats_2.size(0), -1, pts_feats_2.size(3))
        
        
        loss_dict = dict()
        for task_id, (preds_dict, preds_dict_diff) in enumerate(zip(preds_dicts_1, preds_dicts_2)):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()   # torch.Size([8, 128, 128, 10])
            pred = pred.view(pred.size(0), -1, pred.size(3))                    # torch.Size([8, 16384, 10])
            pred = self._gather_feat(pred, ind)                                 # torch.Size([8, 500, 10])
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            '''
            # ------------------- add obj contrastive loss-----------------
            box_feat = self._gather_feat(obj_feats, ind)        # torch.Size([8, 500, 64])
            box_feat_v1, box_feat_v2 = torch.split(box_feat, box_feat.size(0) // 2)
            '''
        # ------------------- add dense feat contrastive loss-----------------
        if self.feat_contrast:
            num_feat = pts_feats_1.size(1)
            loss_feat_contrast = self.loss_feat_contrast(
                pts_feats_1, pts_feats_2,
                avg_factor=max(num_feat, 1))
            loss_dict[f'loss_feat_contrast'] = loss_feat_contrast
            
        return loss_dict
    
    @force_fp32(apply_to=('preds_dicts'))
    def lossV4(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, aligned_pts_feats, **kwargs):
        """Loss function for CenterHead.
        与ceterpointV3对应使用，输入的为align过的feat, 以及默认的pred和gt
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        
        preds_dicts_1, obj_feats_1 = preds_dicts[0][:-1], preds_dicts[0][-1]
        preds_dicts_2, obj_feats_2 = preds_dicts[1][:-1], preds_dicts[1][-1]
        obj_feats_1 = obj_feats_1[0].permute(0,2,3,1).contiguous()                  # torch.Size([4, 128, 128, 64])
        obj_feats_1 = obj_feats_1.view(obj_feats_1.size(0), -1, obj_feats_1.size(3))    # torch.Size([4, 16384, 64])
        obj_feats_2 = obj_feats_2[0].permute(0,2,3,1).contiguous()                  
        obj_feats_2 = obj_feats_2.view(obj_feats_2.size(0), -1, obj_feats_2.size(3)) 
        
        pts_feats_1, pts_feats_2 = aligned_pts_feats
        pts_feats_1 = pts_feats_1.permute(0,2,3,1).contiguous() 
        pts_feats_1 = pts_feats_1.view(pts_feats_1.size(0), -1, pts_feats_1.size(3))
        pts_feats_2 = pts_feats_2.permute(0,2,3,1).contiguous() 
        pts_feats_2 = pts_feats_2.view(pts_feats_2.size(0), -1, pts_feats_2.size(3))
        
        loss_dict = dict()
        for task_id, (preds_dict, preds_dict_diff) in enumerate(zip(preds_dicts_1, preds_dicts_2)):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()   # torch.Size([8, 128, 128, 10])
            pred = pred.view(pred.size(0), -1, pred.size(3))                    # torch.Size([8, 16384, 10])
            pred = self._gather_feat(pred, ind)                                 # torch.Size([8, 500, 10])
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            ''''''
            # ------------------- add obj contrastive loss-----------------
            if self.obj_contrast:
                box_feat_1 = self._gather_feat(obj_feats_1, ind)        # torch.Size([8, 500, 64])
                box_feat_2 = self._gather_feat(obj_feats_2, ind)        # torch.Size([8, 500, 64])
                # obj_weights = mask[...,0:1].repeat(1,1,box_feat_1.size(-1))
                obj_weights = mask[...,0:1].squeeze(-1)
                loss_obj_contrast = self.loss_obj_contrast(
                    box_feat_1, box_feat_2, obj_weights, 
                    avg_factor=(num + 1e-4))
                loss_dict[f'task{task_id}.loss_obj_contrast'] = loss_obj_contrast
        # ------------------- add dense feat contrastive loss-----------------
        if self.feat_contrast:
            num_feat = pts_feats_1.size(1)
            loss_feat_contrast = self.loss_feat_contrast(
                pts_feats_1, pts_feats_2,
                avg_factor=max(num_feat, 1))
            loss_dict[f'loss_feat_contrast'] = loss_feat_contrast
            
        return loss_dict
    
    @force_fp32(apply_to=('preds_dicts'))
    def lossV5(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, aligned_pts_feats, **kwargs):
        """Loss function for CenterHead.
        增加 point-guide feature loss
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        
        preds_dicts_1, obj_feats_1 = preds_dicts[0][:-1], preds_dicts[0][-1]
        preds_dicts_2, obj_feats_2 = preds_dicts[1][:-1], preds_dicts[1][-1]
        obj_feats_1 = obj_feats_1[0].permute(0,2,3,1).contiguous()                  # torch.Size([4, 128, 128, 64])
        obj_feats_1 = obj_feats_1.view(obj_feats_1.size(0), -1, obj_feats_1.size(3))    # torch.Size([4, 16384, 64])
        obj_feats_2 = obj_feats_2[0].permute(0,2,3,1).contiguous()                  
        obj_feats_2 = obj_feats_2.view(obj_feats_2.size(0), -1, obj_feats_2.size(3)) 
        
        pts_feats_1, pts_feats_2 = aligned_pts_feats
        if self.point_guide:
            point_guide_mask = self.point_guide_mask(pts_feats_1, kwargs['aligned_gt_bboxes_3d'], gt_labels_3d) # [bs, 128, 128]
        pts_feats_1 = pts_feats_1.permute(0,2,3,1).contiguous() 
        pts_feats_1 = pts_feats_1.view(pts_feats_1.size(0), -1, pts_feats_1.size(3))
        pts_feats_2 = pts_feats_2.permute(0,2,3,1).contiguous() 
        pts_feats_2 = pts_feats_2.view(pts_feats_2.size(0), -1, pts_feats_2.size(3))
        
        loss_dict = dict()
        for task_id, (preds_dict, preds_dict_diff) in enumerate(zip(preds_dicts_1, preds_dicts_2)):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()   # torch.Size([8, 128, 128, 10])
            pred = pred.view(pred.size(0), -1, pred.size(3))                    # torch.Size([8, 16384, 10])
            pred = self._gather_feat(pred, ind)                                 # torch.Size([8, 500, 10])
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            ''''''
            # ------------------- add obj contrastive loss-----------------
            if self.obj_contrast:
                box_feat_1 = self._gather_feat(obj_feats_1, ind)        # torch.Size([8, 500, 64])
                box_feat_2 = self._gather_feat(obj_feats_2, ind)        # torch.Size([8, 500, 64])
                # obj_weights = mask[...,0:1].repeat(1,1,box_feat_1.size(-1))
                obj_weights = mask[...,0:1].squeeze(-1)
                
                box_feat_1 = F.normalize(box_feat_1, p=2, dim=-1)
                box_feat_2 = F.normalize(box_feat_2, p=2, dim=-1)
            
                loss_obj_contrast = self.loss_obj_contrast(
                    box_feat_1, box_feat_2, obj_weights, 
                    avg_factor=(num + 1e-4))
                loss_dict[f'task{task_id}.loss_obj_contrast'] = loss_obj_contrast
        
        # ------------------- add dense feat contrastive loss-----------------
        if self.feat_contrast:
            cl_weight = None
            num_feat = pts_feats_1.size(1)
            if self.point_guide:
                cl_weight = point_guide_mask.reshape(pts_feats_1.size(0), -1)
                num_feat = (num + 1e-4)
            
            pts_feats_1 = F.normalize(pts_feats_1, p=2, dim=-1)
            pts_feats_2 = F.normalize(pts_feats_2, p=2, dim=-1)
            
            loss_feat_contrast = self.loss_feat_contrast(
                pts_feats_1, pts_feats_2, weight=cl_weight, 
                avg_factor=num_feat)
            loss_dict[f'loss_feat_contrast'] = loss_feat_contrast
            
        return loss_dict
    
    def point_guide_mask(self, bev_feat, gt_bboxes_3d, gt_labels_3d):
        device = bev_feat.device
        
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']
        # feature_map_size = bev_feat.shape[2:]
        
        gt_bboxes_list = []
        for gt_bboxes, gt_labels in zip(gt_bboxes_3d, gt_labels_3d):
            points = gt_bboxes.gravity_center
            gt_labels = gt_labels.cpu().numpy().tolist() 
            boxes_w = torch.tensor([dict_wlh[gtlabels2names[label]][0] for label in gt_labels])[:, None]
            boxes_l = torch.tensor([dict_wlh[gtlabels2names[label]][2] for label in gt_labels])[:, None]
            boxes_h = torch.tensor([dict_wlh[gtlabels2names[label]][4] for label in gt_labels])[:, None]
            boxes = torch.cat([points, boxes_w, boxes_l, boxes_h], dim=-1).to(device)
            gt_bboxes_list.append(boxes)
        
        point_guide_map = bev_feat.new_zeros((len(gt_bboxes_list), feature_map_size[1], feature_map_size[0]))
        for idx in range(len(gt_bboxes_list)):
            num_objs = gt_bboxes_list[idx].shape[0]
            for k in range(num_objs):
                width = gt_bboxes_list[idx][k][3]
                length = gt_bboxes_list[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg['out_size_factor'] * self.point_size_factor
                length = length / voxel_size[1] / self.train_cfg['out_size_factor'] * self.point_size_factor

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = gt_bboxes_list[idx][k][0], gt_bboxes_list[idx][k][
                        1], gt_bboxes_list[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                        dtype=torch.float32,
                                        device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_heatmap_gaussian(point_guide_map[idx], center_int, radius)
            '''
            import matplotlib.pyplot as plt
            plt.title('Heatmap')
            # plt.plot([box[0] for box in gt_bboxes_3d[idx]], [box[1] for box in gt_bboxes_3d[idx]])
            plt.imshow(point_guide_map[idx].cpu().numpy())
            plt.savefig("Heatmap.png")
            plt.show()
            '''
    
        return point_guide_map  # [4, 128, 128]
    
    def align_aug_data(self, pts_feats, 
                         gt_bboxes_3d,
                         img_metas,
                         return_tensor=True,
                         interploate_mode='bilinear'):
        '''align weak teacher boxes to strong student boxes.
        '''
        # feature_map: [B,C,H,W]
        def horizontal_flip(feature_map):# 水平翻转
            return torch.flip(feature_map, [2])
        def vertical_flip(feature_map):# 垂直翻转
            return torch.flip(feature_map, [1])
        aligned_pts_feats = []
        aligned_gt_bboxes_3d = deepcopy(gt_bboxes_3d)
        for idx, (pts_feat, boxes_3d, img_meta) in enumerate(
            zip(pts_feats, aligned_gt_bboxes_3d, img_metas)):
            # aug顺序： RandomFlip3D, GlobalRotScaleTrans
            # align顺序需要倒过来
            
            # GlobalRotScaleTrans
            if 'pcd_trans' in img_meta:
                assert (img_meta['pcd_trans'] == 0.).all(), \
                    "Translation is not allowed in `GlobalRotScaleTrans`, [0.,0.,0.] only."
                # boxes_3d.translate(-img_meta['pcd_trans'])
            if 'pcd_rotation' in img_meta:
                '''TODO: chcek propety
                '''
                # 1. feat map
                pts_feat = pts_feat.unsqueeze(0)
                Rot = img_meta['pcd_rotation'].T    # [3,3]
                Rot = Rot[:2,:].unsqueeze(0)        # [B,2,3]
                grid = F.affine_grid(Rot, pts_feat.shape).to(pts_feat.device)   # # 仿射变换矩阵
                pts_feat = F.grid_sample(pts_feat, # 输入tensor，shape为[B,C,W,H]
                                    grid, # 上一步输出的gird,shape为[B,C,W,H]
                                    mode=interploate_mode)
                pts_feat = pts_feat[0]
                # pts_feats[idx] == pts_feat
                # assert (pts_feats[idx] == pts_feat).all(), 
                # 2. gt_boxes
                boxes_3d.rotate(img_meta['pcd_rotation'].T)
            if 'pcd_scale_factor' in img_meta:
                assert img_meta['pcd_scale_factor'] == 1., \
                    "Scale is not allowed in `GlobalRotScaleTrans`, `1.` only."
                # boxes_3d.scale(1/img_meta['pcd_scale_factor'])
            
            # RandomFlip3D
            if 'pcd_vertical_flip' in img_meta:
                if img_meta['pcd_vertical_flip']:
                    # 1. feat map
                    pts_feat = vertical_flip(pts_feat)
                    # 2. gt_boxes
                    boxes_3d.flip(bev_direction='vertical')
                    # pts_feats[idx] == pts_feat
                    # pts_feats[0] == pts_feat[0]
                    # pts_feats[0] == horizontal_flip(pts_feats)[0]
                    # pts_feats[0] == vertical_flip(pts_feat)[0]
            if 'pcd_horizontal_flip' in img_meta:
                if img_meta['pcd_horizontal_flip']:
                    # 1. feat map
                    pts_feat = horizontal_flip(pts_feat)
                    # 2. gt_boxes
                    boxes_3d.flip(bev_direction='horizontal')
                    # feat不全是一一对应，因为在卷积过程中，翻转后对应的位置不同，所以feat略有不同是正常现象
                    # pts_feat[0]  == horizontal_flip(pts_feat)[0]
                    
            aligned_pts_feats.append(pts_feat)
            aligned_gt_bboxes_3d[idx] = boxes_3d
        # aligned_gt_bboxes_3d[4].bev == aligned_gt_bboxes_3d[0].bev gt_bboxes_3d[4].bev gt_bboxes_3d[0].bev
        if return_tensor:
            aligned_pts_feats = torch.stack(aligned_pts_feats)
        # aligned_pts_feats[4][0] == aligned_pts_feats[1][0]
        # pts_feats[0][0] pts_feats[4][0]
        return aligned_pts_feats, aligned_gt_bboxes_3d
    
    def align_aug_dataV2(self, pts_feats, 
                         gt_bboxes_3d,
                         img_metas,
                         return_tensor=True,
                         interploate_mode='bilinear'):
        '''align weak teacher boxes to strong student boxes.
        第一版align box的时候直接使用box.flip，会导致错误，
        因为目前的box.tensor的格式为[x, y, z, x_size, y_size, z_size, yaw, vx, vy]，shape为[N, 9]
        所以对gt_box做flip需要分别处理x,y,yaw和vx,vy
        '''
        # feature_map: [B,C,H,W]
        def horizontal_flip(feature_map):# 水平翻转
            return torch.flip(feature_map, [2])
        def vertical_flip(feature_map):# 垂直翻转
            return torch.flip(feature_map, [1])
        def box_flip(boxes, bev_direction='horizontal'):
            assert bev_direction in ('horizontal', 'vertical')
            # rot_sine = boxes.tensor[..., 6:7]
            # rot_cosine = boxes.tensor[..., 7:8]
            # rot = torch.atan2(rot_sine, rot_cosine)
            if bev_direction == 'horizontal':
                boxes.tensor[:, 1] = -boxes.tensor[:, 1]    # y
                boxes.tensor[:, 6] = -boxes.tensor[:, 6] + np.pi 
                boxes.tensor[:, 7] = boxes.tensor[:, 7]    # vx
                boxes.tensor[:, 8] = -boxes.tensor[:, 8]    # vy
            elif bev_direction == 'vertical':
                boxes.tensor[:, 0] = -boxes.tensor[:, 0]    # x
                boxes.tensor[:, 6] = -boxes.tensor[:, 6]
                boxes.tensor[:, 7] = -boxes.tensor[:, 7]    # vx
                boxes.tensor[:, 8] = boxes.tensor[:, 8]    # vy
            return boxes
        aligned_pts_feats = []
        aligned_gt_bboxes_3d = deepcopy(gt_bboxes_3d)
        for idx, (pts_feat, boxes_3d, img_meta) in enumerate(
            zip(pts_feats, aligned_gt_bboxes_3d, img_metas)):
            # aug顺序： RandomFlip3D, GlobalRotScaleTrans(rot, scale, trans)
            # align顺序需要倒过来
            # ----------------------------------------------------
            pts_feat = pts_feat.unsqueeze(0)
            tgt_size = pts_feat.shape
            dev = pts_feat.device
            # GlobalRotScaleTrans
            if 'pcd_trans' in img_meta:
                # assert (img_meta['pcd_trans'] == 0.).all(), \
                #     "Translation is not allowed in `GlobalRotScaleTrans`, [0.,0.,0.] only."
                if not (img_meta['pcd_trans'] == 0.).all():
                    # 1. feat map
                    Trans = torch.zeros_like(img_meta['pcd_rotation'].T)    # [3,3]
                    Trans[0,0], Trans[1,1] = 1, 1
                    # 需要归一化`pcd_trans`,默认为pc_range
                    w, h = self.bbox_coder.pc_range
                    w, h = abs(w*2), abs(h*2)
                    Trans[0,2], Trans[1,2] = img_meta['pcd_trans'][0]/w, \
                                             img_meta['pcd_trans'][1]/h
                    
                    Trans = Trans[:2,:].unsqueeze(0)        # [B,2,3]
                    grid = F.affine_grid(Trans, tgt_size).to(dev)   # # 仿射变换矩阵
                    pts_feat = F.grid_sample(pts_feat, # 输入tensor，shape为[B,C,W,H]
                                        grid, # 上一步输出的gird,shape为[B,C,W,H]
                                        mode=interploate_mode)
                    # 2. gt_boxes
                    boxes_3d.translate(-img_meta['pcd_trans'])
            
            if 'pcd_scale_factor' in img_meta:
                # assert img_meta['pcd_scale_factor'] == 1., \
                    # "Scale is not allowed in `GlobalRotScaleTrans`, `1.` only."
                if img_meta['pcd_scale_factor'] != 1.:
                    # 1. feat map
                    Scl = torch.zeros_like(img_meta['pcd_rotation'].T)    # [3,3]
                    Scl[0,0], Scl[1,1] = 1/img_meta['pcd_scale_factor'], 1/img_meta['pcd_scale_factor']
                    Scl = Scl[:2,:].unsqueeze(0)        # [B,2,3]
                    grid = F.affine_grid(Scl, tgt_size).to(dev)   # # 仿射变换矩阵
                    pts_feat = F.grid_sample(pts_feat, # 输入tensor，shape为[B,C,W,H]
                                        grid, # 上一步输出的gird,shape为[B,C,W,H]
                                        mode=interploate_mode)
                    # 2. gt_boxes
                    boxes_3d.scale(1/img_meta['pcd_scale_factor'])
                    
            if 'pcd_rotation' in img_meta:
                '''TODO: chcek propety
                '''
                if img_meta['pcd_rotation'][0,0] != 1.:
                    # 1. feat map
                    Rot = img_meta['pcd_rotation'].T    # [3,3]
                    Rot = Rot[:2,:].unsqueeze(0)        # [B,2,3]
                    grid = F.affine_grid(Rot, tgt_size).to(dev)   # # 仿射变换矩阵
                    pts_feat = F.grid_sample(pts_feat, # 输入tensor，shape为[B,C,W,H]
                                        grid, # 上一步输出的gird,shape为[B,C,W,H]
                                        mode=interploate_mode)
                    # 2. gt_boxes
                    boxes_3d.rotate(img_meta['pcd_rotation'].T)
            pts_feat = pts_feat[0]
            
            # ----------------------------------------------------
            # RandomFlip3D
            if 'pcd_vertical_flip' in img_meta:
                if img_meta['pcd_vertical_flip']:
                    # 1. feat map
                    pts_feat = vertical_flip(pts_feat)
                    # 2. gt_boxes
                    # boxes_3d.flip(bev_direction='vertical')
                    boxes_3d = box_flip(boxes_3d, bev_direction='vertical')
                    # pts_feats[idx] == pts_feat
                    # pts_feats[0] == pts_feat[0]
                    # pts_feats[0] == horizontal_flip(pts_feats)[0]
                    # pts_feats[0] == vertical_flip(pts_feat)[0]
            if 'pcd_horizontal_flip' in img_meta:
                if img_meta['pcd_horizontal_flip']:
                    # 1. feat map
                    pts_feat = horizontal_flip(pts_feat)
                    # 2. gt_boxes
                    # boxes_3d.flip(bev_direction='horizontal')
                    boxes_3d = box_flip(boxes_3d, bev_direction='horizontal')
                    
                    # feat不全是一一对应，因为在卷积过程中，翻转后对应的位置不同，所以feat略有不同是正常现象
                    # pts_feat[0]  == horizontal_flip(pts_feat)[0]
                    
            aligned_pts_feats.append(pts_feat)
            aligned_gt_bboxes_3d[idx] = boxes_3d
        # aligned_gt_bboxes_3d[4].bev == aligned_gt_bboxes_3d[0].bev gt_bboxes_3d[4].bev gt_bboxes_3d[0].bev
        if return_tensor:
            aligned_pts_feats = torch.stack(aligned_pts_feats)
        # aligned_pts_feats[4][0] == aligned_pts_feats[1][0]
        # pts_feats[0][0] pts_feats[4][0]
        return aligned_pts_feats, aligned_gt_bboxes_3d
    
    def align_aug_dataV3(self, pts_feats, 
                         gt_bboxes_3d,
                         img_metas,
                         return_tensor=True,
                         interploate_mode='bilinear'):
        '''align weak teacher boxes to strong student boxes.
        第一版align box的时候直接使用box.flip，会导致错误，
        因为目前的box.tensor的格式为[x, y, z, x_size, y_size, z_size, yaw, vx, vy]，shape为[N, 9]
        所以对gt_box做flip需要分别处理x,y,yaw和vx,vy
        '''
        # feature_map: [B,C,H,W]
        def horizontal_flip(feature_map):# 水平翻转
            return torch.flip(feature_map, [2])
        def vertical_flip(feature_map):# 垂直翻转
            return torch.flip(feature_map, [1])
        def box_flip(boxes, bev_direction='horizontal'):
            assert bev_direction in ('horizontal', 'vertical')
            # rot_sine = boxes.tensor[..., 6:7]
            # rot_cosine = boxes.tensor[..., 7:8]
            # rot = torch.atan2(rot_sine, rot_cosine)
            if bev_direction == 'horizontal':
                boxes.tensor[:, 1] = -boxes.tensor[:, 1]    # y
                boxes.tensor[:, 6] = -boxes.tensor[:, 6] + np.pi 
                boxes.tensor[:, 7] = boxes.tensor[:, 7]    # vx
                boxes.tensor[:, 8] = -boxes.tensor[:, 8]    # vy
            elif bev_direction == 'vertical':
                boxes.tensor[:, 0] = -boxes.tensor[:, 0]    # x
                boxes.tensor[:, 6] = -boxes.tensor[:, 6]
                boxes.tensor[:, 7] = -boxes.tensor[:, 7]    # vx
                boxes.tensor[:, 8] = boxes.tensor[:, 8]    # vy
            return boxes
        aligned_pts_feats = []
        aligned_gt_bboxes_3d = deepcopy(gt_bboxes_3d)
        for idx, (pts_feat, boxes_3d, img_meta) in enumerate(
            zip(pts_feats, aligned_gt_bboxes_3d, img_metas)):
            # aug顺序： RandomFlip3D, GlobalRotScaleTrans
            # align顺序需要倒过来
            
            # GlobalRotScaleTrans
            if 'pcd_trans' in img_meta:
                assert (img_meta['pcd_trans'] == 0.).all(), \
                    "Translation is not allowed in `GlobalRotScaleTrans`, [0.,0.,0.] only."
                # boxes_3d.translate(-img_meta['pcd_trans'])
            if 'pcd_rotation' in img_meta:
                '''TODO: chcek propety
                '''
                # 1. feat map
                pts_feat = pts_feat.unsqueeze(0)
                Rot = img_meta['pcd_rotation'].T    # [3,3]
                Rot = Rot[:2,:].unsqueeze(0)        # [B,2,3]
                grid = F.affine_grid(Rot, tgt_size).to(pts_feat.device)   # # 仿射变换矩阵
                pts_feat = F.grid_sample(pts_feat, # 输入tensor，shape为[B,C,W,H]
                                    grid, # 上一步输出的gird,shape为[B,C,W,H]
                                    mode=interploate_mode)
                pts_feat = pts_feat[0]
                # pts_feats[idx] == pts_feat
                # assert (pts_feats[idx] == pts_feat).all(), 
                # 2. gt_boxes
                boxes_3d.rotate(img_meta['pcd_rotation'].T)
            if 'pcd_scale_factor' in img_meta:
                assert img_meta['pcd_scale_factor'] == 1., \
                    "Scale is not allowed in `GlobalRotScaleTrans`, `1.` only."
                # boxes_3d.scale(1/img_meta['pcd_scale_factor'])
            
            # RandomFlip3D
            if 'pcd_vertical_flip' in img_meta:
                if img_meta['pcd_vertical_flip']:
                    # 1. feat map
                    pts_feat = vertical_flip(pts_feat)
                    # 2. gt_boxes
                    # boxes_3d.flip(bev_direction='vertical')
                    boxes_3d = box_flip(boxes_3d, bev_direction='vertical')
                    # pts_feats[idx] == pts_feat
                    # pts_feats[0] == pts_feat[0]
                    # pts_feats[0] == horizontal_flip(pts_feats)[0]
                    # pts_feats[0] == vertical_flip(pts_feat)[0]
            if 'pcd_horizontal_flip' in img_meta:
                if img_meta['pcd_horizontal_flip']:
                    # 1. feat map
                    pts_feat = horizontal_flip(pts_feat)
                    # 2. gt_boxes
                    # boxes_3d.flip(bev_direction='horizontal')
                    boxes_3d = box_flip(boxes_3d, bev_direction='horizontal')
                    
                    # feat不全是一一对应，因为在卷积过程中，翻转后对应的位置不同，所以feat略有不同是正常现象
                    # pts_feat[0]  == horizontal_flip(pts_feat)[0]
                    
            aligned_pts_feats.append(pts_feat)
            aligned_gt_bboxes_3d[idx] = boxes_3d
        # aligned_gt_bboxes_3d[4].bev == aligned_gt_bboxes_3d[0].bev gt_bboxes_3d[4].bev gt_bboxes_3d[0].bev
        if return_tensor:
            aligned_pts_feats = torch.stack(aligned_pts_feats)
        # aligned_pts_feats[4][0] == aligned_pts_feats[1][0]
        # pts_feats[0][0] pts_feats[4][0]
        return aligned_pts_feats, aligned_gt_bboxes_3d
    
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the \
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the \
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the \
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # Apply NMS in birdeye view

            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_gpu(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_maxsize=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
