# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import functional as F
from copy import deepcopy

from mmdet.models import DETECTORS
from mmdet3d.models import builder
from projects.mmdet3d_plugin.utils.structure_utils import dict_split, weighted_loss

from mmdet3d.models.detectors import CenterPoint

@DETECTORS.register_module()
class WSStudentCenterPointV3(CenterPoint):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 unsup_weight=1.0,
                 warmup_iter=0,
                 loss_type="V4",
                 **kwargs):
        super(WSStudentCenterPointV3, self).__init__(
            **kwargs)
        self.unsup_weight = unsup_weight
        self.warmup_iter = warmup_iter
        self.loss_type = loss_type
        assert loss_type in ['V4', 'V5'], "loss_type only support as 'V4' or 'V5'"
        self.iter = 1

    def combine_student_data(self, data_1, data_2):
        '''combine sup and unsup data for student model
        '''
        assert isinstance(data_1, dict) and \
            isinstance(data_2, dict)
            
        new_student_data = deepcopy(data_1)
        keys = data_1.keys()
        for key in keys:
            new_student_data[key] = data_1[key] + \
                                    data_2[key]
        return new_student_data


    def forward_train(self, data, **kwargs):
        '''
        - V2是只对view_1反传,view_2 infer无梯度;同时将pts_feat
        - V3是用默认的pts_feat过head,
        '''
        # TODO: 
        
        data_1, data_2 = data
        
        data_all = self.combine_student_data(data_1, data_2)
        
        gt_bboxes_3d = data_all['gt_bboxes_3d']
        gt_labels_3d = data_all['gt_labels_3d']
        img_metas = data_all['img_metas']
        
        # backbone forward
        # view_1
        _, pts_feats_1 = self.extract_feat(points=data_1['points'], 
                            img=None, img_metas=data_1['img_metas'])
        # view_2
        with torch.no_grad():
            _, pts_feats_2 = self.extract_feat(points=data_2['points'], 
                                img=None, img_metas=data_2['img_metas'])
        
        # align aug for pts_feat
        pts_feats = [feat for feat in pts_feats_1[0]] + [feat for feat in pts_feats_2[0]]
        # aligned_pts_feats, aligned_gt_bboxes_3d = self.pts_bbox_head.align_aug_data(\
                        # pts_feats, gt_bboxes_3d, img_metas, return_tensor=False)
        aligned_pts_feats, aligned_gt_bboxes_3d = self.pts_bbox_head.align_aug_dataV2(\
                        pts_feats, gt_bboxes_3d, img_metas, return_tensor=False)
        num = len(aligned_pts_feats)//2
        aligned_pts_feats_1 = torch.stack(aligned_pts_feats[:num])  # torch.Size([bs, 384, 128, 128])
        aligned_pts_feats_2 = torch.stack(aligned_pts_feats[num:])
        aligned_gt_bboxes_3d = aligned_gt_bboxes_3d[:num]
        
        # head forward
        # view_1
        outs_1 = self.pts_bbox_head.forwardV2(pts_feats_1)
        # view_2
        with torch.no_grad():
            outs_2 = self.pts_bbox_head.forwardV2(pts_feats_2)
            
            # aligned_outs_1 = self.pts_bbox_head.forwardV2([aligned_pts_feats_1])
            # aligned_outs_2 = self.pts_bbox_head.forwardV2([aligned_pts_feats_2])
            
        
        losses = dict()

        
        if self.loss_type == 'V4':
            loss_inputs = [data_1['gt_bboxes_3d'], data_1['gt_labels_3d'], 
                [outs_1, outs_2], [aligned_pts_feats_1, aligned_pts_feats_2]]
            # normal loss + feat contrastive loss + obj contrastive loss
            losses_pts = self.pts_bbox_head.lossV4(*loss_inputs, img_metas=img_metas)
        elif self.loss_type == 'V5':
            loss_inputs = [data_1['gt_bboxes_3d'], data_1['gt_labels_3d'], 
                [outs_1, outs_2], [aligned_pts_feats_1, aligned_pts_feats_2]]
            losses_pts = self.pts_bbox_head.lossV5(*loss_inputs, img_metas=img_metas,
                aligned_gt_bboxes_3d=aligned_gt_bboxes_3d)
        losses.update(losses_pts)
        return losses

    # aligned_gt_bboxes_3d[4].bev == aligned_gt_bboxes_3d[0].bev
    # aligned_gt_bboxes_3d[0].bev 
    # gt_bboxes_3d[4].bev == gt_bboxes_3d[0].bev
    # gt_bboxes_3d[0].bev
    # True: aligned_gt_bboxes_3d[0].bev == gt_bboxes_3d[0].bev 
    # aligned_gt_bboxes_3d[4].corners[0]
    # aligned_gt_bboxes_3d[0].corners[0]
    # gt_bboxes_3d[4].corners[0]
    # gt_bboxes_3d[0].corners[0]
    # True: aligned_gt_bboxes_3d[4].corners[0] == gt_bboxes_3d[0].corners[0]
    # aligned_gt_bboxes_3d[4].corners[0] == aligned_gt_bboxes_3d[0].corners[0]
    # aligned_gt_bboxes_3d[4].yaw
    # aligned_gt_bboxes_3d[0].yaw
    # aligned_gt_bboxes_3d[0].height
    # aligned_gt_bboxes_3d[4].height
    # gt_bboxes_3d[0].yaw
    # gt_bboxes_3d[4].yaw
    # gt_bboxes_3d[4].with_yaw
    # gt_bboxes_3d[0].tensor[0]
    # aligned_gt_bboxes_3d[4].tensor[0]
    # 测试： (1-3应当相同)
    # 1.aligned_aug     aligned_gt_bboxes_3d[0].tensor[0]
    # 2.aligned_aug     aligned_gt_bboxes_3d[4].tensor[0]
    # 3.无aug           gt_bboxes_3d[4].tensor[0]
    # 4.强aug           gt_bboxes_3d[0].tensor[0]