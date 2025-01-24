# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import functional as F
from copy import deepcopy

from mmdet.models import DETECTORS
from mmdet3d.models import builder
from projects.mmdet3d_plugin.utils.structure_utils import dict_split, weighted_loss

from mmdet3d.models.detectors import CenterPoint

@DETECTORS.register_module()
class WSStudentCenterPointV2(CenterPoint):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 unsup_weight=1.0,
                 warmup_iter=0,
                 **kwargs):
        super(WSStudentCenterPointV2, self).__init__(
            **kwargs)
        self.unsup_weight = unsup_weight
        self.warmup_iter = warmup_iter
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
        # TODO: 只对view_1反传，view_2 infer无梯度
        
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
        aligned_pts_feats_1 = torch.stack(aligned_pts_feats[:num])
        aligned_pts_feats_2 = torch.stack(aligned_pts_feats[num:])
        
        # head forward
        # view_1
        outs_1 = self.pts_bbox_head.forwardV2([aligned_pts_feats_1])
        # view_2
        with torch.no_grad():
            outs_2 = self.pts_bbox_head.forwardV2([aligned_pts_feats_2])
        
        losses = dict()

        loss_inputs = [aligned_gt_bboxes_3d[:num], gt_labels_3d[:num], 
                       [outs_1, outs_2], [aligned_pts_feats_1, aligned_pts_feats_2]]
        # normal loss + feat contrastive loss + obj contrastive loss
        losses_pts = self.pts_bbox_head.lossV3(*loss_inputs, img_metas=img_metas)
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
    # 测试：
    # gt_bboxes_3d[4].tensor[0] == gt_bboxes_3d[0].tensor[0]
    # aligned_gt_bboxes_3d[0].tensor[0] == aligned_gt_bboxes_3d[4].tensor[0]
    

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses
        