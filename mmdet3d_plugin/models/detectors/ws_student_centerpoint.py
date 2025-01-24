# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import functional as F
from copy import deepcopy

from mmdet.models import DETECTORS
from mmdet3d.models import builder
from projects.mmdet3d_plugin.utils.structure_utils import dict_split, weighted_loss

from mmdet3d.models.detectors import CenterPoint

@DETECTORS.register_module()
class WSStudentCenterPoint(CenterPoint):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 unsup_weight=1.0,
                 warmup_iter=0,
                #  obj_feat
                 **kwargs):
        super(WSStudentCenterPoint, self).__init__(
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
        # TODO: 目前收到了不同aug的data，接下来写forward，两个combine，然后分开，做reverse aug，再计算loss
        
        data_1, data_2 = data
        
        stu_data = self.combine_student_data(data_1, data_2)
        
        # stu_data_tag = self.get_data_tag(stu_data)
        points = stu_data['points']
        img_metas = stu_data['img_metas']
        gt_bboxes_3d = stu_data['gt_bboxes_3d']
        gt_labels_3d = stu_data['gt_labels_3d']
        
        img_feats, pts_feats = self.extract_feat(
            points, img=None, img_metas=img_metas)
        
        
        losses = dict()
        # losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                            # gt_labels_3d, img_metas,
                                            # gt_bboxes_ignore=None)
        # outs = self.pts_bbox_head(pts_feats)
        '''
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        # losses_pts = self.pts_bbox_head.loss(*loss_inputs)
        '''
        outs = self.pts_bbox_head.forwardV2(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, pts_feats]
        # normal loss + feat contrastive loss + obj contrastive loss
        losses_pts = self.pts_bbox_head.lossV2(*loss_inputs, img_metas=img_metas)
        losses.update(losses_pts)
        return losses
        
        
        '''
        if "unsup_student" in data_groups:
            unsup_loss = self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"])
            weight = min(self.iter / (self.warmup_iter + 1), 1.0)
            unsup_loss = weighted_loss(
                unsup_loss,
                weight=self.unsup_weight * weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)
        self.iter += 1
        return loss
        '''

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
        
    def split_student_data(self, student_data, student_data_length):
        data_1, data_2 = [], []
        for idx, data in enumerate(student_data):   # len(list) = 6
            data_1.append([{}])
            data_2.append([{}])
            # len(list) = 1
            for key in data[0].keys():
                data_1[idx][0][key] = data[0][key][:student_data_length[0], ...].clone()
                # detach upservised data gradient
                data_2[idx][0][key] = data[0][key][student_data_length[0]:, ...].clone()
        # data_1[0][0]['reg'], data_2[0][0]['reg']
        return tuple(data_1), tuple(data_2)

    def forward_unsup_qfl_train(self, 
                            points,
                            img_metas,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            gt_bboxes_ignore=None):
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.qfl_loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses


    def foward_unsup_train(self, teacher_data, student_data):
        unsup_loss = dict()
        sup_loss_s = self.forward_unsup_qfl_train(**student_data)
        # sup_loss_t = super().forward_train(**teacher_data)

        sup_loss_s = {"s_" + k: v for k, v in sup_loss_s.items()}
        # sup_loss_t = {"t_" + k: v for k, v in sup_loss_t.items()}

        unsup_loss.update(sup_loss_s)
        # unsup_loss.update(sup_loss_t)
        
        # teacher_pts = teacher_data["points"]
        # student_pts = student_data["points"]

        # img_metas_teacher = teacher_data["img_metas"]
        # img_metas_student = student_data["img_metas"]

        # # extract feature and labels of the teacher model
        # img_feats_t, pts_feats_t = self.extract_feat(teacher_pts, 
        #             img=None, img_metas=img_metas_teacher)
        # outs_teacher = self.pts_bbox_head(pts_feats_t)

        # img_feats_s, pts_feats_s = self.extract_feat(student_pts,
        #             img=None, img_metas=img_metas_student)
        # outs_student = self.pts_bbox_head(pts_feats_s)

        # # reverse the points
        # feats_t, preds_t = self.reverse_aug(pts_feats_t, outs_teacher, img_metas_teacher)
        # feats_s, preds_s = self.reverse_aug(pts_feats_s, outs_student, img_metas_student)

        # # dense feature imitation
        # valid_mask = (feats_t != INVALID_VAL) & (feats_s != INVALID_VAL)
        # loss_feats = torch.sum((feats_t - feats_s) ** 2 * valid_mask) / torch.sum(valid_mask)
        
        # # dense label supervision
        # valid_mask = (preds_t != INVALID_VAL) & (preds_s != INVALID_VAL)
        # loss_preds = torch.sum((preds_t - preds_s) ** 2 * valid_mask) / torch.sum(valid_mask)

        return unsup_loss
    
    def reverse_aug(self, img_feats, preds_list, img_metas):
        num_data = len(img_metas)
        reverse_img_list = []
        reverse_pred_list = []
        for num_id in range(num_data):
            img_meta = img_metas[num_id]
            # import pdb
            # pdb.set_trace()
            # img reverse
            img_feat = img_feats[0][num_id]
            reverse_img_list.append(self.reverse_data(img_feat, img_meta))
            # pred reverse
            num_cls = len(preds_list)
            merge_pred_list = []
            for cid in range(num_cls):
                pred_data = preds_list[cid][0]
                pred_merge = torch.cat([pred_data['reg'][num_id], pred_data['height'][num_id], \
                    pred_data['dim'][num_id], pred_data['rot'][num_id], pred_data['heatmap'][num_id]], dim=0)
                reverse_pred_merge = self.reverse_data(pred_merge, img_meta)
                merge_pred_list.append(reverse_pred_merge)
            merge_preds = torch.stack(merge_pred_list)
            reverse_pred_list.append(merge_preds)
        reverse_imgs = torch.stack(reverse_img_list)
        reverse_pred = torch.stack(reverse_pred_list)
        return reverse_imgs, reverse_pred

    def reverse_data(self, img_feat, img_meta):
        pcd_scale_factor = img_meta['pcd_scale_factor']
        pcd_rotation_angle = img_meta['pcd_rotation_angle'] / (2 * np.pi) * 360

        reverse_img_feat = affine(img_feat, angle=-pcd_rotation_angle, translate=(0, 0),
            scale=1/pcd_scale_factor, shear=0, fill=INVALID_VAL)
        if img_meta['pcd_horizontal_flip'] == True:
            reverse_img_feat = hflip(reverse_img_feat)
        if img_meta['pcd_vertical_flip'] == True:
            reverse_img_feat = vflip(reverse_img_feat)
        # reverse_img_feat = img_feat
        return reverse_img_feat
