import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16

from mmdet.core import multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models.utils.transformer import DeformableDetrTransformer, DetrTransformerEncoder
from ..utils.point_encoder import PointEncoderV2,PointEncoderV3,Hybrid_PointEncoder
from ..utils.group_point_encoder import GroupPointEncoder

@HEADS.register_module()
class GropuPointDGCNN3DHead(DETRHead):
    """Head of DeformDETR3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 num_group=1,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.voxel_size = self.bbox_coder.voxel_size

        self.bev_shape = (int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]),
                          int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]))

        self.num_cls_fcs = num_cls_fcs - 1
        self.num_group = num_group
        super(GropuPointDGCNN3DHead, self).__init__(
            *args, transformer=transformer, **kwargs)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats,point_coord,labels):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is Ture it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is Ture it would be returned, otherwise \
                `None` would be returned.
        """

        # [4,bs,256,256,256]
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = self.bev_shape
        img_masks = mlvl_feats[0].new_zeros(
            (batch_size, input_img_h, input_img_w))

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))


        pointencoder = GroupPointEncoder(10, 128,point_coord,labels,self.num_group)
        query_embeds,group_point_coord,group_labels = pointencoder.forward(point_coord, labels, self.pc_range)


        # query_embeds = None
        # if not self.as_two_stage:
        #     # [300,512]
        #     query_embeds = self.query_embedding.weight

        hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            group_point_coord,
            group_labels,
            self.num_group,
            self.pc_range,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
        )
        hs = hs.permute(0, 2, 1, 3)  # [num_layer,num_query,bs*num_group,256]  --> [num_layer,bs*num_group,num_query,256]
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            # 将reference从0-1 转到 BEV的尺度下
            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., 0:2] += reference
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                # 在记录最后输出的坐标之前，先x,y坐标转到BEV尺度下
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])

            if tmp.size(-1) > 8:
                outputs_coord = torch.cat((tmp[..., :6], tmp[..., 6:8], tmp[..., 8:]), -1)
            else:
                outputs_coord = torch.cat((tmp[..., :6], tmp[..., 6:8]), -1)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        if self.as_two_stage:
            outs = {
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': enc_outputs_class,
                'enc_bbox_preds': enc_outputs_coord.sigmoid(),
            }
        else:
            outs = {
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
            }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets  默认10都是背景类
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        # 只有将分为正样本的query 将其赋值为gt对应的label
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :self.code_size - 1]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR  bbox_targets只有将分为正样本的预测值,赋值为gt,作为targets,其余都是0
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # 其实就是bs
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # ------------------ new_get_targets---------------------------
        # cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
        #                                    gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)

        cls_reg_targets = self.new_get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # 将不同batch的结果cat在一起,方便后面一起计算
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            # reduce_mean:Obtain the mean of tensor on different GPUs
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        # cls_scores:[300*2,10]  labels:[300*2] 这里面预测和gt都是一一对应的
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # bbox_targets中被分为背景的预测值都是0,在normalize_bbox中求log之后就会变成负无穷
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        # 只计算分为正样本的 预测值的loss
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :8], normalized_bbox_targets[isnotnan, :8], bbox_weights[isnotnan, :8],
            avg_factor=num_total_pos)
        # 计算速度的loss
        if self.code_size > 8:
            loss_bbox_vel = self.loss_bbox(
                bbox_preds[isnotnan, 8:], normalized_bbox_targets[isnotnan, 8:], bbox_weights[isnotnan, 8:],
                avg_factor=num_total_pos)
            loss_bbox = loss_bbox + loss_bbox_vel * 0.2

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        # [6,bs,300,10]
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        # [6,2,num_gt,9]  重复六份
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        # [6,bs,num_gt,1]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]


        # -----------new_loss_single------------------
        # losses_cls, losses_bbox = multi_apply(
        #     self.loss_single, all_cls_scores, all_bbox_preds,
        #     all_gt_bboxes_list, all_gt_labels_list,
        #     all_gt_bboxes_ignore_list)

        losses_cls, losses_bbox = multi_apply(
            self.new_loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)
        # losses_cls, losses_bbox = multi_apply(
        #     self.group_loss_single, all_cls_scores, all_bbox_preds,
        #     all_gt_bboxes_list, all_gt_labels_list,
        #     all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            if bboxes.size(-1) == 9:
                bboxes = img_metas[i]['box_type_3d'](bboxes, 9)  # 将bbox实例化为LiDARInstance3DBoxes类
            else:
                bboxes = img_metas[i]['box_type_3d'](bboxes, 7)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list


    def new_get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        # ---------------------去掉补上的部分--------------------------
        bs = len(gt_bboxes_list)
        positive_num = [len(gt_bboxes_list[idx]) for idx in range(bs)]
        padding_num = [max(positive_num) - positive_num[idx] for idx in range(bs)]
        for idx in range(num_imgs):
            if padding_num[idx%bs] == 0:
                continue
            else:
                bbox_preds_list[idx] = bbox_preds_list[idx][:max(positive_num)-padding_num[idx%bs],:]
                cls_scores_list[idx] = cls_scores_list[idx][:max(positive_num)-padding_num[idx%bs],:]
        # ------------------------------------------------------------

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def new_loss_single(self,
                        cls_scores,
                        bbox_preds,
                        gt_bboxes_list,
                        gt_labels_list,
                        gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # 其实就是bs
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # ------------------ new_get_targets---------------------------
        # cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
        #                                    gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)

        cls_reg_targets = self.new_get_targets(cls_scores_list, bbox_preds_list,
                                               gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # 将不同batch的结果cat在一起,方便后面一起计算
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # -----------------去掉补上的部分-------------------------
        bs = len(gt_bboxes_list)
        positive_num = [len(gt_bboxes_list[idx]) for idx in range(bs)]
        padding_num = [max(positive_num) - positive_num[idx] for idx in range(bs)]
        new_bbox_pred = []
        new_cls_scores = []
        for idx in range(bs):
            if padding_num[idx] == 0:
                new_cls_scores.append(cls_scores[idx])
                new_bbox_pred.append(bbox_preds[idx])
                continue
            else:
                new_cls_scores.append(cls_scores[idx, :max(positive_num) - padding_num[idx], :])
                new_bbox_pred.append(bbox_preds[idx, :max(positive_num) - padding_num[idx], :])
        new_cls_scores = torch.cat(new_cls_scores, dim=0)
        new_bbox_pred = torch.cat(new_bbox_pred, dim=0)
        # ------------------------------------------------------

        # classification loss
        # cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            # reduce_mean:Obtain the mean of tensor on different GPUs
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        # cls_scores:[300*2,10]  labels:[300*2] 这里面预测和gt都是一一对应的
        loss_cls = self.loss_cls(
            new_cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        # bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # bbox_targets中被分为背景的预测值都是0,在normalize_bbox中求log之后就会变成负无穷
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        # 只计算分为正样本的 预测值的loss
        loss_bbox = self.loss_bbox(
            new_bbox_pred[isnotnan, :8], normalized_bbox_targets[isnotnan, :8], bbox_weights[isnotnan, :8],
            avg_factor=num_total_pos)
        # 计算速度的loss
        if self.code_size > 8:
            loss_bbox_vel = self.loss_bbox(
                new_bbox_pred[isnotnan, 8:], normalized_bbox_targets[isnotnan, 8:], bbox_weights[isnotnan, 8:],
                avg_factor=num_total_pos)
            loss_bbox = loss_bbox + loss_bbox_vel * 0.2

        return loss_cls, loss_bbox


    def group_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # 其实就是bs
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        bs = len(gt_bboxes_list)
        num_group = int(num_imgs / bs)

        positive_num = [len(gt_bboxes_list[idx]) for idx in range(bs)]
        padding_num = [max(positive_num) - positive_num[idx] for idx in range(bs)]

        group_gt_bboxes_list = []
        group_gt_labels_list = []
        for idx in range(num_group):
            for batch_id in range(bs):
                group_gt_bboxes_list.append(gt_bboxes_list[batch_id])
                group_gt_labels_list.append(gt_labels_list[batch_id])

        cls_reg_targets = self.new_get_targets(cls_scores_list, bbox_preds_list,
                                           group_gt_bboxes_list, group_gt_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # 将不同batch的结果cat在一起,方便后面一起计算
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # 去掉补上的部分
        cls_scores = cls_scores.view(num_group,bs,-1,10)
        bbox_preds = bbox_preds.view(num_group,bs,-1,10)
        new_bbox_pred = []
        new_cls_scores = []
        for group_id in range(num_group):
            for batch_id in range(bs):
                if padding_num[batch_id]==0:
                    new_cls_scores.append(cls_scores[group_id][batch_id])
                    new_bbox_pred.append(bbox_preds[group_id][batch_id])
                    continue
                else:
                    new_cls_scores.append(cls_scores[group_id][batch_id, :max(positive_num) - padding_num[batch_id], :])
                    new_bbox_pred.append(bbox_preds[group_id][batch_id, :max(positive_num) - padding_num[batch_id], :])
        new_cls_scores = torch.cat(new_cls_scores,dim=0)
        new_bbox_pred = torch.cat(new_bbox_pred,dim=0)

        # classification loss
        # cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            # reduce_mean:Obtain the mean of tensor on different GPUs
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        # cls_scores:[300*2,10]  labels:[300*2] 这里面预测和gt都是一一对应的
        loss_cls = self.loss_cls(
            new_cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        # bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # bbox_targets中被分为背景的预测值都是0,在normalize_bbox中求log之后就会变成负无穷
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        # 只计算分为正样本的 预测值的loss
        loss_bbox = self.loss_bbox(
            new_bbox_pred[isnotnan, :8], normalized_bbox_targets[isnotnan, :8], bbox_weights[isnotnan, :8],
            avg_factor=num_total_pos)
        # 计算速度的loss
        if self.code_size > 8:
            loss_bbox_vel = self.loss_bbox(
                new_bbox_pred[isnotnan, 8:], normalized_bbox_targets[isnotnan, 8:], bbox_weights[isnotnan, 8:],
                avg_factor=num_total_pos)
            loss_bbox = loss_bbox + loss_bbox_vel * 0.2

        return loss_cls, loss_bbox