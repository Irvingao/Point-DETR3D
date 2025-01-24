# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import numpy as np
import pyquaternion
from pyquaternion import Quaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose

# kitti评测增加
# from ..core.bbox import Box3DMode, points_cam2img
from mmcv.utils import print_log


@DATASETS.register_module()
class NuScenesCarEvalDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 speed_mode='abs_dis',
                 max_interval=3,
                 min_interval=0,
                 prev_only=False,
                 next_only=False,
                 test_adj = 'prev',
                 fix_direction=False,
                 test_adj_ids=None,
                 pcd_limit_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 vihicle_names=None):
        self.vihicle_names = vihicle_names

        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.img_info_prototype = img_info_prototype

        self.speed_mode = speed_mode
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.prev_only = prev_only
        self.next_only = next_only
        self.test_adj = test_adj
        self.fix_direction = fix_direction
        self.test_adj_ids = test_adj_ids

        self.pcd_limit_range = pcd_limit_range

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']

        # 将Nus所有和车相关的类别合成一类
        if self.vihicle_names is not None:
            for i in range(len(data_infos)):
                for j in range(len(data_infos[i]['gt_names'])):
                    if data_infos[i]['gt_names'][j] in self.vihicle_names:
                        data_infos[i]['gt_names'][j] = 'car'

        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # exit()
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        # print("info['cams'].keys():", info['cams'].keys())
        # print("info['cams']['CAM_FRONT'].keys():", info['cams']['CAM_FRONT'].keys())
        # raise Exception


        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))
            elif self.img_info_prototype == 'bevdet':
                image_paths = []
                lidar2img_rts = []
                lidar2cam = []
                intrins = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                    lidar2cam.append(lidar2cam_rt.T)   # 为了在kitti上评测加的 
                    intrins.append(viewpad)   # 为了在kitti上评测加的
                    info['cams'][cam_type].update(lidar2cam=lidar2cam_rt.T,
                                                  lidar2img=lidar2img_rt)   # agu外参加的

                # 可视化加的代码
                input_dict.update(
                    dict(
                        filename=image_paths,
                        lidar2img=lidar2img_rts,
                        lidar2ego_translation=info['lidar2ego_translation'],
                        lidar2ego_rotation=info['lidar2ego_rotation']
                    ))

                input_dict.update(dict(img_info=info['cams'],
                                        intrins=intrins,
                                        lidar2cam=lidar2cam))
            elif self.img_info_prototype == 'bevdet_sequential':
                if info ['prev'] is None or info['next'] is None:
                    adjacent= 'prev' if info['next'] is None else 'next'
                else:
                    if self.prev_only or self.next_only:
                        adjacent = 'prev' if self.prev_only else 'next'
                    elif self.test_mode:
                        adjacent = self.test_adj
                    else:
                        adjacent = np.random.choice(['prev', 'next'])
                if type(info[adjacent]) is list:
                    if self.test_mode:
                        if self.test_adj_ids is not None:
                            info_adj=[]
                            select_id = self.test_adj_ids
                            for id_tmp in select_id:
                                id_tmp = min(id_tmp, len(info[adjacent])-1)
                                info_adj.append(info[adjacent][id_tmp])
                        else:
                            select_id = min((self.max_interval+self.min_interval)//2,
                                            len(info[adjacent])-1)
                            info_adj = info[adjacent][select_id]
                    else:
                        if len(info[adjacent])<= self.min_interval:
                            select_id = len(info[adjacent])-1
                        else:
                            select_id = np.random.choice([adj_id for adj_id in range(
                                min(self.min_interval,len(info[adjacent])),
                                min(self.max_interval,len(info[adjacent])))])
                        info_adj = info[adjacent][select_id]
                else:
                    info_adj = info[adjacent]
                # print("info_adj:", type(info_adj))
                # print("info_adj:", info_adj)
                input_dict.update(dict(img_info=info['cams'],
                                       curr=info,
                                       adjacent=info_adj,
                                       adjacent_type=adjacent))

        if True:  # not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.img_info_prototype == 'bevdet_sequential':
                bbox = input_dict['ann_info']['gt_bboxes_3d'].tensor
                if 'abs' in self.speed_mode:
                    bbox[:, 7:9] = bbox[:, 7:9] + torch.from_numpy(info['velo']).view(1,2)
                if input_dict['adjacent_type'] == 'next' and not self.fix_direction:
                    bbox[:, 7:9] = -bbox[:, 7:9]
                if 'dis' in self.speed_mode:
                    time = abs(input_dict['timestamp'] - 1e-6 * input_dict['adjacent']['timestamp'])
                    bbox[:, 7:9] = bbox[:, 7:9] * time
                input_dict['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(bbox,
                                                                              box_dim=bbox.shape[-1],
                                                                             origin=(0.5, 0.5, 0.0))
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results


    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        gt_annos = {}   # 新加入转化waymo gt到nus形式
        ego2global_translation = {} # 新加入保存waymo  ego2global
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            second_annos= []
            # print("boxes:", det['scores_3d'].shape)
            boxes = output_to_nus_box(det)
            gt_boxes, num_pts_list = output_gt_to_nus_box(self.data_infos[sample_id]['gt_boxes'], self.data_infos[sample_id]['gt_names'], self.CLASSES,
                            self.data_infos[sample_id]['num_lidar_pts'], self.data_infos[sample_id]['num_radar_pts'])
            sample_token = self.data_infos[sample_id]['token']
            # print("mapped_class_names:", mapped_class_names)
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs)
            gt_boxes, num_pts_list = lidar_nusc_box_to_global(self.data_infos[sample_id], gt_boxes,
                                    mapped_class_names,
                                    self.eval_detection_configs,
                                    num_pts_list)
            # print("after_box:", boxes)
            # print("gt_boxes:", gt_boxes)
            # exit()
            for i, box in enumerate(boxes):
                # filter 预测不在三个类别里的pred
                if box.label >= len(mapped_class_names):
                    continue
                name = mapped_class_names[box.label]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name='')
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

            # 把gt转为nus格式并写入json文件
            for i, gt_box in enumerate(gt_boxes):
                name = mapped_class_names[gt_box.label]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=gt_box.center.tolist(),
                    size=gt_box.wlh.tolist(),
                    rotation=gt_box.orientation.elements.tolist(),
                    velocity=gt_box.velocity[:2].tolist(),
                    detection_name=name,
                    num_pts=num_pts_list[i],
                    attribute_name='')
                second_annos.append(nusc_anno)
            gt_annos[sample_token] = second_annos

            # 把 ego2global_translation 写入json文件
            ego2global_translation[sample_token] = self.data_infos[sample_id]['ego2global_translation']

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        gt_nusc_submissions = {
            'meta': self.modality,
            'results': gt_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)

        # 转化waymo gt的到一个json文件
        gt_jsonfile_prefix = jsonfile_prefix + '_gt'
        mmcv.mkdir_or_exist(gt_jsonfile_prefix)
        gt_path = osp.join(gt_jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', gt_path)
        mmcv.dump(gt_nusc_submissions, gt_path)

        # 存储 ego2global_translation 到一个json文件
        e2g_jsonfile_prefix = jsonfile_prefix + '_e2gT'
        mmcv.mkdir_or_exist(e2g_jsonfile_prefix)
        e2g_path = osp.join(e2g_jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', e2g_path)
        mmcv.dump(ego2global_translation, e2g_path)

        return res_path, gt_path, e2g_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in Lyft protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from mmdet3d.core.evaluation.like_nus import LikeNuScenesEval

        result_path, gt_path, e2g_path = result_path
        output_dir = osp.join(*osp.split(result_path)[:-1])

        nusc_eval = LikeNuScenesEval(
            config=self.eval_detection_configs,
            result_path=result_path,
            gt_path=gt_path,
            e2g_path=e2g_path,
            output_dir=output_dir,
            verbose=False,
            class_names=self.CLASSES)
        metrics = nusc_eval.main(render_curves=False)

        # record metrics
        # metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                if k not in self.ErrNameMapping.keys():
                    continue
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail


    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        
        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='nus',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in Lyft protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            csv_savepath (str | None): The path for saving csv files.
                It includes the file path and the csv filename,
                e.g., "a/b/filename.csv". If not specified,
                the result will not be converted to csv file.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Evaluation results.
        """
        assert ('nus' in metric or 'bbox' in metric), \
            f'invalid metric {metric}'

        if 'nus' in metric:
            assert isinstance(results, list), 'results must be a list'
            assert len(results) == len(self), (
                'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

            if submission_prefix is None:
                tmp_dir = tempfile.TemporaryDirectory()
                submission_prefix = osp.join(tmp_dir.name, 'results')
            else:
                tmp_dir = None

            # currently the output prediction results could be in two formats
            # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
            # 2. list of dict('pts_bbox' or 'img_bbox':
            #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
            # this is a workaround to enable evaluation of both formats on nuScenes
            # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
            if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
                result_files = self._format_bbox(results, submission_prefix)
            else:
                # 从这走
                # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
                result_files = dict()
                for name in results[0]:
                    print(f'\nFormating bboxes of {name}')
                    results_ = [out[name] for out in results]
                    tmp_file_ = osp.join(submission_prefix, name)
                    result_files.update(
                        {name: self._format_bbox(results_, tmp_file_)})

            if isinstance(result_files, dict):
                ap_dict = dict()
                result_names=['pts_bbox']
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                ap_dict.update(ret_dict)
            elif isinstance(result_files, str):
                ap_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return ap_dict


    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)


def output_gt_to_nus_box(gt, name, CLASSES, num_lidar_pts, num_radar_pts):
    """
    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    from nuscenes.utils.data_classes import Box as NuScenesBox

    box_gravity_center = gt[:, :3]
    box_dims = gt[:, 3:6]
    box_yaw = gt[:, 6]

    box_yaw = -box_yaw - np.pi / 2 

    velocity_all = torch.zeros((len(gt), 2))

    box_list = []
    num_pts_list = []
    for i in range(len(gt)):
        if name[i] not in CLASSES:
            continue
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*velocity_all[i,:], 0.0)

        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=np.array([CLASSES.index(name[i])], dtype='int32'),
            velocity=velocity)
        box_list.append(box)
        num_pts_list.append(num_lidar_pts[i]+num_radar_pts[i])
    return box_list, num_pts_list

def output_to_nus_box(detection):
    """Convert the output to the box class in the Lyft.

    Args:
        detection (dict): Detection results.

    Returns:
        list[:obj:`LyftBox`]: List of standard LyftBoxes.
    """
    from nuscenes.utils.data_classes import Box as NuScenesBox

    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    if box3d.tensor.size(-1) != 9:
        velocity_all = torch.zeros((box3d.tensor.size(0), 2))
    else:
        velocity_all = box3d.tensor[:, 7:9]

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*velocity_all[i,:], 0.0)

        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(info, boxes, classes, eval_configs, num_pts=None):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`LyftBox`]): List of predicted LyftBoxes.

    Returns:
        list: List of standard LyftBoxes in the global
            coordinate.
    """
    box_list = []
    num_pts_list = []
    if num_pts is None:
        for i, box in enumerate(boxes):
            # Move box to ego vehicle coord system
            box.rotate(Quaternion(info['lidar2ego_rotation']))
            box.translate(np.array(info['lidar2ego_translation']))
            # filter det in ego.
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            if box.label >= len(classes):
                continue
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
            # Move box to global coord system
            box.rotate(Quaternion(info['ego2global_rotation']))
            box.translate(np.array(info['ego2global_translation']))
            box_list.append(box)
        return box_list
    else:
        for i, box in enumerate(boxes):
            # Move box to ego vehicle coord system
            box.rotate(Quaternion(info['lidar2ego_rotation']))
            box.translate(np.array(info['lidar2ego_translation']))
            # filter det in ego.
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            if box.label >= len(classes):
                continue
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
            # Move box to global coord system
            box.rotate(Quaternion(info['ego2global_rotation']))
            box.translate(np.array(info['ego2global_translation']))
            box_list.append(box)
            num_pts_list.append(num_pts[i])
        return box_list, num_pts_list
        