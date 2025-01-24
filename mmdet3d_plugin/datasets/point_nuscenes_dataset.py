import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from os import path as osp
import mmcv

import os
import shutil
from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box
import pyquaternion

@DATASETS.register_module()
class PointNuScenesDataset(NuScenesDataset):
    
    def __init__(self, gt_ratio=None, load_interval=1, without_vel=False, verbose=False, **kwargs):
        self.gt_ratio = gt_ratio
        if self.gt_ratio:
            assert load_interval == 1
        super(PointNuScenesDataset,self).__init__(load_interval=load_interval, **kwargs)
        self.without_vel = without_vel
        self.verbose = verbose

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        
        if self.load_interval > 1:
            data_infos = data_infos[::self.load_interval]
        if self.gt_ratio:
            num_samples = len(data_infos)
            selected_num_samples = int(self.gt_ratio * num_samples)
            data_infos = data_infos[:selected_num_samples]
            
        self.metadata = data['metadata']
        self.version = self.metadata['version']
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
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
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

        # if not self.test_mode:
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        return input_dict
    '''
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

        # the nuscenes box center is [0.5, 0.5, 0.], we change it to be
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
    '''

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

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
        from nuscenes import NuScenes
        # from nuscenes.eval.detection.evaluate import NuScenesEval
        from projects.mmdet3d_plugin.datasets.nuscenes_utils.nuscenes_eval import NuScenesEval
        

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            # 'v1.0-mini': 'mini_train',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            fix_det_range=self.fix_det_range,
            output_dir=output_dir,
            verbose=self.verbose)
        nusc_eval.main(render_curves=False, SPNDS=self.without_vel)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
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
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        if self.without_vel:
            detail['{}/SPNDS'.format(metric_prefix)] = metrics['spnd_score']
        # Static Properties
        return detail
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 fix_det_range=None,    # add
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        self.fix_det_range = fix_det_range
        # self.fix_det_range = None
        # for k in list(self.eval_detection_configs.class_range.keys()):
            # self.eval_detection_configs.class_range[k] = fix_det_range[1]   # max
        
        # save_path = "/".join(os.path.abspath(__file__).split("/")[:-4])
        
        # if save_res_json_file is not None:
            # json_dir = os.path.join(save_path, save_res_json_file)
        # else:
            # json_dir = None
        
        # if results is not None:
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        # if json_dir is not None:
        #     shutil.copyfile(result_files['pts_bbox'], json_dir)
        # else:
            # result_files = dict(pts_bbox=json_dir)
            # tmp_dir = None
        # os.path.isfile(result_files['pts_bbox']) 
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict
    
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
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            # print(f"lidar_nusc_box_to_global: {self.fix_det_range}")
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.fix_det_range,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
    
    
    def show_multiview(self, index):
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
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        import cv2
        import copy
        from projects.mmdet3d_plugin.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
        anno_info = self.get_ann_info(index)
        gt_bboxes = anno_info['gt_bboxes_3d']
        cam_imgs = []
        for cam_type, cam_info in info['cams'].items():
            # img
            img_path = cam_info['data_path']
            img = mmcv.imread(img_path)
            
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

            gt_bbox_color=(61, 102, 255)
            if len(gt_bboxes) != 0:
                img = draw_lidar_bbox3d_on_img(
                    copy.deepcopy(gt_bboxes), img, lidar2img_rt, None, color=gt_bbox_color,thickness=3)
            cam_imgs.append(img)
        img_size = (1600,2400,3)
        pano = np.zeros(img_size, np.uint8)

        cam1 = cam_imgs[0]
        mmcv.imwrite(cam1, 'multiview_1.png')
        cam1 = cv2.resize(cam1, (800,400))
        pano[:400,800:1600]=cam1

        cam2 = cam_imgs[2]
        cam2 = cv2.resize(cam2, (800,400))
        pano[400:800,:800]=cam2

        cam3 = cam_imgs[4]
        cam3 = cv2.resize(cam3, (800,400))
        pano[800:1200,:800]=cam3

        cam4 = cam_imgs[3]
        cam4 = cv2.resize(cam4, (800,400))
        pano[-400:,800:1600]=cam4

        cam5 = cam_imgs[5]
        cam5 = cv2.resize(cam5, (800,400))
        pano[800:1200,-800:]=cam5

        cam6 = cam_imgs[1]
        cam6 = cv2.resize(cam6, (800,400))
        pano[400:800,-800:]=cam6
        # save_path_list = file_name.split('__')
        # save_path = osp.join(out_dir, save_path_list[0] + '_' + save_path_list[-1] + '.png')
        if img is not None:
            # mmcv.imwrite(pano, save_path)
            mmcv.imwrite(pano, 'multiview.png')
            
    def show(self, results, out_dir, show=True, pipeline=None, save_show=False):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
        from projects.mmdet3d_plugin.core.visualizer.show_result import show_result
        
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        print(f"data num: {len(results)}")
        jump_idx = 133
        for i, result in enumerate(results):
            if i < jump_idx:
                continue
            print(f"frame idx: {i}")
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.0
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH) # (N_gt, 9)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH) # (N_pred, 9)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show, save_show)

    def show_point_anno(self, results, out_dir, show=True, pipeline=None, save_show=False, point_anno=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
        from projects.mmdet3d_plugin.core.visualizer.show_result import show_point_result, show_multiview
        
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        print(f"data num: {len(results)}")
        jump_idx = 100 # 3924
        draw_pred = False
        for i, result in enumerate(results):
            if i < jump_idx:
                continue
            print(f"frame idx: {i}")
            # image
            show_multiview(info=self.data_infos[i],anno_info=self.get_ann_info(i),draw_pred=draw_pred)
            input("pause")
            # lidar
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            # lidar point
            points = self._extract_data(i, pipeline, 'points').numpy()
            # point anno
            point_coord = self._extract_data(i, pipeline, 'point_coord').numpy()
            # point_coord = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()[...,:3]
            point_coord = Coord3DMode.convert_point(point_coord, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            labels = self._extract_data(i, pipeline, 'labels').numpy()
            
            #  'point_coord', 'labels'
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.0
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH) # (N_gt, 9)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH) # (N_pred, 9)
            '''
            show_point_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        str(i), show, save_show, point_anno=(point_coord, labels))
            '''

    
def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             fix_det_range=None,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.
    
    Add range filter!!!

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        ''
        if fix_det_range:
            min_range = fix_det_range[0]
            max_range = fix_det_range[1]
            # print(f"fix_det_range: {fix_det_range}")
            if radius > max_range:
                continue
            if radius < min_range:
                continue
        else:
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list