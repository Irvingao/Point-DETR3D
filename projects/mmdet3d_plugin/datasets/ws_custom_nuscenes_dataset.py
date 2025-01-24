import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.datasets import NuScenesDataset
from os import path as osp
import mmcv

import os
import shutil
from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box
import pyquaternion

@DATASETS.register_module()
class WSStudentSampledNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """
    def __init__(self, gt_ratio=None, load_interval=1, without_vel=False, verbose=False, **kwargs):
        self.gt_ratio = gt_ratio
        if self.gt_ratio:
            assert load_interval == 1
        super(WSStudentSampledNuScenesDataset,self).__init__(load_interval=load_interval, **kwargs)
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
        # if self.load_ratio:
        #     num_samples = len(data_infos)
        #     selected_num_samples = int(self.load_ratio * num_samples)
        #     data_infos = data_infos[:selected_num_samples]
        num_samples = len(data_infos)
        self.num_gt_samples = num_samples
        if self.gt_ratio:
            self.num_gt_samples = int(self.gt_ratio * num_samples)
        
            
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        # add gt/pseduo tag
        if index < self.num_gt_samples:
            input_dict['label_flag'] = "gt"
        else:
            input_dict['label_flag'] = "pseduo"
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if isinstance(example, list):
            for exam in example:
                if self.filter_empty_gt and \
                        (exam is None or
                            ~(exam['gt_labels_3d']._data != -1).any()):
                    return None
        else:
            if self.filter_empty_gt and \
                    (example is None or
                        ~(example['gt_labels_3d']._data != -1).any()):
                return None
        
        example = dict(data=example, img_metas=example[0]['img_metas']) # `img_metas`` is not used here. 
        
        return example

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
