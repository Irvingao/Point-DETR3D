# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys

import mmcv
from mmcv import Config

from mmdet3d.datasets import build_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) # insert project abs path
print(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the gt')
    # parser.add_argument('config', help='test config file path')
    parser.add_argument('--config', help='test config file path',
        default="projects/configs/dgcnn_futr3d/pillar_futr3dv4_C_L_DeformRoiWiseAttn.py")
    parser.add_argument('--result', help='results file in pickle format',
                        default="tec_voxel_01_res.pkl")
                        # default="/dataset/nuscenes_mini/nuscenes_infos_train.pkl")
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved',
        default="vis_save_dir/")
    parser.add_argument(
        '--save_pic', type=bool, 
        default=True)
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('The results file must be a pkl file.')
    
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)


    # build the dataset
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)
    
    
    print("load pkl successfully.")

    if getattr(dataset, 'show', None) is not None:
        # data loading pipeline for showing
        eval_pipeline = cfg.get('eval_pipeline', {})
        if eval_pipeline:
            dataset.show(results, args.show_dir, pipeline=eval_pipeline, save_show=args.save_pic)
        else:
            dataset.show(results, args.show_dir, save_show=args.save_pic)  # use default pipeline
    else:
        raise NotImplementedError(
            'Show is not implemented for dataset {}!'.format(
                type(dataset).__name__))


if __name__ == '__main__':
    main()
