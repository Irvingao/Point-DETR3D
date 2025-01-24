# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import trimesh
from os import path as osp

# from mmdet3d.core.visualizer.open3d_vis import Visualizer, _draw_points
from projects.mmdet3d_plugin.core.visualizer.open3d_vis import Visualizer, _draw_points
from mmdet3d.core.visualizer.image_vis import (draw_camera_bbox3d_on_img, draw_depth_bbox3d_on_img,
                        draw_lidar_bbox3d_on_img)


def show_result(points,
                gt_bboxes,
                pred_bboxes,
                out_dir,
                filename,
                show=True,
                snapshot=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool): Visualize the results online. Defaults to False.
        snapshot (bool): Whether to save the online results. Defaults to False.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:

        vis = Visualizer(points)
        if pred_bboxes is not None:
            vis.add_bboxes(bbox3d=pred_bboxes, bbox_color=(0, 0, 1),points_in_box_color=(0,0,0))
        if gt_bboxes is not None:
            vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 1, 0),points_in_box_color=(0,0,0))
        
        # vis.add_seg_mask(points)
        
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        
        vis.show(show_path)
        print(show_path)
        # if pred_bboxes is not None:
        #     vis.add_bboxes(bbox3d=pred_bboxes,bbox_color=(1,0,0),points_in_box_color=(1,0,0)) # 红色
        # if gt_bboxes is not None:
        #     vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1),points_in_box_color=(0,0,1)) # 蓝色

 


from projects.mmdet3d_plugin.datasets.nuscenes_utils.statistics_data import gtlabels2names, color_dict
from .color_map import *
def show_point_result(points,
                    gt_bboxes,
                    pred_bboxes,
                    out_dir,
                    filename,
                    show=True,
                    snapshot=False,
                    point_anno=None,
                    color_map="gradient",     # gradient, circle
                    color_mode="xyzrgb"):   # xyzrgb, xyz
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool): Visualize the results online. Defaults to False.
        snapshot (bool): Whether to save the online results. Defaults to False.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    point_color = (0.8,0.8,0.8)
    black = (0,0,0)
    white = (1,1,1)
    if color_map == "gradient":
        points = gradient_point_cloud_color_map(points)
    elif color_map == "circle":
        points = circle_distance_point_cloud_color_map(points)
    else:
        raise ValueError(f"{color_map} is not supported.")
    
    if show:
        vis = Visualizer(points,point_color=point_color, mode=color_mode,background_color=white,points_size=2)
        if pred_bboxes is not None:
            vis.add_bboxes(bbox3d=pred_bboxes, bbox_color=(1, 0, 0),points_in_box_color=point_color)    # ,points_in_box_color=(1,1,1)
        if gt_bboxes is not None:
            vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 0),points_in_box_color=point_color)  # ,points_in_box_color=(1,1,1)
        # if point_anno is not None:
        #     for idx, (point_coord, label) in enumerate(zip(point_anno[0], point_anno[1])):
        #         pts, color = _draw_points(np.expand_dims(point_coord, axis=0), vis.o3d_visualizer, points_size=5, 
        #                     point_color=color_dict[gtlabels2names[int(label)]],mode="xyz")
        #         vis.pcd += pts
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        
        vis.show(show_path)
        # print(show_path)

def show_multiview(info, anno_info, draw_pred=True):
    # info = self.data_infos[index]
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
    # from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
    # anno_info = self.get_ann_info(index)
    gt_bboxes = anno_info['gt_bboxes_3d']
    i=0
    cam_imgs = []
    for cam_type, cam_info in info['cams'].items():
        i += 1
        # img
        img_path = cam_info['data_path']
        img = mmcv.imread(img_path)
        
        mmcv.imwrite(img, f'multiview_clear_{i}.png')
        
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

        gt_bbox_color=(61, 102, 255)    # orange
        # gt_bbox_color=(0, 165, 255)    # gloden yellow
        if len(gt_bboxes) != 0 and draw_pred:
            img = draw_lidar_bbox3d_on_img(
                copy.deepcopy(gt_bboxes), img, lidar2img_rt, None, color=gt_bbox_color,thickness=2)
        cam_imgs.append(img)
    img_size = (800,2400,3)
    pano = np.zeros(img_size, np.uint8)

    cam1 = cam_imgs[0]
    mmcv.imwrite(cam1, 'multiview_det_1.png')
    cam1 = cv2.resize(cam1, (800,400))
    pano[:400,800:1600]=cam1

    # 左侧
    cam2 = cam_imgs[2]
    mmcv.imwrite(cam2, 'multiview_det_3.png')
    cam2 = cv2.resize(cam2, (800,400))
    pano[:400,:800]=cam2

    cam3 = cam_imgs[4]
    mmcv.imwrite(cam3, 'multiview_det_5.png')
    cam3 = cv2.resize(cam3, (800,400))
    pano[-400:,:800]=cam3

    # 后视
    cam4 = cam_imgs[3]
    mmcv.imwrite(cam4, 'multiview_det_4.png')
    cam4 = cv2.resize(cam4, (800,400))
    pano[-400:,800:1600]=cam4

    # 右侧
    cam5 = cam_imgs[5]
    mmcv.imwrite(cam5, 'multiview_det_6.png')
    cam5 = cv2.resize(cam5, (800,400))
    pano[:400,-800:]=cam5

    cam6 = cam_imgs[1]
    mmcv.imwrite(cam6, 'multiview_det_2.png')
    cam6 = cv2.resize(cam6, (800,400))
    pano[-400:,-800:]=cam6
    # save_path_list = file_name.split('__')
    # save_path = osp.join(out_dir, save_path_list[0] + '_' + save_path_list[-1] + '.png')
    if img is not None:
        # mmcv.imwrite(pano, save_path)
        mmcv.imwrite(pano, 'cam_multiview.png')



def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode='lidar',
                               img_metas=None,
                               show=True,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str): Coordinate system the boxes are in. Should be one of
           'depth', 'lidar' and 'camera'. Defaults to 'lidar'.
        img_metas (dict): Used in projecting depth bbox.
        show (bool): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
        pred_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    elif box_mode == 'lidar':
        draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        show_img = img.copy()
        if gt_bboxes is not None:
            show_img = draw_bbox(
                gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
        if pred_bboxes is not None:
            show_img = draw_bbox(
                pred_bboxes,
                show_img,
                proj_mat,
                img_metas,
                color=pred_bbox_color)
        mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)

    if img is not None:
        mmcv.imwrite(img, osp.join(result_path, f'{filename}_img.png'))

    if gt_bboxes is not None:
        gt_img = draw_bbox(
            gt_bboxes, img, proj_mat, img_metas, color=gt_bbox_color)
        mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_gt.png'))

    if pred_bboxes is not None:
        pred_img = draw_bbox(
            pred_bboxes, img, proj_mat, img_metas, color=pred_bbox_color)
        mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred.png'))
